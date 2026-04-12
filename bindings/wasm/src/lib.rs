use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use hexchess_engine::board::{self, PieceKind};
use hexchess_engine::game::GameState;
use hexchess_engine::inference::TractEvaluator;
use hexchess_engine::mcts::{DirichletConfig, HeuristicEvaluator, MctsSearch};
use hexchess_engine::movegen::{self, Move as EngineMove};
use hexchess_engine::serialization;

// ---------------------------------------------------------------------------
// Tsify-typed value objects
// ---------------------------------------------------------------------------

/// A move on the hex board.
#[derive(Tsify, Serialize, Deserialize, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct JsMove {
    pub from_q: i8,
    pub from_r: i8,
    pub to_q: i8,
    pub to_r: i8,
    pub promotion: Option<String>,
    /// Glinski notation like "f5-f6" or "f10-f11=Q".
    pub notation: String,
}

impl JsMove {
    fn from_engine(mv: &EngineMove) -> Self {
        Self {
            from_q: mv.from.q,
            from_r: mv.from.r,
            to_q: mv.to.q,
            to_r: mv.to.r,
            promotion: mv.promotion.map(|k| k.as_str().to_string()),
            notation: mv.to_notation().unwrap_or_default(),
        }
    }
}

/// A piece on the board at a given cell.
#[derive(Tsify, Serialize, Deserialize, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct JsPiece {
    pub q: i8,
    pub r: i8,
    pub piece: String,
    pub color: String,
    pub square: String,
}

/// MCTS search result.
#[derive(Tsify, Serialize, Deserialize, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct JsMctsResult {
    pub best_move: JsMove,
    pub value: f32,
    pub nodes: u32,
    pub policy: Vec<f32>,
}

/// Transposition-table stats.
#[derive(Tsify, Serialize, Deserialize, Clone, Copy)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct JsTtStats {
    pub hits: u64,
    pub misses: u64,
    pub clears: u64,
    pub current_size: usize,
}

// ---------------------------------------------------------------------------
// Game class
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct Game {
    state: GameState,
}

#[wasm_bindgen]
#[allow(clippy::new_without_default)]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Game {
        Game {
            state: GameState::new(),
        }
    }

    /// Deep copy of this game.
    #[wasm_bindgen(js_name = "clone")]
    pub fn clone_game(&self) -> Game {
        Game {
            state: self.state.clone(),
        }
    }

    /// All legal moves in the current position.
    #[wasm_bindgen(js_name = "legalMoves")]
    pub fn legal_moves(&self) -> Vec<JsMove> {
        self.state
            .legal_moves()
            .iter()
            .map(JsMove::from_engine)
            .collect()
    }

    /// Apply a move given as `(from_q, from_r, to_q, to_r, promotion?)`.
    /// Prefer `apply(move)` or `applyNotation("f5-f6")` in new code.
    #[wasm_bindgen(js_name = "applyMove")]
    pub fn apply_move(
        &mut self,
        from_q: i8,
        from_r: i8,
        to_q: i8,
        to_r: i8,
        promotion: Option<String>,
    ) -> Result<(), JsError> {
        let promo_kind = parse_promotion(promotion.as_deref())?;
        let from = board::HexCoord::new(from_q, from_r);
        let to = board::HexCoord::new(to_q, to_r);
        self.apply_resolved(from, to, promo_kind)
    }

    /// Apply a `Move` value as returned by `legalMoves()`.
    #[wasm_bindgen(js_name = "apply")]
    pub fn apply(&mut self, mv: JsMove) -> Result<(), JsError> {
        let promo_kind = parse_promotion(mv.promotion.as_deref())?;
        let from = board::HexCoord::new(mv.from_q, mv.from_r);
        let to = board::HexCoord::new(mv.to_q, mv.to_r);
        self.apply_resolved(from, to, promo_kind)
    }

    /// Apply a move given in Glinski notation, e.g. `"f5-f6"` or `"f10-f11=Q"`.
    #[wasm_bindgen(js_name = "applyNotation")]
    pub fn apply_notation(&mut self, notation: &str) -> Result<(), JsError> {
        let (from, to, promo) = EngineMove::parse_notation(notation)
            .ok_or_else(|| JsError::new(&format!("invalid move notation: {notation}")))?;
        self.apply_resolved(from, to, promo)
    }

    /// Undo the last move.
    #[wasm_bindgen(js_name = "undoMove")]
    pub fn undo_move(&mut self) -> Result<(), JsError> {
        if self.state.move_count() == 0 {
            return Err(JsError::new("no moves to undo"));
        }
        self.state.undo_move();
        Ok(())
    }

    /// Game status string: "ongoing", "checkmate_white", "checkmate_black",
    /// "stalemate", "draw_repetition", "draw_fifty", "draw_material".
    #[wasm_bindgen(js_name = "status")]
    pub fn status(&self) -> String {
        self.state.status().as_str().into()
    }

    #[wasm_bindgen(js_name = "isGameOver")]
    pub fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    /// `"white"` or `"black"`.
    #[wasm_bindgen(js_name = "sideToMove")]
    pub fn side_to_move(&self) -> String {
        self.state.side_to_move().as_str().into()
    }

    /// Number of half-moves played.
    #[wasm_bindgen(js_name = "moveCount")]
    pub fn move_count(&self) -> usize {
        self.state.move_count()
    }

    /// All pieces currently on the board.
    #[wasm_bindgen(js_name = "boardState")]
    pub fn board_state(&self) -> Vec<JsPiece> {
        self.state
            .board
            .cells
            .iter()
            .enumerate()
            .filter_map(|(idx, cell)| {
                cell.map(|piece| {
                    let coord = board::index_to_coord(idx);
                    JsPiece {
                        q: coord.q,
                        r: coord.r,
                        piece: piece.kind.as_str().into(),
                        color: piece.color.as_str().into(),
                        square: coord.to_notation().unwrap_or_default(),
                    }
                })
            })
            .collect()
    }

    /// Is the current side to move in check?
    #[wasm_bindgen(js_name = "isInCheck")]
    pub fn is_in_check(&self) -> bool {
        movegen::is_in_check(&self.state.board, self.state.side_to_move())
    }

    /// Encode this game's board state as a Float32Array of shape (19, 11, 11),
    /// flattened in C (row-major) order.
    #[wasm_bindgen(js_name = "encodeBoard")]
    pub fn encode_board(&self) -> Vec<f32> {
        serialization::encode_board(&self.state).to_vec()
    }
}

impl Game {
    fn apply_resolved(
        &mut self,
        from: board::HexCoord,
        to: board::HexCoord,
        promo_kind: Option<PieceKind>,
    ) -> Result<(), JsError> {
        let legal = self.state.legal_moves();
        let mv = legal
            .iter()
            .find(|m| m.from == from && m.to == to && m.promotion == promo_kind)
            .ok_or_else(|| {
                JsError::new(&format!(
                    "illegal move: ({},{})->({},{})",
                    from.q, from.r, to.q, to.r
                ))
            })?;
        self.state.apply_move(*mv);
        Ok(())
    }
}

fn parse_promotion(s: Option<&str>) -> Result<Option<PieceKind>, JsError> {
    s.map(|s| {
        PieceKind::parse(s).ok_or_else(|| JsError::new(&format!("invalid promotion piece: {s}")))
    })
    .transpose()
}

// ---------------------------------------------------------------------------
// Glinski notation helpers (free functions)
// ---------------------------------------------------------------------------

/// Convert axial `(q, r)` to Glinski notation like `"f6"`.
#[wasm_bindgen(js_name = "toNotation")]
pub fn to_notation(q: i8, r: i8) -> Result<String, JsError> {
    board::HexCoord::new(q, r)
        .to_notation()
        .ok_or_else(|| JsError::new(&format!("invalid cell: ({q},{r})")))
}

/// Parse Glinski notation like `"f6"` into `{q, r}`.
#[derive(Tsify, Serialize, Deserialize)]
#[tsify(into_wasm_abi)]
pub struct JsCoord {
    pub q: i8,
    pub r: i8,
}

#[wasm_bindgen(js_name = "fromNotation")]
pub fn from_notation(s: &str) -> Result<JsCoord, JsError> {
    board::HexCoord::from_notation(s)
        .map(|c| JsCoord { q: c.q, r: c.r })
        .ok_or_else(|| JsError::new(&format!("invalid notation: {s}")))
}

/// Total number of move indices in the policy vector.
#[wasm_bindgen(js_name = "numMoveIndices")]
pub fn num_move_indices() -> usize {
    serialization::num_move_indices()
}

/// Board tensor shape (channels, height, width).
#[wasm_bindgen(js_name = "tensorShape")]
pub fn tensor_shape() -> Vec<usize> {
    vec![
        serialization::NUM_CHANNELS,
        serialization::BOARD_DIM,
        serialization::BOARD_DIM,
    ]
}

// ---------------------------------------------------------------------------
// AiPlayer class
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct AiPlayer {
    search: MctsSearch,
    simulations: u32,
}

#[wasm_bindgen]
impl AiPlayer {
    /// Create an AI player that uses the built-in heuristic evaluator.
    #[wasm_bindgen(constructor)]
    pub fn new(simulations: u32) -> AiPlayer {
        AiPlayer {
            search: MctsSearch::new(Box::new(HeuristicEvaluator::default())),
            simulations,
        }
    }

    /// Create an AI player that uses a neural-network model.
    /// `model_bytes` should be the contents of an ONNX model file.
    #[wasm_bindgen(js_name = "withModel")]
    pub fn with_model(simulations: u32, model_bytes: &[u8]) -> Result<AiPlayer, JsError> {
        let evaluator = TractEvaluator::from_bytes(model_bytes)
            .map_err(|e| JsError::new(&format!("failed to load model: {e}")))?;
        Ok(AiPlayer {
            search: MctsSearch::new(Box::new(evaluator)),
            simulations,
        })
    }

    /// Current simulation count per move.
    #[wasm_bindgen(getter)]
    pub fn simulations(&self) -> u32 {
        self.simulations
    }

    #[wasm_bindgen(setter)]
    pub fn set_simulations(&mut self, simulations: u32) {
        self.simulations = simulations;
    }

    /// Enable root-move Dirichlet noise with the given epsilon/alpha.
    /// Pass `epsilon = 0` to disable.
    #[wasm_bindgen(js_name = "setDirichlet")]
    pub fn set_dirichlet(&mut self, epsilon: f32, alpha: f64) {
        if epsilon > 0.0 {
            self.search
                .set_dirichlet(Some(DirichletConfig { epsilon, alpha }));
        } else {
            self.search.set_dirichlet(None);
        }
    }

    /// Run MCTS and return the full result (best move, policy, value, nodes).
    #[wasm_bindgen(js_name = "search")]
    pub fn search_game(&mut self, game: &Game, temperature: f32) -> JsMctsResult {
        let result =
            self.search
                .search_with_temperature(&game.state, self.simulations, temperature);
        JsMctsResult {
            best_move: JsMove::from_engine(&result.best_move),
            value: result.value,
            nodes: result.nodes_searched,
            policy: result.policy,
        }
    }

    /// Convenience: pick the best move greedily (temperature 0).
    #[wasm_bindgen(js_name = "bestMove")]
    pub fn best_move(&mut self, game: &Game) -> JsMove {
        self.best_move_with_temperature(game, 0.0)
    }

    /// Convenience: pick the best move with a sampling temperature.
    /// `temperature = 0` is greedy; `temperature = 1` samples proportionally
    /// to visit counts.
    #[wasm_bindgen(js_name = "bestMoveWithTemperature")]
    pub fn best_move_with_temperature(&mut self, game: &Game, temperature: f32) -> JsMove {
        let result =
            self.search
                .search_with_temperature(&game.state, self.simulations, temperature);
        JsMove::from_engine(&result.best_move)
    }

    /// Transposition-table stats.
    #[wasm_bindgen(js_name = "ttStats")]
    pub fn tt_stats(&self) -> JsTtStats {
        let s = self.search.tt_stats();
        JsTtStats {
            hits: s.hits,
            misses: s.misses,
            clears: s.clears,
            current_size: s.current_size,
        }
    }
}
