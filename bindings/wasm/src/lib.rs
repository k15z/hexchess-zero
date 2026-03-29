use wasm_bindgen::prelude::*;
use serde::Serialize;

use hexchess_engine::board::{self, PieceKind};
use hexchess_engine::game::GameState;
use hexchess_engine::mcts::{MctsSearch, HeuristicEvaluator};
use hexchess_engine::movegen;

// ---------------------------------------------------------------------------
// Serializable intermediate structs for JS interop
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsMove {
    from_q: i8,
    from_r: i8,
    to_q: i8,
    to_r: i8,
    promotion: Option<&'static str>,
}

#[derive(Serialize)]
struct JsPiece {
    q: i8,
    r: i8,
    piece: &'static str,
    color: &'static str,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn promotion_str(kind: Option<PieceKind>) -> Option<&'static str> {
    kind.map(PieceKind::as_str)
}

fn to_js<T: Serialize>(val: &T) -> JsValue {
    serde_wasm_bindgen::to_value(val).unwrap_or(JsValue::NULL)
}

// ---------------------------------------------------------------------------
// Game class
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct Game {
    state: GameState,
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Game {
        Game { state: GameState::new() }
    }

    /// All legal moves as [{from_q, from_r, to_q, to_r, promotion}].
    #[wasm_bindgen(js_name = "legalMoves")]
    pub fn legal_moves(&self) -> JsValue {
        let moves: Vec<JsMove> = self.state.legal_moves()
            .iter()
            .map(|m| JsMove {
                from_q: m.from.q,
                from_r: m.from.r,
                to_q: m.to.q,
                to_r: m.to.r,
                promotion: promotion_str(m.promotion),
            })
            .collect();
        to_js(&moves)
    }

    /// Apply a move. Promotion: "queen"/"rook"/"bishop"/"knight" or null.
    #[wasm_bindgen(js_name = "applyMove")]
    pub fn apply_move(
        &mut self,
        from_q: i8,
        from_r: i8,
        to_q: i8,
        to_r: i8,
        promotion: Option<String>,
    ) -> Result<(), JsError> {
        let promo_kind = promotion.as_deref()
            .map(|s| PieceKind::parse(s)
                .ok_or_else(|| JsError::new(&format!("invalid promotion piece: {s}"))))
            .transpose()?;

        let from = board::HexCoord::new(from_q, from_r);
        let to = board::HexCoord::new(to_q, to_r);

        let legal = self.state.legal_moves();
        let mv = legal.iter()
            .find(|m| m.from == from && m.to == to && m.promotion == promo_kind)
            .ok_or_else(|| JsError::new(&format!(
                "illegal move: ({from_q},{from_r})->({to_q},{to_r})"
            )))?;

        self.state.apply_move(*mv);
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

    /// "white" or "black".
    #[wasm_bindgen(js_name = "sideToMove")]
    pub fn side_to_move(&self) -> String {
        self.state.side_to_move().as_str().into()
    }

    /// All pieces on the board as [{q, r, piece, color}].
    #[wasm_bindgen(js_name = "boardState")]
    pub fn board_state(&self) -> JsValue {
        let pieces: Vec<JsPiece> = self.state.board.cells.iter().enumerate()
            .filter_map(|(idx, cell)| {
                cell.map(|piece| {
                    let coord = board::index_to_coord(idx);
                    JsPiece {
                        q: coord.q,
                        r: coord.r,
                        piece: piece.kind.as_str(),
                        color: piece.color.as_str(),
                    }
                })
            })
            .collect();
        to_js(&pieces)
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

    /// Is the current side to move in check?
    #[wasm_bindgen(js_name = "isInCheck")]
    pub fn is_in_check(&self) -> bool {
        movegen::is_in_check(&self.state.board, self.state.side_to_move())
    }
}

// ---------------------------------------------------------------------------
// AiPlayer class
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct AiPlayer {
    simulations: u32,
}

#[wasm_bindgen]
impl AiPlayer {
    #[wasm_bindgen(constructor)]
    pub fn new(simulations: u32) -> AiPlayer {
        AiPlayer { simulations }
    }

    /// Returns {from_q, from_r, to_q, to_r, promotion}.
    #[wasm_bindgen(js_name = "bestMove")]
    pub fn best_move(&self, game: &Game) -> JsValue {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        let result = search.search(&game.state, self.simulations);
        let mv = result.best_move;
        to_js(&JsMove {
            from_q: mv.from.q,
            from_r: mv.from.r,
            to_q: mv.to.q,
            to_r: mv.to.r,
            promotion: promotion_str(mv.promotion),
        })
    }
}
