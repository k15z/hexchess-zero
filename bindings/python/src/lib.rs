use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray3, PyArray4};

use hexchess_engine::board::{self, HexCoord, PieceKind};
use hexchess_engine::eval::EvalWeights;
use hexchess_engine::game::GameState;
use hexchess_engine::inference::OnnxEvaluator;
use hexchess_engine::mcts::{
    DirichletConfig, Evaluator, HeuristicEvaluator, MctsSearch as EngineSearch, SearchConfig,
};
use hexchess_engine::minimax;
use hexchess_engine::movegen::{self, Move as EngineMove};
use hexchess_engine::serialization;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn promotion_str(kind: Option<PieceKind>) -> Option<&'static str> {
    kind.map(PieceKind::as_str)
}

fn parse_promotion(s: Option<&str>) -> PyResult<Option<PieceKind>> {
    s.map(|s| {
        PieceKind::parse(s)
            .ok_or_else(|| PyValueError::new_err(format!("invalid promotion piece: {s}")))
    })
    .transpose()
}

// ---------------------------------------------------------------------------
// Move class
// ---------------------------------------------------------------------------

/// A move on the hex board. Immutable value type.
#[pyclass(name = "Move", frozen)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PyMove {
    #[pyo3(get)]
    from_q: i8,
    #[pyo3(get)]
    from_r: i8,
    #[pyo3(get)]
    to_q: i8,
    #[pyo3(get)]
    to_r: i8,
    promotion: Option<PieceKind>,
}

impl PyMove {
    fn from_engine(mv: &EngineMove) -> Self {
        Self {
            from_q: mv.from.q,
            from_r: mv.from.r,
            to_q: mv.to.q,
            to_r: mv.to.r,
            promotion: mv.promotion,
        }
    }

    fn to_engine_key(self) -> (HexCoord, HexCoord, Option<PieceKind>) {
        (
            HexCoord::new(self.from_q, self.from_r),
            HexCoord::new(self.to_q, self.to_r),
            self.promotion,
        )
    }
}

#[pymethods]
impl PyMove {
    /// Construct a move. `promotion` is one of "queen"/"rook"/"bishop"/"knight" or None.
    #[new]
    #[pyo3(signature = (from_q, from_r, to_q, to_r, promotion=None))]
    fn new(from_q: i8, from_r: i8, to_q: i8, to_r: i8, promotion: Option<&str>) -> PyResult<Self> {
        Ok(Self {
            from_q,
            from_r,
            to_q,
            to_r,
            promotion: parse_promotion(promotion)?,
        })
    }

    /// Parse a move from Glinski notation like "f5-f6" or "f10-f11=Q".
    #[staticmethod]
    fn from_notation(s: &str) -> PyResult<Self> {
        let (from, to, promo) = EngineMove::parse_notation(s)
            .ok_or_else(|| PyValueError::new_err(format!("invalid move notation: {s}")))?;
        Ok(Self {
            from_q: from.q,
            from_r: from.r,
            to_q: to.q,
            to_r: to.r,
            promotion: promo,
        })
    }

    /// Promotion piece as a string, or None.
    #[getter]
    fn promotion(&self) -> Option<&'static str> {
        promotion_str(self.promotion)
    }

    /// `(q, r)` tuple of the from-square.
    #[getter]
    #[pyo3(name = "from_")]
    fn from_(&self) -> (i8, i8) {
        (self.from_q, self.from_r)
    }

    /// `(q, r)` tuple of the to-square.
    #[getter]
    fn to(&self) -> (i8, i8) {
        (self.to_q, self.to_r)
    }

    /// Glinski notation like "f5-f6" or "f10-f11=Q".
    #[getter]
    fn notation(&self) -> PyResult<String> {
        let from = HexCoord::new(self.from_q, self.from_r);
        let to = HexCoord::new(self.to_q, self.to_r);
        let mv = EngineMove::new(from, to, None).with_promotion_opt(self.promotion);
        mv.to_notation()
            .ok_or_else(|| PyValueError::new_err("move endpoints are off-board"))
    }

    fn __repr__(&self) -> String {
        match self.notation() {
            Ok(s) => format!("Move({})", s),
            Err(_) => format!(
                "Move(from=({},{}), to=({},{}), promotion={:?})",
                self.from_q, self.from_r, self.to_q, self.to_r, self.promotion
            ),
        }
    }

    fn __str__(&self) -> String {
        self.notation().unwrap_or_else(|_| {
            format!(
                "({},{})->({},{})",
                self.from_q, self.from_r, self.to_q, self.to_r
            )
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

// ---------------------------------------------------------------------------
// Piece class
// ---------------------------------------------------------------------------

/// A piece on the board at a given cell.
#[pyclass(name = "Piece", frozen)]
#[derive(Clone, Copy)]
struct PyPiece {
    #[pyo3(get)]
    q: i8,
    #[pyo3(get)]
    r: i8,
    kind: PieceKind,
    color: board::Color,
}

#[pymethods]
impl PyPiece {
    #[getter]
    fn piece(&self) -> &'static str {
        self.kind.as_str()
    }

    #[getter]
    fn color(&self) -> &'static str {
        self.color.as_str()
    }

    #[getter]
    fn square(&self) -> Option<String> {
        HexCoord::new(self.q, self.r).to_notation()
    }

    fn __repr__(&self) -> String {
        format!(
            "Piece(q={}, r={}, piece={:?}, color={:?})",
            self.q,
            self.r,
            self.kind.as_str(),
            self.color.as_str()
        )
    }
}

// ---------------------------------------------------------------------------
// TtStats class
// ---------------------------------------------------------------------------

#[pyclass(name = "TtStats", frozen)]
#[derive(Clone, Copy)]
struct PyTtStats {
    #[pyo3(get)]
    hits: u64,
    #[pyo3(get)]
    misses: u64,
    #[pyo3(get)]
    clears: u64,
    #[pyo3(get)]
    current_size: usize,
}

#[pymethods]
impl PyTtStats {
    fn __repr__(&self) -> String {
        format!(
            "TtStats(hits={}, misses={}, clears={}, current_size={})",
            self.hits, self.misses, self.clears, self.current_size
        )
    }
}

// ---------------------------------------------------------------------------
// MctsResult class
// ---------------------------------------------------------------------------

#[pyclass(name = "MctsResult")]
struct PyMctsResult {
    #[pyo3(get)]
    best_move: PyMove,
    policy: Vec<f32>,
    #[pyo3(get)]
    value: f32,
    wdl: [f32; 3],
    nn_wdl: [f32; 3],
    #[pyo3(get)]
    nodes: u32,
}

#[pymethods]
impl PyMctsResult {
    /// Policy vector as a numpy array of length `num_move_indices()`.
    ///
    /// Copies once into numpy-owned memory; repeated access re-copies, so
    /// bind to a local variable if you need it more than once.
    #[getter]
    fn policy<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        numpy::PyArray1::from_slice(py, &self.policy)
    }

    /// Root `[W, D, L]` distribution from STM perspective. Trainer and
    /// callers that want the full WDL instead of just `W - L`.
    #[getter]
    fn wdl(&self) -> (f32, f32, f32) {
        (self.wdl[0], self.wdl[1], self.wdl[2])
    }

    /// Raw NN evaluation `[W, D, L]` before MCTS backups. For "NN vs MCTS"
    /// diagnostics comparing network intuition to search result.
    #[getter]
    fn nn_wdl(&self) -> (f32, f32, f32) {
        (self.nn_wdl[0], self.nn_wdl[1], self.nn_wdl[2])
    }

    fn __repr__(&self) -> String {
        format!(
            "MctsResult(best_move={}, value={:.3}, wdl=({:.3}, {:.3}, {:.3}), nodes={})",
            self.best_move.__str__(),
            self.value,
            self.wdl[0],
            self.wdl[1],
            self.wdl[2],
            self.nodes
        )
    }
}

// ---------------------------------------------------------------------------
// RankedMove + minimax result classes
// ---------------------------------------------------------------------------

#[pyclass(name = "RankedMove", frozen)]
#[derive(Clone, Copy)]
struct PyRankedMove {
    #[pyo3(get)]
    r#move: PyMove,
    #[pyo3(get)]
    score: i32,
}

#[pymethods]
impl PyRankedMove {
    fn __repr__(&self) -> String {
        format!(
            "RankedMove(move={}, score={})",
            self.r#move.__str__(),
            self.score
        )
    }
}

#[pyclass(name = "MinimaxResult", frozen)]
struct PyMinimaxResult {
    #[pyo3(get)]
    best_move: PyMove,
    #[pyo3(get)]
    score: i32,
    #[pyo3(get)]
    nodes: u64,
}

#[pymethods]
impl PyMinimaxResult {
    fn __repr__(&self) -> String {
        format!(
            "MinimaxResult(best_move={}, score={}, nodes={})",
            self.best_move.__str__(),
            self.score,
            self.nodes
        )
    }
}

#[pyclass(name = "MinimaxAllResult", frozen)]
struct PyMinimaxAllResult {
    #[pyo3(get)]
    moves: Vec<PyRankedMove>,
    #[pyo3(get)]
    nodes: u64,
}

#[pymethods]
impl PyMinimaxAllResult {
    fn __repr__(&self) -> String {
        format!(
            "MinimaxAllResult(moves=[{}], nodes={})",
            self.moves.len(),
            self.nodes
        )
    }
}

#[pyclass(name = "MinimaxPolicyResult", frozen)]
struct PyMinimaxPolicyResult {
    #[pyo3(get)]
    best_move: PyMove,
    #[pyo3(get)]
    best_score: i32,
    #[pyo3(get)]
    moves: Vec<PyRankedMove>,
    #[pyo3(get)]
    nodes: u64,
}

#[pymethods]
impl PyMinimaxPolicyResult {
    fn __repr__(&self) -> String {
        format!(
            "MinimaxPolicyResult(best_move={}, best_score={}, moves=[{}], nodes={})",
            self.best_move.__str__(),
            self.best_score,
            self.moves.len(),
            self.nodes
        )
    }
}

// ---------------------------------------------------------------------------
// EvalWeights class
// ---------------------------------------------------------------------------

#[pyclass(name = "EvalWeights")]
#[derive(Clone)]
struct PyEvalWeights {
    inner: EvalWeights,
}

#[pymethods]
impl PyEvalWeights {
    #[new]
    #[pyo3(signature = (
        material=1, mobility=0, pawn_advance=1, center_control=4,
        king_safety=0, bishop_color_bonus=30, pawn_connected=7, pawn_isolated=-10,
        passed_pawn=3, king_tropism=2
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        material: i32,
        mobility: i32,
        pawn_advance: i32,
        center_control: i32,
        king_safety: i32,
        bishop_color_bonus: i32,
        pawn_connected: i32,
        pawn_isolated: i32,
        passed_pawn: i32,
        king_tropism: i32,
    ) -> Self {
        PyEvalWeights {
            inner: EvalWeights {
                material,
                mobility,
                pawn_advance,
                center_control,
                king_safety,
                bishop_color_bonus,
                pawn_connected,
                pawn_isolated,
                passed_pawn,
                king_tropism,
            },
        }
    }

    /// Material counting only — no positional signals.
    #[staticmethod]
    fn material_only() -> Self {
        PyEvalWeights {
            inner: EvalWeights::material_only(),
        }
    }

    fn __repr__(&self) -> String {
        let w = &self.inner;
        format!(
            "EvalWeights(material={}, mobility={}, pawn_advance={}, center_control={}, \
             king_safety={}, bishop_color_bonus={}, pawn_connected={}, pawn_isolated={}, \
             passed_pawn={}, king_tropism={})",
            w.material,
            w.mobility,
            w.pawn_advance,
            w.center_control,
            w.king_safety,
            w.bishop_color_bonus,
            w.pawn_connected,
            w.pawn_isolated,
            w.passed_pawn,
            w.king_tropism
        )
    }
}

// ---------------------------------------------------------------------------
// Game class
// ---------------------------------------------------------------------------

#[pyclass(name = "Game")]
struct PyGame {
    state: GameState,
}

impl PyGame {
    /// Resolve a `Move | str | (fq, fr, tq, tr[, promo])`-ish input to an
    /// engine move by scanning the legal move list.
    fn resolve_legal_move(
        &self,
        mv_key: (HexCoord, HexCoord, Option<PieceKind>),
    ) -> PyResult<EngineMove> {
        let (from, to, promo) = mv_key;
        let legal = self.state.legal_moves();
        legal
            .iter()
            .find(|m| m.from == from && m.to == to && m.promotion == promo)
            .copied()
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "illegal move: ({},{})->({},{})",
                    from.q, from.r, to.q, to.r
                ))
            })
    }
}

#[pymethods]
impl PyGame {
    #[new]
    fn new() -> Self {
        PyGame {
            state: GameState::new(),
        }
    }

    /// Return the list of legal moves as `Move` objects.
    fn legal_moves(&self) -> Vec<PyMove> {
        self.state
            .legal_moves()
            .iter()
            .map(PyMove::from_engine)
            .collect()
    }

    /// Apply a move specified as five positional args (legacy API).
    /// Prefer `apply(mv)` for new code — it accepts a `Move` or a notation string.
    #[pyo3(signature = (from_q, from_r, to_q, to_r, promotion=None))]
    fn apply_move(
        &mut self,
        from_q: i8,
        from_r: i8,
        to_q: i8,
        to_r: i8,
        promotion: Option<&str>,
    ) -> PyResult<()> {
        let promo = parse_promotion(promotion)?;
        let from = HexCoord::new(from_q, from_r);
        let to = HexCoord::new(to_q, to_r);
        let mv = self.resolve_legal_move((from, to, promo))?;
        self.state.apply_move(mv);
        Ok(())
    }

    /// Apply a move given as a `Move` object or a Glinski notation string
    /// (e.g. "f5-f6", "f10-f11=Q"). Raises `ValueError` if the move is illegal.
    fn apply(&mut self, mv: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = if let Ok(py_mv) = mv.extract::<PyRef<PyMove>>() {
            py_mv.to_engine_key()
        } else if let Ok(s) = mv.extract::<String>() {
            let (from, to, promo) = EngineMove::parse_notation(&s)
                .ok_or_else(|| PyValueError::new_err(format!("invalid move notation: {s}")))?;
            (from, to, promo)
        } else {
            return Err(PyValueError::new_err(
                "apply() expects a Move or a notation string",
            ));
        };
        let engine_mv = self.resolve_legal_move(key)?;
        self.state.apply_move(engine_mv);
        Ok(())
    }

    /// Undo the last move.
    fn undo_move(&mut self) -> PyResult<()> {
        if self.state.move_count() == 0 {
            return Err(PyValueError::new_err("no moves to undo"));
        }
        self.state.undo_move();
        Ok(())
    }

    /// Game status string: "ongoing", "checkmate_white", "checkmate_black",
    /// "stalemate", "draw_repetition", "draw_fifty", "draw_material".
    fn status(&self) -> &'static str {
        self.state.status().as_str()
    }

    fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    /// "white" or "black".
    fn side_to_move(&self) -> &'static str {
        self.state.side_to_move().as_str()
    }

    /// Number of half-moves played.
    fn move_count(&self) -> usize {
        self.state.move_count()
    }

    /// All pieces on the board as a list of `Piece` objects.
    fn board_state(&self) -> Vec<PyPiece> {
        let mut out = Vec::new();
        for (idx, cell) in self.state.board.cells.iter().enumerate() {
            if let Some(piece) = cell {
                let coord = board::index_to_coord(idx);
                out.push(PyPiece {
                    q: coord.q,
                    r: coord.r,
                    kind: piece.kind,
                    color: piece.color,
                });
            }
        }
        out
    }

    fn is_in_check(&self) -> bool {
        movegen::is_in_check(&self.state.board, self.state.side_to_move())
    }

    /// STM-relative policy-vector index for a move, using this game's current
    /// side-to-move. Use this instead of the top-level `move_to_index` when
    /// building policy targets that will be paired with an STM-frame board
    /// tensor (i.e. any target consumed by the trainer): it applies the
    /// mirror remap when black is to move so the target lands in the same
    /// slot the NN emits.
    #[pyo3(signature = (from_q, from_r, to_q, to_r, promotion=None))]
    fn policy_index(
        &self,
        from_q: i8,
        from_r: i8,
        to_q: i8,
        to_r: i8,
        promotion: Option<&str>,
    ) -> PyResult<usize> {
        let promo = parse_promotion(promotion)?;
        let from = HexCoord::new(from_q, from_r);
        let to = HexCoord::new(to_q, to_r);
        let mv = EngineMove::new(from, to, None).with_promotion_opt(promo);
        serialization::stm_policy_index(&mv, self.state.side_to_move())
            .ok_or_else(|| PyValueError::new_err("move not in index table"))
    }

    /// Deep copy of this game.
    fn clone(&self) -> Self {
        PyGame {
            state: self.state.clone(),
        }
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Game(side={}, ply={}, status={})",
            self.state.side_to_move().as_str(),
            self.state.move_count(),
            self.state.status().as_str()
        )
    }

    /// ASCII rendering of the board — useful in notebooks.
    fn __str__(&self) -> String {
        format!("{}", self.state.board)
    }
}

// ---------------------------------------------------------------------------
// MctsSearch class
// ---------------------------------------------------------------------------

#[pyclass(name = "MctsSearch")]
struct PyMctsSearch {
    search: EngineSearch,
    #[pyo3(get, set)]
    simulations: u32,
    rng: StdRng,
}

#[pymethods]
impl PyMctsSearch {
    /// Create an MCTS search engine.
    ///
    /// If `model_path` is provided, loads the ONNX neural network once.
    /// Otherwise uses a heuristic evaluator (uniform policy, material value).
    ///
    /// `eval_mode=True` swaps the underlying `SearchConfig` to
    /// `SearchConfig::eval()` (no forced playouts, LCB move selection,
    /// greedy temperature, no Dirichlet, no policy-target pruning).
    /// Use this for Elo, gauntlet, benchmark, and replay paths.
    /// The default (`eval_mode=False`) uses `SearchConfig::training()`
    /// (c_puct=2.5, Dirichlet noise, PCR, temperature decay, etc.).
    ///
    /// If `c_puct` is provided it overrides the config's default for both
    /// training and eval modes.
    ///
    /// `dirichlet_epsilon` / `dirichlet_alpha` enable root-move Dirichlet
    /// noise on top of whichever config was selected; set
    /// `dirichlet_epsilon=0` (the default) to use the config's built-in
    /// value (training: 0.25 epsilon / 0.25 alpha; eval: none).
    ///
    /// PCR and temperature parameters override `SearchConfig::training()`
    /// defaults when provided; eval mode ignores them.
    #[new]
    #[pyo3(signature = (
        simulations=800, c_puct=None, model_path=None, batch_size=32,
        tt_capacity=500_000, intra_threads=0,
        dirichlet_epsilon=0.0, dirichlet_alpha=0.3, eval_mode=false,
        pcr_p_full=None, pcr_n_full=None, pcr_n_fast=None,
        temperature_high=None, temperature_low=None, temperature_threshold=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        simulations: u32,
        c_puct: Option<f32>,
        model_path: Option<String>,
        batch_size: usize,
        tt_capacity: usize,
        intra_threads: usize,
        dirichlet_epsilon: f32,
        dirichlet_alpha: f64,
        eval_mode: bool,
        pcr_p_full: Option<f32>,
        pcr_n_full: Option<u32>,
        pcr_n_fast: Option<u32>,
        temperature_high: Option<f32>,
        temperature_low: Option<f32>,
        temperature_threshold: Option<u32>,
    ) -> PyResult<Self> {
        let evaluator: Box<dyn Evaluator> = match model_path {
            Some(path) => {
                let eval =
                    OnnxEvaluator::from_path_with_threads(&path, intra_threads).map_err(|e| {
                        PyValueError::new_err(format!("failed to load ONNX model '{path}': {e}"))
                    })?;
                Box::new(eval)
            }
            None => Box::new(HeuristicEvaluator::default()),
        };
        let mut search = EngineSearch::new(evaluator);
        if eval_mode {
            search.set_config(SearchConfig::eval());
        }
        if let Some(c_puct) = c_puct {
            search.set_c_puct(c_puct);
        }
        // Apply PCR overrides (training mode only — eval uses p_full=1.0).
        if !eval_mode {
            let cfg = search.config_mut();
            if let Some(v) = pcr_p_full {
                cfg.pcr.p_full = v;
            }
            if let Some(v) = pcr_n_full {
                cfg.pcr.n_full = v;
            }
            if let Some(v) = pcr_n_fast {
                cfg.pcr.n_fast = v;
            }
            if let Some(v) = temperature_high {
                cfg.temperature.tau_max = v;
            }
            if let Some(v) = temperature_low {
                cfg.temperature.tau_min = v;
            }
            if let Some(v) = temperature_threshold {
                cfg.temperature.hard_greedy_ply = v;
            }
        }
        search.set_batch_size(batch_size);
        search.set_tt_capacity(tt_capacity);
        if dirichlet_epsilon > 0.0 {
            search.set_dirichlet(Some(DirichletConfig {
                epsilon: dirichlet_epsilon,
                alpha: dirichlet_alpha,
            }));
        }
        Ok(PyMctsSearch {
            search,
            simulations,
            rng: StdRng::from_os_rng(),
        })
    }

    /// Seed both the binding-level RNG (used by `run_pcr` for coin flips)
    /// and the engine's internal RNG (Dirichlet noise, etc.) so searches
    /// become deterministic given the same inputs.
    fn set_rng_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
        self.search.set_rng_seed(seed);
    }

    /// Run a Playout-Cap-Randomization search step. Returns a dict:
    ///   {best_move, value, wdl, nn_wdl, nodes, was_full_search, temperature,
    ///    policy_target (or None)}
    /// where `wdl` is a `(W, D, L)` tuple from the STM's perspective (after
    /// MCTS backups) and `nn_wdl` is the raw NN evaluation before search.
    /// Only full-search steps should be recorded as training samples.
    #[pyo3(signature = (game, ply=0))]
    fn run_pcr<'py>(
        &mut self,
        py: Python<'py>,
        game: &PyGame,
        ply: u32,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let outcome = self.search.run_pcr(&game.state, ply, &mut self.rng);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item(
            "best_move",
            PyMove::from_engine(&outcome.best_move).into_pyobject(py)?,
        )?;
        dict.set_item("value", outcome.value)?;
        dict.set_item("wdl", (outcome.wdl[0], outcome.wdl[1], outcome.wdl[2]))?;
        dict.set_item(
            "nn_wdl",
            (outcome.nn_wdl[0], outcome.nn_wdl[1], outcome.nn_wdl[2]),
        )?;
        dict.set_item("nodes", outcome.nodes_searched)?;
        dict.set_item("was_full_search", outcome.was_full_search)?;
        dict.set_item("temperature", outcome.temperature)?;
        match outcome.policy_target {
            Some(p) => {
                let arr = numpy::PyArray1::from_slice(py, &p);
                dict.set_item("policy_target", arr)?;
            }
            None => {
                dict.set_item("policy_target", py.None())?;
            }
        }
        Ok(dict)
    }

    /// Run MCTS search on `game`. Returns an `MctsResult`.
    ///
    /// `temperature=0` selects the most-visited move; `temperature>0` samples
    /// proportionally to visit counts ^ (1/temperature).
    #[pyo3(signature = (game, temperature=0.0))]
    fn run(&mut self, game: &PyGame, temperature: f32) -> PyMctsResult {
        let result =
            self.search
                .search_with_temperature(&game.state, self.simulations, temperature);
        PyMctsResult {
            best_move: PyMove::from_engine(&result.best_move),
            policy: result.policy,
            value: result.value,
            wdl: result.wdl,
            nn_wdl: result.nn_wdl,
            nodes: result.nodes_searched,
        }
    }

    /// Set the draw-utility ("contempt") coefficient. Positive values bias
    /// search toward draws; negative values bias away. Default 0 keeps the
    /// plain `W − L` AlphaZero Q function.
    fn set_draw_utility(&mut self, draw_utility: f32) {
        self.search.set_draw_utility(draw_utility);
    }

    /// Return the opponent-reply visit distribution from the previous
    /// search as a numpy float32 array of length `num_move_indices()`, or
    /// `None` if unavailable (no search yet, or the best child is
    /// unexpanded).
    fn aux_opponent_policy<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, numpy::PyArray1<f32>>> {
        self.search
            .aux_opponent_policy()
            .map(|v| numpy::PyArray1::from_vec(py, v))
    }

    /// Transposition-table stats.
    fn tt_stats(&self) -> PyTtStats {
        let s = self.search.tt_stats();
        PyTtStats {
            hits: s.hits,
            misses: s.misses,
            clears: s.clears,
            current_size: s.current_size,
        }
    }

    /// Snapshot of the key `SearchConfig` fields as a dict. Intended for
    /// tests that need to assert eval vs training mode; not a stable API.
    fn config_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let cfg = self.search.config();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("c_puct", cfg.c_puct)?;
        dict.set_item("c_puct_root", cfg.c_puct_root)?;
        dict.set_item("fpu_reduction", cfg.fpu_reduction)?;
        dict.set_item("batch_size", cfg.batch_size)?;
        dict.set_item("tt_capacity", cfg.tt_capacity)?;
        dict.set_item("forced_playout_k", cfg.forced_playout_k)?;
        dict.set_item("policy_target_pruning", cfg.policy_target_pruning)?;
        dict.set_item("use_lcb", cfg.use_lcb)?;
        dict.set_item("draw_utility", cfg.draw_utility)?;
        match &cfg.dirichlet {
            Some(d) => {
                let dnode = pyo3::types::PyDict::new(py);
                dnode.set_item("epsilon", d.epsilon)?;
                dnode.set_item("alpha", d.alpha)?;
                dict.set_item("dirichlet", dnode)?;
            }
            None => {
                dict.set_item("dirichlet", py.None())?;
            }
        }
        Ok(dict)
    }
}

// ---------------------------------------------------------------------------
// Encoding functions
// ---------------------------------------------------------------------------

/// Encode a game's board state as a numpy array of shape (22, 11, 11).
#[pyfunction]
fn encode_board<'py>(py: Python<'py>, game: &PyGame) -> Bound<'py, PyArray3<f32>> {
    let flat = serialization::encode_board(&game.state);
    let array = Array3::from_shape_vec(
        (
            serialization::NUM_CHANNELS,
            serialization::BOARD_DIM,
            serialization::BOARD_DIM,
        ),
        flat.to_vec(),
    )
    .expect("shape mismatch in encode_board");
    array.into_pyarray(py)
}

/// Encode a batch of games as a numpy array of shape (N, 22, 11, 11).
#[pyfunction]
fn encode_batch<'py>(
    py: Python<'py>,
    games: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let n = games.len();
    let c = serialization::NUM_CHANNELS;
    let h = serialization::BOARD_DIM;
    let w = serialization::BOARD_DIM;

    let mut data = Vec::with_capacity(n * c * h * w);
    for game_obj in &games {
        let game: PyRef<'_, PyGame> = game_obj.extract()?;
        let flat = serialization::encode_board(&game.state);
        data.extend_from_slice(&flat);
    }

    let array = Array4::from_shape_vec((n, c, h, w), data).expect("shape mismatch in encode_batch");
    Ok(array.into_pyarray(py))
}

/// Convert (from_q, from_r, to_q, to_r, promotion) to a policy-vector index.
#[pyfunction]
#[pyo3(signature = (from_q, from_r, to_q, to_r, promotion=None))]
fn move_to_index(
    from_q: i8,
    from_r: i8,
    to_q: i8,
    to_r: i8,
    promotion: Option<&str>,
) -> PyResult<usize> {
    let promo = parse_promotion(promotion)?;
    let from = HexCoord::new(from_q, from_r);
    let to = HexCoord::new(to_q, to_r);
    let mv = EngineMove::new(from, to, None).with_promotion_opt(promo);

    serialization::move_to_index(&mv)
        .ok_or_else(|| PyValueError::new_err("move not in index table"))
}

/// Convert a policy-vector index to a `Move`.
#[pyfunction]
fn index_to_move(idx: usize) -> PyResult<PyMove> {
    let n = serialization::num_move_indices();
    if idx >= n {
        return Err(PyValueError::new_err(format!(
            "index {idx} out of range [0, {n})"
        )));
    }
    let (from, to, promo) = serialization::index_to_move(idx);
    Ok(PyMove {
        from_q: from.q,
        from_r: from.r,
        to_q: to.q,
        to_r: to.r,
        promotion: promo,
    })
}

/// Total number of move indices in the policy vector.
#[pyfunction]
fn num_move_indices() -> usize {
    serialization::num_move_indices()
}

// ---------------------------------------------------------------------------
// Minimax search
// ---------------------------------------------------------------------------

fn resolve_weights(weights: Option<&PyEvalWeights>) -> EvalWeights {
    match weights {
        Some(w) => w.inner,
        None => EvalWeights::default(),
    }
}

fn ranked_moves(moves: &[minimax::RankedMove]) -> Vec<PyRankedMove> {
    moves
        .iter()
        .map(|r| PyRankedMove {
            r#move: PyMove::from_engine(&r.mv),
            score: r.score,
        })
        .collect()
}

/// Alpha-beta minimax search. Raises `ValueError` if the game is terminal.
#[pyfunction]
#[pyo3(signature = (game, depth, weights=None))]
fn minimax_search(
    game: &mut PyGame,
    depth: u32,
    weights: Option<&PyEvalWeights>,
) -> PyResult<PyMinimaxResult> {
    let w = resolve_weights(weights);
    let result = minimax::search(&mut game.state, depth, &w)
        .ok_or_else(|| PyValueError::new_err("cannot search a terminal position"))?;
    Ok(PyMinimaxResult {
        best_move: PyMove::from_engine(&result.best_move),
        score: result.score,
        nodes: result.nodes,
    })
}

/// Minimax returning scores for all legal moves.
#[pyfunction]
#[pyo3(signature = (game, depth, weights=None))]
fn minimax_search_all(
    game: &mut PyGame,
    depth: u32,
    weights: Option<&PyEvalWeights>,
) -> PyResult<PyMinimaxAllResult> {
    let w = resolve_weights(weights);
    let result = minimax::search_all_moves(&mut game.state, depth, &w)
        .ok_or_else(|| PyValueError::new_err("cannot search a terminal position"))?;
    Ok(PyMinimaxAllResult {
        moves: ranked_moves(&result.moves),
        nodes: result.nodes,
    })
}

/// Two-phase minimax: optimized best-move search + shallow all-root re-search.
#[pyfunction]
#[pyo3(signature = (game, depth, weights=None))]
fn minimax_search_with_policy(
    game: &mut PyGame,
    depth: u32,
    weights: Option<&PyEvalWeights>,
) -> PyResult<PyMinimaxPolicyResult> {
    let w = resolve_weights(weights);
    let result = minimax::search_with_policy(&mut game.state, depth, &w)
        .ok_or_else(|| PyValueError::new_err("cannot search a terminal position"))?;
    Ok(PyMinimaxPolicyResult {
        best_move: PyMove::from_engine(&result.best_move),
        best_score: result.best_score,
        moves: ranked_moves(&result.move_scores),
        nodes: result.nodes,
    })
}

// ---------------------------------------------------------------------------
// Glinski notation helpers
// ---------------------------------------------------------------------------

/// Convert axial (q, r) to Glinski notation like "f6".
#[pyfunction]
fn to_notation(q: i8, r: i8) -> PyResult<String> {
    HexCoord::new(q, r)
        .to_notation()
        .ok_or_else(|| PyValueError::new_err(format!("invalid cell: ({q},{r})")))
}

/// Parse Glinski notation like "f6" into (q, r). Raises ValueError on bad input.
#[pyfunction]
fn from_notation(s: &str) -> PyResult<(i8, i8)> {
    HexCoord::from_notation(s)
        .map(|c| (c.q, c.r))
        .ok_or_else(|| PyValueError::new_err(format!("invalid notation: {s}")))
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn hexchess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyGame>()?;
    m.add_class::<PyMctsSearch>()?;
    m.add_class::<PyEvalWeights>()?;
    m.add_class::<PyMove>()?;
    m.add_class::<PyPiece>()?;
    m.add_class::<PyMctsResult>()?;
    m.add_class::<PyTtStats>()?;
    m.add_class::<PyRankedMove>()?;
    m.add_class::<PyMinimaxResult>()?;
    m.add_class::<PyMinimaxAllResult>()?;
    m.add_class::<PyMinimaxPolicyResult>()?;

    // Functions
    m.add_function(wrap_pyfunction!(encode_board, m)?)?;
    m.add_function(wrap_pyfunction!(encode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(move_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_to_move, m)?)?;
    m.add_function(wrap_pyfunction!(num_move_indices, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_search, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_search_all, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_search_with_policy, m)?)?;
    m.add_function(wrap_pyfunction!(to_notation, m)?)?;
    m.add_function(wrap_pyfunction!(from_notation, m)?)?;

    // Constants
    let tensor_shape = (
        serialization::NUM_CHANNELS,
        serialization::BOARD_DIM,
        serialization::BOARD_DIM,
    );
    m.add("TENSOR_SHAPE", tensor_shape)?;
    m.add("NUM_MOVE_INDICES", serialization::num_move_indices())?;

    Ok(())
}
