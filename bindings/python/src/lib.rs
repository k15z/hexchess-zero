use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use numpy::ndarray::{Array1, Array3, Array4};
use numpy::{IntoPyArray, PyArray3, PyArray4};

use hexchess_engine::board::{self, HexCoord, PieceKind};
use hexchess_engine::game::GameState;
use hexchess_engine::inference::OnnxEvaluator;
use hexchess_engine::mcts::{
    DirichletConfig, Evaluator, HeuristicEvaluator, MctsSearch as EngineSearch,
};
use hexchess_engine::minimax;

use hexchess_engine::movegen;
use hexchess_engine::serialization;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn promotion_str(kind: Option<PieceKind>) -> Option<&'static str> {
    kind.map(PieceKind::as_str)
}

/// Build a Python dict representing a move.
fn move_to_pydict<'py>(py: Python<'py>, mv: &movegen::Move) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("from_q", mv.from.q)?;
    dict.set_item("from_r", mv.from.r)?;
    dict.set_item("to_q", mv.to.q)?;
    dict.set_item("to_r", mv.to.r)?;
    dict.set_item("promotion", promotion_str(mv.promotion))?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// PyGame class
// ---------------------------------------------------------------------------

#[pyclass(name = "Game")]
struct PyGame {
    state: GameState,
}

#[pymethods]
impl PyGame {
    #[new]
    fn new() -> Self {
        PyGame {
            state: GameState::new(),
        }
    }

    /// Return a list of dicts {from_q, from_r, to_q, to_r, promotion}.
    fn legal_moves<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let moves = self.state.legal_moves();
        let list = PyList::empty(py);
        for mv in &moves {
            list.append(move_to_pydict(py, mv)?)?;
        }
        Ok(list)
    }

    /// Apply a move. Promotion should be "queen", "rook", "bishop", "knight", or None.
    #[pyo3(signature = (from_q, from_r, to_q, to_r, promotion=None))]
    fn apply_move(
        &mut self,
        from_q: i8,
        from_r: i8,
        to_q: i8,
        to_r: i8,
        promotion: Option<&str>,
    ) -> PyResult<()> {
        let promo_kind = promotion
            .map(|s| {
                PieceKind::parse(s)
                    .ok_or_else(|| PyValueError::new_err(format!("invalid promotion piece: {s}")))
            })
            .transpose()?;

        let from = HexCoord::new(from_q, from_r);
        let to = HexCoord::new(to_q, to_r);

        let legal = self.state.legal_moves();
        let mv = legal
            .iter()
            .find(|m| m.from == from && m.to == to && m.promotion == promo_kind)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "illegal move: ({from_q},{from_r})->({to_q},{to_r})"
                ))
            })?;

        self.state.apply_move(*mv);
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

    /// Is the game over?
    fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    /// "white" or "black".
    fn side_to_move(&self) -> &'static str {
        self.state.side_to_move().as_str()
    }

    /// All pieces on the board as a list of dicts {q, r, piece, color}.
    fn board_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for (idx, cell) in self.state.board.cells.iter().enumerate() {
            if let Some(piece) = cell {
                let coord = board::index_to_coord(idx);
                let dict = PyDict::new(py);
                dict.set_item("q", coord.q)?;
                dict.set_item("r", coord.r)?;
                dict.set_item("piece", piece.kind.as_str())?;
                dict.set_item("color", piece.color.as_str())?;
                list.append(dict)?;
            }
        }
        Ok(list)
    }

    /// Is the current side to move in check?
    fn is_in_check(&self) -> bool {
        movegen::is_in_check(&self.state.board, self.state.side_to_move())
    }

    /// Deep copy of this game.
    fn clone(&self) -> Self {
        PyGame {
            state: self.state.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyMctsSearch class
// ---------------------------------------------------------------------------

#[pyclass(name = "MctsSearch")]
struct PyMctsSearch {
    search: EngineSearch,
    simulations: u32,
}

#[pymethods]
impl PyMctsSearch {
    /// Create an MCTS search engine.
    ///
    /// If `model_path` is provided, loads the ONNX neural network once and
    /// uses it as the evaluator for all subsequent `run()` calls.
    /// Otherwise, uses a heuristic evaluator (uniform policy, material value).
    #[new]
    #[pyo3(signature = (simulations=800, c_puct=1.5, model_path=None, batch_size=32, tt_capacity=500_000, intra_threads=0))]
    fn new(
        simulations: u32,
        c_puct: f32,
        model_path: Option<String>,
        batch_size: usize,
        tt_capacity: usize,
        intra_threads: usize,
    ) -> PyResult<Self> {
        let evaluator: Box<dyn Evaluator> = match model_path {
            Some(path) => {
                let eval =
                    OnnxEvaluator::from_path_with_threads(&path, intra_threads).map_err(|e| {
                        PyValueError::new_err(format!("failed to load ONNX model '{path}': {e}"))
                    })?;
                Box::new(eval)
            }
            None => Box::new(HeuristicEvaluator),
        };
        let mut search = EngineSearch::new(evaluator);
        search.set_c_puct(c_puct);
        search.set_batch_size(batch_size);
        search.set_tt_capacity(tt_capacity);
        Ok(PyMctsSearch {
            search,
            simulations,
        })
    }

    /// Run MCTS search. Returns dict {best_move, policy, value, nodes}.
    #[pyo3(signature = (game, temperature=None, dirichlet_epsilon=None, dirichlet_alpha=None))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        game: &PyGame,
        temperature: Option<f32>,
        dirichlet_epsilon: Option<f32>,
        dirichlet_alpha: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if let Some(epsilon) = dirichlet_epsilon {
            self.search.set_dirichlet(Some(DirichletConfig {
                epsilon,
                alpha: dirichlet_alpha.unwrap_or(0.3),
            }));
        } else {
            self.search.set_dirichlet(None);
        }

        let temp = temperature.unwrap_or(0.0);
        let result = self
            .search
            .search_with_temperature(&game.state, self.simulations, temp);

        // Build the result dict.
        let dict = PyDict::new(py);
        dict.set_item("best_move", move_to_pydict(py, &result.best_move)?)?;

        // Policy as numpy array.
        let policy_array = Array1::from_vec(result.policy);
        dict.set_item("policy", policy_array.into_pyarray(py))?;

        dict.set_item("value", result.value)?;
        dict.set_item("nodes", result.nodes_searched)?;

        Ok(dict)
    }

    /// Return transposition table statistics as a dict.
    fn tt_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.search.tt_stats();
        let dict = PyDict::new(py);
        dict.set_item("hits", stats.hits)?;
        dict.set_item("misses", stats.misses)?;
        dict.set_item("clears", stats.clears)?;
        dict.set_item("current_size", stats.current_size)?;
        Ok(dict)
    }
}

// ---------------------------------------------------------------------------
// Standalone encoding functions
// ---------------------------------------------------------------------------

/// Encode a game's board state as a numpy array of shape (19, 11, 11).
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

/// Encode a batch of games as a numpy array of shape (N, 19, 11, 11).
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
    let promo_kind = promotion
        .map(|s| {
            PieceKind::parse(s)
                .ok_or_else(|| PyValueError::new_err(format!("invalid promotion piece: {s}")))
        })
        .transpose()?;

    let from = HexCoord::new(from_q, from_r);
    let to = HexCoord::new(to_q, to_r);
    let mv = movegen::Move::new(from, to, None).with_promotion_opt(promo_kind);

    serialization::move_to_index(&mv)
        .ok_or_else(|| PyValueError::new_err("move not in index table"))
}

/// Convert a policy-vector index to a dict {from_q, from_r, to_q, to_r, promotion}.
#[pyfunction]
fn index_to_move(py: Python<'_>, idx: usize) -> PyResult<Bound<'_, PyDict>> {
    let n = serialization::num_move_indices();
    if idx >= n {
        return Err(PyValueError::new_err(format!(
            "index {idx} out of range [0, {n})"
        )));
    }
    let (from, to, promo) = serialization::index_to_move(idx);
    let dict = PyDict::new(py);
    dict.set_item("from_q", from.q)?;
    dict.set_item("from_r", from.r)?;
    dict.set_item("to_q", to.q)?;
    dict.set_item("to_r", to.r)?;
    dict.set_item("promotion", promotion_str(promo))?;
    Ok(dict)
}

/// Total number of move indices in the policy vector.
#[pyfunction]
fn num_move_indices() -> usize {
    serialization::num_move_indices()
}

// ---------------------------------------------------------------------------
// Minimax search
// ---------------------------------------------------------------------------

/// Run alpha-beta minimax search. Returns dict {best_move, score, nodes}.
#[pyfunction]
fn minimax_search<'py>(py: Python<'py>, game: &mut PyGame, depth: u32) -> PyResult<Bound<'py, PyDict>> {
    let result = minimax::search(&mut game.state, depth);
    let dict = PyDict::new(py);
    dict.set_item("best_move", move_to_pydict(py, &result.best_move)?)?;
    dict.set_item("score", result.score)?;
    dict.set_item("nodes", result.nodes)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn hexchess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyMctsSearch>()?;
    m.add_function(wrap_pyfunction!(encode_board, m)?)?;
    m.add_function(wrap_pyfunction!(encode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(move_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_to_move, m)?)?;
    m.add_function(wrap_pyfunction!(num_move_indices, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_search, m)?)?;

    // TENSOR_SHAPE constant
    let tensor_shape = (
        serialization::NUM_CHANNELS,
        serialization::BOARD_DIM,
        serialization::BOARD_DIM,
    );
    m.add("TENSOR_SHAPE", tensor_shape)?;

    Ok(())
}
