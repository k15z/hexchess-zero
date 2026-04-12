//! Monte Carlo Tree Search (AlphaZero variant) for hexagonal chess.
//!
//! Uses arena-style allocation with nodes stored in a `Vec<MctsNode>` and
//! referenced by index, avoiding pointer overhead and improving cache locality.

use std::collections::{HashMap, VecDeque};

/// 95% one-sided normal z-score, used for LCB confidence intervals.
const LCB_Z_95: f64 = 1.96;

use crate::board::Color;
use crate::eval;
use crate::game::GameState;
use crate::movegen::Move;
use crate::serialization;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// SearchConfig
// ---------------------------------------------------------------------------

/// Optional Dirichlet noise parameters for root exploration.
#[derive(Clone, Debug)]
pub struct DirichletConfig {
    /// Mixing weight for noise (typically 0.25).
    pub epsilon: f32,
    /// Dirichlet concentration parameter (typically 0.25 for hex chess).
    pub alpha: f64,
}

impl Default for DirichletConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.25,
            alpha: 0.25,
        }
    }
}

/// Playout Cap Randomization config (KataGo §3.1).
#[derive(Clone, Debug)]
pub struct PcrConfig {
    /// Probability of running a full (noisy, recorded) search.
    pub p_full: f32,
    /// Simulation count for full searches.
    pub n_full: u32,
    /// Simulation count for fast (non-recorded) searches.
    pub n_fast: u32,
}

impl Default for PcrConfig {
    fn default() -> Self {
        Self {
            p_full: 0.25,
            n_full: 800,
            n_fast: 160,
        }
    }
}

/// Resignation config (KataGo §3.3).
#[derive(Clone, Debug)]
pub struct ResignConfig {
    /// Whether resignation is active at all for this game.
    pub enabled: bool,
    /// Win-probability threshold below which STM is considered resigning.
    pub v_resign: f32,
    /// Number of consecutive moves below threshold required to resign.
    pub k: usize,
}

impl Default for ResignConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            v_resign: 0.05,
            k: 5,
        }
    }
}

/// Temperature-decay schedule for training-time move selection.
#[derive(Clone, Debug)]
pub struct TemperatureSchedule {
    /// Maximum temperature at ply 0.
    pub tau_max: f32,
    /// Minimum temperature floor before hard-greedy cutoff.
    pub tau_min: f32,
    /// Half-life in plies for the exponential decay.
    pub halflife: f32,
    /// Ply at which to force hard-greedy (tau = 0).
    pub hard_greedy_ply: u32,
}

impl Default for TemperatureSchedule {
    fn default() -> Self {
        Self {
            tau_max: 1.0,
            tau_min: 0.1,
            halflife: 20.0,
            hard_greedy_ply: 60,
        }
    }
}

impl TemperatureSchedule {
    /// Return the effective temperature for a given ply.
    pub fn temperature_at(&self, ply: u32) -> f32 {
        if ply >= self.hard_greedy_ply {
            return 0.0;
        }
        let raw = self.tau_max * (-(ply as f32) / self.halflife).exp();
        raw.clamp(self.tau_min, self.tau_max)
    }
}

/// Unified configuration for one `MctsSearch`. Training vs eval settings
/// should differ only in a handful of fields.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Base PUCT constant used at non-root nodes (this is the `c1` in the
    /// AlphaZero dynamic formula for interior nodes).
    pub c_puct: f32,
    /// Base PUCT constant used at the root only.
    pub c_puct_root: f32,
    /// Dynamic PUCT denominator constant (`c2` in AlphaZero formula).
    pub c2: f32,
    /// FPU reduction applied to unvisited children: `Q_FPU = Q_parent - fpu_reduction`.
    pub fpu_reduction: f32,
    /// Virtual loss applied during leaf-batched selection.
    pub virtual_loss: f32,
    /// Leaf batch size. 1 means sequential path.
    pub batch_size: usize,
    /// Transposition-table capacity (in entries).
    pub tt_capacity: usize,
    /// Optional Dirichlet noise applied at the root.
    pub dirichlet: Option<DirichletConfig>,
    /// Top-K count used by shaped Dirichlet (half the noise mass is placed on
    /// the top-K prior moves).
    pub dirichlet_top_k: usize,
    /// Forced-playout exponent `k` (KataGo §3.2). 0 disables forced playouts.
    pub forced_playout_k: f32,
    /// Whether to apply policy-target pruning when exporting the training target.
    pub policy_target_pruning: bool,
    /// Temperature schedule for training move selection.
    pub temperature: TemperatureSchedule,
    /// Playout Cap Randomization settings.
    pub pcr: PcrConfig,
    /// Resignation settings.
    pub resign: ResignConfig,
    /// Use LCB (lower confidence bound) for final move selection.
    pub use_lcb: bool,
    /// Draw-utility ("contempt") added to PUCT Q as `draw_utility * D`.
    /// 0.0 preserves pure `W - L` AlphaZero behavior. Positive values make
    /// draws feel like partial wins; negative values are classic Lc0-style
    /// contempt where draws feel like partial losses.
    pub draw_utility: f32,
    /// Treat 2-fold repetition encountered within a search path as an immediate draw.
    pub two_fold_as_draw: bool,
    /// Optional seed for the internal RNG (used for Dirichlet noise sampling
    /// and temperature-based move selection). `None` draws from thread RNG.
    pub rng_seed: Option<u64>,
}

impl SearchConfig {
    /// Default training configuration.
    pub fn training() -> Self {
        Self {
            c_puct: 2.5,
            c_puct_root: 3.5,
            c2: 19652.0,
            fpu_reduction: 0.0,
            virtual_loss: 1.0,
            batch_size: 32,
            tt_capacity: 500_000,
            dirichlet: Some(DirichletConfig::default()),
            dirichlet_top_k: 10,
            forced_playout_k: 2.0,
            policy_target_pruning: true,
            temperature: TemperatureSchedule::default(),
            pcr: PcrConfig::default(),
            resign: ResignConfig::default(),
            use_lcb: false,
            draw_utility: 0.0,
            two_fold_as_draw: true,
            rng_seed: None,
        }
    }

    /// Default evaluation / match-play configuration.
    pub fn eval() -> Self {
        Self {
            c_puct: 2.5,
            c_puct_root: 3.5,
            c2: 19652.0,
            fpu_reduction: 0.2,
            virtual_loss: 1.0,
            batch_size: 32,
            tt_capacity: 500_000,
            dirichlet: None,
            dirichlet_top_k: 10,
            forced_playout_k: 0.0,
            policy_target_pruning: false,
            temperature: TemperatureSchedule {
                tau_max: 0.0,
                tau_min: 0.0,
                halflife: 1.0,
                hard_greedy_ply: 0,
            },
            pcr: PcrConfig {
                p_full: 1.0,
                n_full: 800,
                n_fast: 800,
            },
            resign: ResignConfig {
                enabled: false,
                ..ResignConfig::default()
            },
            use_lcb: true,
            draw_utility: 0.0,
            two_fold_as_draw: true,
            rng_seed: None,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self::training()
    }
}

/// Dynamic PUCT coefficient: `base + log((1 + N + c2) / c2)`.
///
/// `base` is the node-type constant (`c_puct` for interior nodes,
/// `c_puct_root` for the root). At low parent-visit counts the log term is
/// near zero and the constant dominates; past `c2 ≈ 19652` the log term
/// begins to meaningfully scale up exploration.
pub fn dynamic_c_puct(base: f32, c2: f32, parent_visits: u32) -> f32 {
    let n = parent_visits as f32;
    base + ((1.0 + n + c2) / c2).ln()
}

// ---------------------------------------------------------------------------
// Evaluator trait
// ---------------------------------------------------------------------------

/// Trait for position evaluation, returning a policy vector and a WDL
/// probability distribution.
///
/// The policy vector is a probability distribution over move indices (length
/// `serialization::num_move_indices()`). The value is `[W, D, L]` — win,
/// draw, and loss probabilities from the side-to-move's perspective, summing
/// to 1. Historically this was a single scalar `W - L`; carrying the full
/// distribution lets MCTS drive resignation off real `P(W)` and lets an
/// optional draw-contempt knob reshape PUCT Q.
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, state: &GameState) -> (Vec<f32>, [f32; 3]);

    /// Evaluate multiple positions in a single batch. The default implementation
    /// calls `evaluate` sequentially; backends that support batched inference
    /// (e.g. ONNX Runtime) should override this for better throughput.
    fn evaluate_batch(&self, states: &[&GameState]) -> Vec<(Vec<f32>, [f32; 3])> {
        states.iter().map(|s| self.evaluate(s)).collect()
    }
}

/// Project a scalar value in `[-1, 1]` into a WDL distribution with zero
/// draw mass: `[(1+v)/2, 0, (1-v)/2]`. This preserves `W - L == v` exactly
/// and is used by non-NN evaluators that only produce a scalar.
pub fn scalar_to_wdl(v: f32) -> [f32; 3] {
    let v = v.clamp(-1.0, 1.0);
    [0.5 * (1.0 + v), 0.0, 0.5 * (1.0 - v)]
}

// ---------------------------------------------------------------------------
// HeuristicEvaluator
// ---------------------------------------------------------------------------

/// Uniform policy over legal moves with material-aware value estimate.
pub struct HeuristicEvaluator;

impl Evaluator for HeuristicEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<f32>, [f32; 3]) {
        let moves = state.legal_moves();
        let num_indices = serialization::num_move_indices();
        let mut policy = vec![0.0f32; num_indices];
        let stm = state.board.side_to_move;

        if !moves.is_empty() {
            let prob = 1.0 / moves.len() as f32;
            for mv in &moves {
                if let Some(idx) = serialization::stm_policy_index(mv, stm) {
                    policy[idx] = prob;
                }
            }
        }

        let cp = eval::evaluate(state) as f32;
        let value = (cp / eval::CP_TANH_SCALE).tanh();

        (policy, scalar_to_wdl(value))
    }
}

// ---------------------------------------------------------------------------
// WeightedHeuristicEvaluator
// ---------------------------------------------------------------------------

/// One-ply lookahead policy with full positional evaluation.
pub struct WeightedHeuristicEvaluator {
    weights: eval::EvalWeights,
    policy_temperature: f32,
}

impl WeightedHeuristicEvaluator {
    pub fn new(weights: eval::EvalWeights) -> Self {
        Self {
            weights,
            policy_temperature: 200.0,
        }
    }

    pub fn with_policy_temperature(mut self, temperature: f32) -> Self {
        self.policy_temperature = temperature;
        self
    }
}

impl Evaluator for WeightedHeuristicEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<f32>, [f32; 3]) {
        let moves = state.legal_moves();
        let num_indices = serialization::num_move_indices();
        let mut policy = vec![0.0f32; num_indices];
        let stm = state.board.side_to_move;

        if !moves.is_empty() {
            // Score each move: apply, evaluate from opponent's perspective, negate.
            let mut scores: Vec<(usize, f32)> = Vec::with_capacity(moves.len());
            let mut max_score = f32::NEG_INFINITY;
            let mut state_clone = state.clone();

            for mv in &moves {
                if let Some(idx) = serialization::stm_policy_index(mv, stm) {
                    state_clone.apply_move(*mv);
                    let score =
                        -eval::evaluate_board_weighted(&state_clone.board, &self.weights) as f32;
                    state_clone.undo_move();
                    if score > max_score {
                        max_score = score;
                    }
                    scores.push((idx, score));
                }
            }

            // Softmax with temperature.
            let mut sum = 0.0f32;
            for (_, score) in &mut scores {
                let exp = ((*score - max_score) / self.policy_temperature).exp();
                *score = exp;
                sum += exp;
            }
            if sum > 0.0 {
                for (idx, prob) in scores {
                    policy[idx] = prob / sum;
                }
            }
        }

        let cp = eval::evaluate_weighted(state, &self.weights) as f32;
        let value = (cp / eval::CP_TANH_SCALE).tanh();

        (policy, scalar_to_wdl(value))
    }
}

// ---------------------------------------------------------------------------
// MCTS Node (arena-allocated)
// ---------------------------------------------------------------------------

struct MctsNode {
    action: Option<Move>,
    action_index: Option<usize>,
    children: Vec<usize>,
    visit_count: u32,
    /// Running sum of backed-up WDL probability vectors `[W, D, L]`, always
    /// from this node's side-to-move perspective. Backprop swaps W and L on
    /// each parent hop.
    value_sum: [f64; 3],
    /// Welford M2 accumulator (sum of squared deviations) of the scalar
    /// `W - L` projections of each backup, used for LCB confidence intervals
    /// on the scalar choice function. Tracking variance on the WDL vector
    /// would require a covariance matrix for no added signal since LCB
    /// picks a single move on the scalar axis.
    m2: f64,
    prior: f32,
    is_expanded: bool,
}

impl MctsNode {
    fn new(action: Option<Move>, action_index: Option<usize>, prior: f32) -> Self {
        Self {
            action,
            action_index,
            children: Vec::new(),
            visit_count: 0,
            value_sum: [0.0; 3],
            m2: 0.0,
            prior,
            is_expanded: false,
        }
    }

    /// Average WDL distribution `[W, D, L]` over this node's visits.
    fn q_wdl(&self) -> [f64; 3] {
        if self.visit_count == 0 {
            return [0.0; 3];
        }
        let n = self.visit_count as f64;
        [
            self.value_sum[0] / n,
            self.value_sum[1] / n,
            self.value_sum[2] / n,
        ]
    }

    /// Scalar `W - L` projection of the node's average WDL. This is the
    /// canonical AlphaZero Q and is what PUCT compares when `draw_utility`
    /// is 0. Currently only used in tests; the search itself goes through
    /// [`Self::q_value_contempt`] which reduces to this when contempt is 0.
    #[cfg_attr(not(test), allow(dead_code))]
    fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            let n = self.visit_count as f64;
            (self.value_sum[0] - self.value_sum[2]) / n
        }
    }

    /// Contempt-adjusted scalar Q: `W - L + draw_utility * D`. With
    /// `draw_utility == 0` this is exactly [`Self::q_value`]; positive
    /// values treat draws as partial wins (appropriate for the weaker
    /// side) and negative values treat draws as partial losses (classic
    /// contempt for the stronger side).
    fn q_value_contempt(&self, draw_utility: f64) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            let n = self.visit_count as f64;
            (self.value_sum[0] - self.value_sum[2] + draw_utility * self.value_sum[1]) / n
        }
    }

    /// Sample variance of the scalar `W - L` backups (Welford / (n-1)).
    /// Zero when fewer than 2 samples are present.
    fn variance(&self) -> f64 {
        if self.visit_count < 2 {
            0.0
        } else {
            self.m2 / (self.visit_count as f64 - 1.0)
        }
    }
}

/// Map a terminal `GameState` to a 1-hot WDL vector from the side-to-move
/// perspective. `None` returns a neutral draw, matching the legacy scalar
/// fallback of `outcome_value().unwrap_or(0.0)`.
fn terminal_wdl(state: &GameState) -> [f64; 3] {
    match state.outcome_value() {
        Some(v) if v > 0.5 => [1.0, 0.0, 0.0],
        Some(v) if v < -0.5 => [0.0, 0.0, 1.0],
        _ => [0.0, 1.0, 0.0],
    }
}

/// Swap the win and loss components of a WDL vector. Applied at each parent
/// hop during backpropagation because the STM changes.
#[inline]
fn flip_wdl(wdl: [f64; 3]) -> [f64; 3] {
    [wdl[2], wdl[1], wdl[0]]
}

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------

pub struct SearchResult {
    /// Best move chosen by the search.
    pub best_move: Move,
    /// Visit-count distribution over all move indices (normalized to sum to 1).
    pub policy: Vec<f32>,
    /// Scalar value estimate `W - L` of the root position, for back-compat
    /// with callers that only want a single number. Equals `wdl[0] - wdl[2]`.
    pub value: f32,
    /// Average WDL distribution `[W, D, L]` at the root, from the root's
    /// side-to-move perspective. Preserves the draw mass that callers need
    /// for resignation (`P(W) < threshold`) and calibration.
    pub wdl: [f32; 3],
    /// Visit count at the root node (= number of simulations whose
    /// backups reached the root). For a search with no early-terminal
    /// hits, this equals the requested simulation count.
    pub nodes_searched: u32,
}

// ---------------------------------------------------------------------------
// MCTS Search
// ---------------------------------------------------------------------------

/// Cumulative transposition table statistics.
#[derive(Clone, Debug, Default)]
pub struct TtStats {
    pub hits: u64,
    pub misses: u64,
    pub clears: u64,
    pub current_size: usize,
}

/// Transposition-table key. Positions with identical zobrist/side but
/// different repetition counts are semantically distinct because a
/// third occurrence is a forced draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TtKey {
    /// Raw zobrist hash of the board.
    pub zobrist: u64,
    /// Side to move (0 = white, 1 = black).
    pub side: u8,
    /// Repetition count of this exact position in the game history.
    pub repetition: u32,
}

impl TtKey {
    pub fn from_state(state: &GameState) -> Self {
        Self {
            zobrist: state.board.zobrist_hash,
            side: state.board.side_to_move.index() as u8,
            repetition: state.repetition_count(),
        }
    }
}

/// Outcome of walking the tree from root to a leaf in one simulation.
enum DescendOutcome {
    /// Reached an unexpanded leaf (or a terminal state).
    Leaf,
    /// Selection hit a position already seen earlier on the same descent
    /// path, which `two_fold_as_draw` treats as an immediate draw.
    TwoFoldDraw,
}

struct LeafInfo {
    path: Vec<usize>,
    leaf_state: Option<GameState>,
    terminal_wdl: [f64; 3],
    key: TtKey,
}

pub struct MctsSearch {
    nodes: Vec<MctsNode>,
    evaluator: Box<dyn Evaluator>,
    config: SearchConfig,
    tt: HashMap<TtKey, (Vec<f32>, [f32; 3])>,
    tt_hits: u64,
    tt_misses: u64,
    tt_clears: u64,
    /// Rolling buffer of STM P(W) at the root for the last `k` moves of
    /// each side, used by `should_resign`.
    resign_history: [VecDeque<f32>; 2],
    /// Reusable scratch path buffer for `simulate` (avoids per-sim alloc).
    path_scratch: Vec<usize>,
    /// Reusable buffer of leaves for `simulate_batched`.
    leaves_scratch: Vec<LeafInfo>,
    /// Seeded RNG for Dirichlet noise and temperature sampling. `None` falls
    /// back to thread-local RNG.
    rng: Option<StdRng>,
    /// Index of the child selected by the most recent `extract_result` call.
    /// Used by `aux_opponent_policy` to report grandchildren of the actually
    /// played child, not just the visit-max child.
    last_selected_child_idx: Option<usize>,
}

impl MctsSearch {
    pub fn new(evaluator: Box<dyn Evaluator>) -> Self {
        let config = SearchConfig::training();
        let rng = config.rng_seed.map(StdRng::seed_from_u64);
        Self {
            nodes: Vec::new(),
            evaluator,
            config,
            tt: HashMap::new(),
            tt_hits: 0,
            tt_misses: 0,
            tt_clears: 0,
            resign_history: [VecDeque::new(), VecDeque::new()],
            path_scratch: Vec::new(),
            leaves_scratch: Vec::new(),
            rng,
            last_selected_child_idx: None,
        }
    }

    /// Seed the internal RNG used for Dirichlet noise and temperature
    /// sampling so searches become deterministic given the same inputs.
    pub fn set_rng_seed(&mut self, seed: u64) {
        self.config.rng_seed = Some(seed);
        self.rng = Some(StdRng::seed_from_u64(seed));
    }

    /// Replace the entire search configuration.
    pub fn set_config(&mut self, config: SearchConfig) {
        self.rng = config.rng_seed.map(StdRng::seed_from_u64);
        self.config = config;
    }

    /// Immutable view of the current search configuration.
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// Mutable view of the current search configuration.
    pub fn config_mut(&mut self) -> &mut SearchConfig {
        &mut self.config
    }

    /// Compatibility shim for legacy bindings that only knew a single
    /// `c_puct`. Sets `c_puct` to the given value and `c_puct_root` to
    /// `c_puct + 1.0` (the historical offset). New code should prefer
    /// `config_mut()` and set both fields explicitly.
    pub fn set_c_puct(&mut self, c_puct: f32) {
        self.config.c_puct = c_puct;
        self.config.c_puct_root = c_puct + 1.0;
    }

    pub fn set_tt_capacity(&mut self, capacity: usize) {
        self.config.tt_capacity = capacity;
    }

    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.config.batch_size = batch_size.max(1);
    }

    pub fn set_dirichlet(&mut self, config: Option<DirichletConfig>) {
        self.config.dirichlet = config;
    }

    /// Set the draw-utility ("contempt") knob in the current config. Positive
    /// values bias search toward draws, negative values bias away. 0 recovers
    /// pure `W - L` AlphaZero behavior.
    pub fn set_draw_utility(&mut self, draw_utility: f32) {
        self.config.draw_utility = draw_utility;
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.last_selected_child_idx = None;
    }

    fn tt_insert(&mut self, key: TtKey, entry: (Vec<f32>, [f32; 3])) {
        if self.tt.len() >= self.config.tt_capacity {
            self.tt.clear();
            self.tt_clears += 1;
        }
        self.tt.insert(key, entry);
    }

    /// Return cumulative transposition table statistics.
    pub fn tt_stats(&self) -> TtStats {
        TtStats {
            hits: self.tt_hits,
            misses: self.tt_misses,
            clears: self.tt_clears,
            current_size: self.tt.len(),
        }
    }

    /// Clear everything including the transposition table.
    pub fn reset_all(&mut self) {
        self.nodes.clear();
        self.tt.clear();
        self.last_selected_child_idx = None;
    }

    /// Run MCTS for `num_simulations` iterations (greedy, temperature = 0).
    pub fn search(&mut self, state: &GameState, num_simulations: u32) -> SearchResult {
        self.search_with_options(state, num_simulations, 0.0, true)
    }

    /// Run MCTS for `num_simulations` iterations with the given temperature.
    ///
    /// * `temperature = 0` (or very small): pick the move with highest visit count.
    /// * `temperature = 1`: sample proportionally to visit counts.
    /// * `temperature > 1`: more exploration.
    pub fn search_with_temperature(
        &mut self,
        state: &GameState,
        num_simulations: u32,
        temperature: f32,
    ) -> SearchResult {
        self.search_with_options(state, num_simulations, temperature, true)
    }

    /// Core search entry point. `enable_noise = false` disables root Dirichlet
    /// noise for this call only, without mutating `self.config`.
    fn search_with_options(
        &mut self,
        state: &GameState,
        num_simulations: u32,
        temperature: f32,
        enable_noise: bool,
    ) -> SearchResult {
        self.reset();

        let root = MctsNode::new(None, None, 1.0);
        self.nodes.push(root);

        let mut working_state = state.clone();

        let noise_gate = enable_noise;
        let saved = if !noise_gate {
            self.config.dirichlet.take()
        } else {
            None
        };

        // Scope guard via a local helper struct would be nicer, but we
        // explicitly restore below — `expand` never panics in practice.
        if self.config.batch_size <= 1 {
            for _ in 0..num_simulations {
                self.simulate(0, &mut working_state);
            }
        } else {
            self.simulate_batched(0, &mut working_state, num_simulations);
        }

        if !noise_gate {
            self.config.dirichlet = saved;
        }

        self.extract_result(state, temperature)
    }

    // ------------------------------------------------------------------
    // Internal helpers — sequential path
    // ------------------------------------------------------------------

    /// Walk from `root_idx` to an unexpanded leaf (or terminal / 2-fold
    /// draw). `state` is mutated via `apply_move`; the caller is
    /// responsible for undoing the applied moves. `path` is cleared and
    /// populated with the traversed node indices.
    fn descend_to_leaf(
        &self,
        root_idx: usize,
        state: &mut GameState,
        path: &mut Vec<usize>,
    ) -> DescendOutcome {
        path.clear();
        path.push(root_idx);
        let mut node_idx = root_idx;

        while self.nodes[node_idx].is_expanded && !self.nodes[node_idx].children.is_empty() {
            node_idx = self.select_child(node_idx, node_idx == root_idx);
            let action = self.nodes[node_idx]
                .action
                .expect("non-root node must have an action");
            state.apply_move(action);
            path.push(node_idx);
            if self.config.two_fold_as_draw && state.repetition_count() >= 1 {
                return DescendOutcome::TwoFoldDraw;
            }
        }
        DescendOutcome::Leaf
    }

    fn simulate(&mut self, root_idx: usize, state: &mut GameState) {
        let mut path = std::mem::take(&mut self.path_scratch);
        let outcome = self.descend_to_leaf(root_idx, state, &mut path);
        let node_idx = *path.last().unwrap();
        let moves_made = path.len() - 1;

        let wdl = match outcome {
            DescendOutcome::TwoFoldDraw => [0.0, 1.0, 0.0],
            DescendOutcome::Leaf => {
                if state.is_game_over() {
                    terminal_wdl(state)
                } else {
                    let key = TtKey::from_state(state);
                    if let Some(cached) = self.tt.get(&key) {
                        self.tt_hits += 1;
                        let cached_wdl = cached.1;
                        Self::expand_nodes(
                            &mut self.nodes,
                            &self.config,
                            self.rng.as_mut(),
                            node_idx,
                            state,
                            &cached.0,
                        );
                        [
                            cached_wdl[0] as f64,
                            cached_wdl[1] as f64,
                            cached_wdl[2] as f64,
                        ]
                    } else {
                        self.tt_misses += 1;
                        let (policy, wdl) = self.evaluator.evaluate(state);
                        Self::expand_nodes(
                            &mut self.nodes,
                            &self.config,
                            self.rng.as_mut(),
                            node_idx,
                            state,
                            &policy,
                        );
                        self.tt_insert(key, (policy, wdl));
                        [wdl[0] as f64, wdl[1] as f64, wdl[2] as f64]
                    }
                }
            }
        };

        self.backpropagate(&path, wdl);

        for _ in 0..moves_made {
            state.undo_move();
        }

        self.path_scratch = path;
    }

    // ------------------------------------------------------------------
    // Internal helpers — batched path with virtual loss
    // ------------------------------------------------------------------

    /// Run `num_simulations` simulations using batched NN inference.
    fn simulate_batched(&mut self, root_idx: usize, state: &mut GameState, num_simulations: u32) {
        let batch_size = self.config.batch_size;
        let mut done = 0u32;
        let mut leaves = std::mem::take(&mut self.leaves_scratch);
        let mut eval_indices: Vec<usize> = Vec::with_capacity(batch_size);

        while done < num_simulations {
            let batch_count = ((num_simulations - done) as usize).min(batch_size);
            leaves.clear();
            eval_indices.clear();

            for _ in 0..batch_count {
                let mut path: Vec<usize> = Vec::new();
                let outcome = self.descend_to_leaf(root_idx, state, &mut path);
                let node_idx = *path.last().unwrap();
                let key = TtKey::from_state(state);
                let moves_made = path.len() - 1;

                match outcome {
                    DescendOutcome::TwoFoldDraw => {
                        self.apply_virtual_loss(&path);
                        leaves.push(LeafInfo {
                            path,
                            leaf_state: None,
                            terminal_wdl: [0.0, 1.0, 0.0],
                            key,
                        });
                    }
                    DescendOutcome::Leaf if state.is_game_over() => {
                        let wdl = terminal_wdl(state);
                        self.apply_virtual_loss(&path);
                        leaves.push(LeafInfo {
                            path,
                            leaf_state: None,
                            terminal_wdl: wdl,
                            key,
                        });
                    }
                    DescendOutcome::Leaf => {
                        if let Some(cached) = self.tt.get(&key) {
                            self.tt_hits += 1;
                            let cached_wdl = cached.1;
                            Self::expand_nodes(
                                &mut self.nodes,
                                &self.config,
                                self.rng.as_mut(),
                                node_idx,
                                state,
                                &cached.0,
                            );
                            self.apply_virtual_loss(&path);
                            leaves.push(LeafInfo {
                                path,
                                leaf_state: None,
                                terminal_wdl: [
                                    cached_wdl[0] as f64,
                                    cached_wdl[1] as f64,
                                    cached_wdl[2] as f64,
                                ],
                                key,
                            });
                        } else {
                            self.tt_misses += 1;
                            let snapshot = state.clone();
                            self.apply_virtual_loss(&path);
                            leaves.push(LeafInfo {
                                path,
                                leaf_state: Some(snapshot),
                                terminal_wdl: [0.0; 3],
                                key,
                            });
                        }
                    }
                }

                for _ in 0..moves_made {
                    state.undo_move();
                }
            }

            let mut seen_keys: HashMap<TtKey, usize> = HashMap::new();
            for (i, l) in leaves.iter().enumerate() {
                if l.leaf_state.is_some() && !seen_keys.contains_key(&l.key) {
                    seen_keys.insert(l.key, eval_indices.len());
                    eval_indices.push(i);
                }
            }

            let eval_results: Vec<(Vec<f32>, [f32; 3])> = if eval_indices.is_empty() {
                Vec::new()
            } else {
                let states_for_eval: Vec<&GameState> = eval_indices
                    .iter()
                    .map(|&i| leaves[i].leaf_state.as_ref().unwrap())
                    .collect();
                self.evaluator.evaluate_batch(&states_for_eval)
            };

            for leaf in &mut leaves {
                self.remove_virtual_loss(&leaf.path);

                let wdl = if let Some(ref leaf_state) = leaf.leaf_state {
                    let result_idx = seen_keys[&leaf.key];
                    let (policy, wdl) = &eval_results[result_idx];
                    let node_idx = *leaf.path.last().unwrap();

                    if !self.nodes[node_idx].is_expanded {
                        Self::expand_nodes(
                            &mut self.nodes,
                            &self.config,
                            self.rng.as_mut(),
                            node_idx,
                            leaf_state,
                            policy,
                        );
                        self.tt_insert(leaf.key, (policy.clone(), *wdl));
                    }
                    [wdl[0] as f64, wdl[1] as f64, wdl[2] as f64]
                } else {
                    leaf.terminal_wdl
                };

                self.backpropagate(&leaf.path, wdl);
            }

            done += batch_count as u32;
        }

        leaves.clear();
        self.leaves_scratch = leaves;
    }

    /// Apply virtual loss along a path: increment visit counts and add
    /// VIRTUAL_LOSS to the win channel of the node's value sum to discourage
    /// re-selection.
    ///
    /// Rationale: PUCT uses `q = -child.q_value_contempt()` (negated for
    /// parent's perspective). Bumping the `W` component of `value_sum` makes
    /// `q_value()` more positive, so `-q_value()` becomes more negative,
    /// lowering the PUCT score and discouraging the parent from re-selecting
    /// this child. Virtual loss has no effect on contempt-adjusted Q beyond
    /// the scalar W-L axis because we only touch the `W` channel.
    fn apply_virtual_loss(&mut self, path: &[usize]) {
        let vl = self.config.virtual_loss as f64;
        for &idx in path {
            self.nodes[idx].visit_count += 1;
            self.nodes[idx].value_sum[0] += vl;
        }
    }

    fn remove_virtual_loss(&mut self, path: &[usize]) {
        let vl = self.config.virtual_loss as f64;
        for &idx in path {
            self.nodes[idx].visit_count -= 1;
            self.nodes[idx].value_sum[0] -= vl;
        }
    }

    /// Select the child of `node_idx` with the highest PUCT score. Root nodes
    /// also honor the forced-playout rule (KataGo §3.2).
    fn select_child(&self, node_idx: usize, is_root: bool) -> usize {
        let parent = &self.nodes[node_idx];
        let parent_visits = parent.visit_count;
        let parent_visits_sqrt = (parent_visits as f64).sqrt();
        let draw_utility = self.config.draw_utility as f64;
        let parent_q = parent.q_value_contempt(draw_utility);

        let base = if is_root {
            self.config.c_puct_root
        } else {
            self.config.c_puct
        };
        let c = dynamic_c_puct(base, self.config.c2, parent_visits) as f64;
        let fpu = self.config.fpu_reduction as f64;
        let q_fpu = parent_q - fpu;

        if is_root && self.config.forced_playout_k > 0.0 {
            let k = self.config.forced_playout_k as f64;
            let sqrt_n = parent_visits_sqrt;
            for &child_idx in &parent.children {
                let child = &self.nodes[child_idx];
                let n_forced = (k * child.prior as f64 * sqrt_n).ceil() as u32;
                if child.visit_count < n_forced && child.prior > 0.0 {
                    return child_idx;
                }
            }
        }

        let mut best_idx = parent.children[0];
        let mut best_score = f64::NEG_INFINITY;

        for &child_idx in &parent.children {
            let child = &self.nodes[child_idx];
            let q = if child.visit_count == 0 {
                q_fpu
            } else {
                -child.q_value_contempt(draw_utility)
            };
            let u = c * child.prior as f64 * parent_visits_sqrt / (1.0 + child.visit_count as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }

        best_idx
    }

    /// Expand `node_idx` by creating children for all legal moves.
    ///
    /// Takes explicit `&mut Vec<MctsNode>` and `&SearchConfig` (rather than
    /// `&mut self`) so that callers can hold a live borrow on `self.tt`
    /// across the call.
    fn expand_nodes(
        nodes: &mut Vec<MctsNode>,
        config: &SearchConfig,
        rng: Option<&mut StdRng>,
        node_idx: usize,
        state: &GameState,
        policy: &[f32],
    ) {
        let legal_moves = state.legal_moves();
        if legal_moves.is_empty() {
            return;
        }

        let mut children_info: Vec<(Move, usize, f32)> = Vec::with_capacity(legal_moves.len());
        let mut prior_sum = 0.0f32;
        let stm = state.board.side_to_move;

        for mv in &legal_moves {
            // STM-frame policy index: the NN (and the heuristic evaluator)
            // both emit policies in STM frame, so children's action_index
            // is STM-relative. This propagates through extract_result /
            // policy_target_pruned / aux_opponent_policy so the emitted
            // training target is also STM-relative by construction.
            if let Some(idx) = serialization::stm_policy_index(mv, stm) {
                let p = if idx < policy.len() { policy[idx] } else { 0.0 };
                prior_sum += p;
                children_info.push((*mv, idx, p));
            } else {
                // Move not in index table — give it a small prior.
                children_info.push((*mv, usize::MAX, 0.0));
            }
        }

        if prior_sum > 1e-8 {
            for info in &mut children_info {
                info.2 /= prior_sum;
            }
        } else {
            // Uniform fallback.
            let uniform = 1.0 / children_info.len() as f32;
            for info in &mut children_info {
                info.2 = uniform;
            }
        }

        if node_idx == 0
            && let Some(ref dcfg) = config.dirichlet
        {
            let top_k = config.dirichlet_top_k;
            match rng {
                Some(r) => apply_shaped_dirichlet_noise(
                    &mut children_info,
                    dcfg.epsilon,
                    dcfg.alpha,
                    top_k,
                    r,
                ),
                None => {
                    let mut thread = rand::rng();
                    apply_shaped_dirichlet_noise(
                        &mut children_info,
                        dcfg.epsilon,
                        dcfg.alpha,
                        top_k,
                        &mut thread,
                    );
                }
            }
        }

        for (mv, mv_idx, prior) in children_info {
            let action_index = if mv_idx == usize::MAX {
                None
            } else {
                Some(mv_idx)
            };
            let child = MctsNode::new(Some(mv), action_index, prior);
            let child_idx = nodes.len();
            nodes.push(child);
            nodes[node_idx].children.push(child_idx);
        }

        nodes[node_idx].is_expanded = true;
    }

    fn backpropagate(&mut self, path: &[usize], leaf_wdl: [f64; 3]) {
        let mut wdl = leaf_wdl;
        let track_variance = self.config.use_lcb;
        for &node_idx in path.iter().rev() {
            let node = &mut self.nodes[node_idx];
            if track_variance {
                // Welford update on the scalar W-L projection. This is the
                // only dimension LCB actually consumes — tracking a full
                // WDL covariance would not change any move choice.
                let scalar = wdl[0] - wdl[2];
                let old_mean = if node.visit_count == 0 {
                    0.0
                } else {
                    (node.value_sum[0] - node.value_sum[2]) / node.visit_count as f64
                };
                node.visit_count += 1;
                node.value_sum[0] += wdl[0];
                node.value_sum[1] += wdl[1];
                node.value_sum[2] += wdl[2];
                let new_mean = (node.value_sum[0] - node.value_sum[2]) / node.visit_count as f64;
                let delta = scalar - old_mean;
                let delta2 = scalar - new_mean;
                node.m2 += delta * delta2;
            } else {
                node.visit_count += 1;
                node.value_sum[0] += wdl[0];
                node.value_sum[1] += wdl[1];
                node.value_sum[2] += wdl[2];
            }
            // STM flips at the parent, so the parent sees W and L swapped.
            wdl = flip_wdl(wdl);
        }
    }

    /// Extract the search result after all simulations.
    fn extract_result(&mut self, _state: &GameState, temperature: f32) -> SearchResult {
        let num_indices = serialization::num_move_indices();
        let mut policy = vec![0.0f32; num_indices];
        let root_children = self.nodes[0].children.clone();

        let total_child_visits: u32 = root_children
            .iter()
            .map(|&i| self.nodes[i].visit_count)
            .sum();

        for &child_idx in &root_children {
            let child = &self.nodes[child_idx];
            if let Some(mv_idx) = child.action_index
                && mv_idx < num_indices
            {
                policy[mv_idx] = child.visit_count as f32;
            }
        }

        let best_child_idx = if temperature < 1e-4 {
            // Greedy: prefer LCB when enabled (eval mode). Falls back to
            // visit-max if no child has any visits yet.
            if self.config.use_lcb {
                self.lcb_best_child_idx(&root_children).unwrap_or_else(|| {
                    *root_children
                        .iter()
                        .max_by_key(|&&i| self.nodes[i].visit_count)
                        .expect("root must have children")
                })
            } else {
                *root_children
                    .iter()
                    .max_by_key(|&&i| self.nodes[i].visit_count)
                    .expect("root must have children")
            }
        } else {
            self.select_by_temperature(&root_children, temperature)
        };

        // Record which child was selected so aux_opponent_policy can report
        // grandchildren of the actually played move, not just visit-max.
        self.last_selected_child_idx = Some(best_child_idx);

        let best_move = self.nodes[best_child_idx]
            .action
            .expect("child must have an action");

        if total_child_visits > 0 {
            if temperature < 1e-4 {
                // For greedy, just normalize raw visit counts.
                let sum: f32 = policy.iter().sum();
                if sum > 0.0 {
                    for p in &mut policy {
                        *p /= sum;
                    }
                }
            } else {
                // Apply temperature via log-space to avoid f32 overflow.
                // p_i' = exp(log(p_i) / temp), then normalize.
                let inv_temp = 1.0 / temperature;
                let mut max_log = f32::NEG_INFINITY;
                for p in policy.iter() {
                    if *p > 0.0 {
                        let lp = p.ln() * inv_temp;
                        if lp > max_log {
                            max_log = lp;
                        }
                    }
                }
                // Subtract max for numerical stability, then exponentiate
                for p in &mut policy {
                    if *p > 0.0 {
                        *p = (p.ln() * inv_temp - max_log).exp();
                    }
                }
                let sum: f32 = policy.iter().sum();
                if sum > 0.0 {
                    for p in &mut policy {
                        *p /= sum;
                    }
                }
            }
        }

        let root_wdl = self.nodes[0].q_wdl();
        let wdl = [root_wdl[0] as f32, root_wdl[1] as f32, root_wdl[2] as f32];
        let value = wdl[0] - wdl[2];

        SearchResult {
            best_move,
            policy,
            value,
            wdl,
            // Root visit count, not arena size: this is what the Python
            // consumer records as `root_n` on each training sample.
            nodes_searched: self.nodes[0].visit_count,
        }
    }

    /// Select a child of root proportionally to visit_count^(1/temperature).
    fn select_by_temperature(&mut self, root_children: &[usize], temperature: f32) -> usize {
        let inv_temp = 1.0 / temperature;
        let weights: Vec<f64> = root_children
            .iter()
            .map(|&i| (self.nodes[i].visit_count as f64).powf(inv_temp as f64))
            .collect();

        let total: f64 = weights.iter().sum();
        if total == 0.0 {
            return root_children[0];
        }

        // Sample from the categorical distribution defined by weights.
        let threshold: f64 = match self.rng.as_mut() {
            Some(rng) => rng.random::<f64>() * total,
            None => {
                let mut thread = rand::rng();
                thread.random::<f64>() * total
            }
        };
        let mut cumulative = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                return root_children[i];
            }
        }
        *root_children.last().unwrap()
    }

    // ------------------------------------------------------------------
    // Public high-level API (v2 rebuild)
    // ------------------------------------------------------------------

    /// Run a search using the current config's temperature schedule for
    /// the given ply. Used by the self-play worker for training games.
    pub fn run_for_ply(
        &mut self,
        state: &GameState,
        ply: u32,
        num_simulations: u32,
    ) -> SearchResult {
        let temperature = self.config.temperature.temperature_at(ply);
        self.search_with_temperature(state, num_simulations, temperature)
    }

    /// Playout Cap Randomization outcome: the caller flips the PCR coin via
    /// `run_pcr`, which dispatches internally to a full (noisy, recorded)
    /// search or a fast (clean, unrecorded) search.
    pub fn run_pcr<R: Rng + ?Sized>(
        &mut self,
        state: &GameState,
        ply: u32,
        rng: &mut R,
    ) -> PcrOutcome {
        let pcr = self.config.pcr.clone();
        let coin: f32 = rng.random();
        let was_full = coin < pcr.p_full;
        let sims = if was_full { pcr.n_full } else { pcr.n_fast };
        // Full searches use the temperature schedule for exploration; fast
        // searches use greedy (temp=0) to be cheap but strong/clean, matching
        // the KataGo PCR design where fast moves don't add noise.
        let temperature = if was_full {
            self.config.temperature.temperature_at(ply)
        } else {
            0.0
        };
        let result = self.search_with_options(state, sims, temperature, was_full);

        let policy_target = if was_full {
            if self.config.policy_target_pruning {
                Some(self.policy_target_pruned())
            } else {
                Some(result.policy.clone())
            }
        } else {
            None
        };

        PcrOutcome {
            best_move: result.best_move,
            value: result.value,
            wdl: result.wdl,
            nodes_searched: result.nodes_searched,
            was_full_search: was_full,
            policy_target,
        }
    }

    /// Compute the PTP-pruned visit-count policy target from the current root.
    ///
    /// Children whose visit count exceeds `n_forced(a) = ceil(k·P·sqrt(N))`
    /// are unaffected; forced children are processed in increasing-visit
    /// order and have their forced visits subtracted from the emitted target
    /// so long as the subtraction does not change the argmax.
    pub fn policy_target_pruned(&self) -> Vec<f32> {
        let num_indices = serialization::num_move_indices();
        let mut policy = vec![0.0f32; num_indices];
        if self.nodes.is_empty() {
            return policy;
        }
        let root = &self.nodes[0];
        if root.children.is_empty() {
            return policy;
        }

        let parent_visits_sqrt = (root.visit_count as f64).sqrt();
        let k = self.config.forced_playout_k as f64;

        let mut adjusted: Vec<(usize, i64, i64)> = root
            .children
            .iter()
            .map(|&ci| {
                let c = &self.nodes[ci];
                let n_forced = if k > 0.0 {
                    (k * c.prior as f64 * parent_visits_sqrt).ceil() as i64
                } else {
                    0
                };
                (ci, c.visit_count as i64, n_forced)
            })
            .collect();

        let raw_argmax_visits = adjusted.iter().map(|e| e.1).max().unwrap_or(0);
        let argmax_child = adjusted
            .iter()
            .find(|e| e.1 == raw_argmax_visits)
            .map(|e| e.0)
            .unwrap_or(root.children[0]);

        let mut forced_order: Vec<usize> = (0..adjusted.len())
            .filter(|&i| adjusted[i].2 > 0 && adjusted[i].0 != argmax_child)
            .collect();
        forced_order.sort_by_key(|&i| adjusted[i].1);

        // Invariant: `forced_order` excludes the argmax child, so subtracting
        // `n_forced` from any entry in it can only lower that entry's visit
        // count — the argmax child is untouched and therefore the argmax
        // cannot change. No tentative recomputation is needed.
        for i in forced_order {
            let (_ci, n_visits, n_forced) = adjusted[i];
            let target = (n_visits - n_forced).max(0);
            adjusted[i].1 = target;
        }
        let _ = raw_argmax_visits;

        let total: i64 = adjusted.iter().map(|e| e.1).sum();
        if total <= 0 {
            return policy;
        }
        let inv = 1.0 / total as f32;
        for (ci, n, _) in &adjusted {
            let child = &self.nodes[*ci];
            if let Some(mv_idx) = child.action_index
                && mv_idx < num_indices
            {
                policy[mv_idx] = (*n as f32) * inv;
            }
        }
        policy
    }

    /// Build an auxiliary policy target representing the visit distribution
    /// over the opponent's reply (the grandchildren of the root, under the
    /// child `extract_result`'s greedy branch would play). Uses LCB selection
    /// when `config.use_lcb` is set to stay consistent with the played move
    /// in eval mode, otherwise falls back to visit-max. Returns `None` if
    /// the root has no children, or the selected child is unexpanded / has
    /// no visits.
    ///
    /// Frame: the returned vector is indexed in the **root's STM frame**,
    /// matching the main policy target emitted by [`Self::extract_result`]
    /// and the board tensor produced by [`serialization::encode_board`] at
    /// the root. The grandchildren were expanded in the opponent's STM
    /// frame (since `expand_nodes` uses `state.board.side_to_move`), so
    /// each `gc.action_index` is remapped through `mirror_move_index` to
    /// reach the root frame — the two STM frames differ by exactly one
    /// color swap, and `MIRROR_INDEX` is an involution, so a single
    /// application converts between them.
    pub fn aux_opponent_policy(&self) -> Option<Vec<f32>> {
        if self.nodes.is_empty() {
            return None;
        }
        if self.nodes[0].children.is_empty() {
            return None;
        }
        // Use the child that extract_result actually selected (which may have
        // been temperature-sampled). Falls back to LCB/visit-max if no prior
        // search recorded a selection (e.g. direct aux_opponent_policy call
        // without going through extract_result), or if the stored index is
        // stale (not a member of the current root's children).
        let root_children = self.nodes[0].children.clone();
        let stored_idx_valid = self
            .last_selected_child_idx
            .is_some_and(|idx| root_children.contains(&idx));
        let best_child_idx = if let Some(idx) = self.last_selected_child_idx.filter(|_| stored_idx_valid) {
            idx
        } else if self.config.use_lcb {
            self.lcb_best_child_idx(&root_children).or_else(|| {
                root_children
                    .iter()
                    .max_by_key(|&&i| self.nodes[i].visit_count)
                    .copied()
            })?
        } else {
            *root_children
                .iter()
                .max_by_key(|&&i| self.nodes[i].visit_count)?
        };
        let best_child = &self.nodes[best_child_idx];
        if !best_child.is_expanded || best_child.visit_count == 0 {
            return None;
        }
        let num_indices = serialization::num_move_indices();
        let mut out = vec![0.0f32; num_indices];
        let mut total: u32 = 0;
        for &gi in &best_child.children {
            total += self.nodes[gi].visit_count;
        }
        if total == 0 {
            return Some(out);
        }
        let inv = 1.0 / total as f32;
        for &gi in &best_child.children {
            let gc = &self.nodes[gi];
            if let Some(opp_stm_idx) = gc.action_index
                && opp_stm_idx < num_indices
            {
                // Grandchildren were indexed in the opponent's STM frame;
                // remap to the root's frame so main and aux targets share
                // a coordinate system.
                let root_stm_idx = serialization::mirror_move_index(opp_stm_idx);
                out[root_stm_idx] = gc.visit_count as f32 * inv;
            }
        }
        Some(out)
    }

    /// LCB score for a root child at arena index `ci`, from the *parent's*
    /// perspective: `-Q(child) − 1.96·sqrt(Var(child)/N(child))`. Children
    /// with fewer than 2 visits use raw Q (no variance term). Returns `None`
    /// for unvisited children.
    fn lcb_score(&self, ci: usize) -> Option<f64> {
        let c = &self.nodes[ci];
        if c.visit_count == 0 {
            return None;
        }
        let draw_utility = self.config.draw_utility as f64;
        let q = -c.q_value_contempt(draw_utility);
        if c.visit_count < 2 {
            return Some(q);
        }
        let var = c.variance().max(0.0);
        Some(q - LCB_Z_95 * (var / c.visit_count as f64).sqrt())
    }

    /// Pick the root-child arena index with the highest LCB score. Returns
    /// `None` if no child has any visits yet.
    fn lcb_best_child_idx(&self, root_children: &[usize]) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for &ci in root_children {
            let Some(score) = self.lcb_score(ci) else {
                continue;
            };
            match best {
                None => best = Some((ci, score)),
                Some((_, bs)) if score > bs => best = Some((ci, score)),
                _ => {}
            }
        }
        best.map(|(i, _)| i)
    }

    /// Select the best root child using LCB: `Q(a) − 1.96·sqrt(Var(a)/N(a))`.
    /// Children with fewer than 2 visits fall back to raw Q.
    pub fn lcb_best_move(&self) -> Option<Move> {
        if self.nodes.is_empty() || self.nodes[0].children.is_empty() {
            return None;
        }
        let root_children = self.nodes[0].children.clone();
        let best_idx = self.lcb_best_child_idx(&root_children)?;
        self.nodes[best_idx].action
    }

    /// Record a root-side win-probability observation for the resignation
    /// tracker. Must be called with the STM's point-of-view `p_win` for each
    /// actual move played. Returns `true` when the rolling window has filled
    /// with consecutive below-threshold values and resignation is warranted.
    pub fn record_and_check_resign(&mut self, stm: Color, p_win: f32) -> bool {
        if !self.config.resign.enabled {
            return false;
        }
        let side = stm.index();
        let cfg = self.config.resign.clone();
        let hist = &mut self.resign_history[side];
        hist.push_back(p_win);
        while hist.len() > cfg.k {
            hist.pop_front();
        }
        hist.len() == cfg.k && hist.iter().all(|&p| p < cfg.v_resign)
    }

    /// Reset the resignation tracker (e.g. at the start of a new game).
    pub fn reset_resign_history(&mut self) {
        self.resign_history[0].clear();
        self.resign_history[1].clear();
    }
}

/// Outcome of a `run_pcr` call.
#[derive(Clone, Debug)]
pub struct PcrOutcome {
    /// The move chosen to actually play.
    pub best_move: Move,
    /// Scalar `W - L` root value, kept for back-compat with callers that
    /// only want one number.
    pub value: f32,
    /// Root WDL distribution `[W, D, L]` from STM perspective. The worker
    /// uses `wdl[0]` as the resignation signal `P(W)`.
    pub wdl: [f32; 3],
    /// Root visit count after the search (= simulations whose backups
    /// reached the root). Equals the requested sim count absent early
    /// terminal hits. This is what the trainer stores as `root_n`.
    pub nodes_searched: u32,
    /// True iff this call ran a full (high-sim, noisy) search; only full
    /// searches should be recorded as training positions.
    pub was_full_search: bool,
    /// PTP-pruned visit-count policy target. `None` for fast searches.
    pub policy_target: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Dirichlet sampling
// ---------------------------------------------------------------------------

/// Apply shaped Dirichlet noise to the priors in `children_info`.
///
/// Half the noise mass is drawn from `Dir(alpha/2)` across all legal moves
/// (a broad, almost-uniform blind-spot distribution); the other half is
/// concentrated uniformly on the top-`K` children ranked by prior probability.
/// This is the notes/02 §7 "shaped Dirichlet" recipe.
fn apply_shaped_dirichlet_noise<R: Rng + ?Sized>(
    children_info: &mut [(Move, usize, f32)],
    epsilon: f32,
    alpha: f64,
    top_k: usize,
    rng: &mut R,
) {
    let n = children_info.len();
    if n == 0 {
        return;
    }

    let broad = sample_dirichlet(rng, n, alpha * 0.5);

    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.sort_by(|&a, &b| {
        children_info[b]
            .2
            .partial_cmp(&children_info[a].2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = top_k.min(n);
    let mut peaked = vec![0.0f64; n];
    if k > 0 {
        let share = 1.0 / k as f64;
        for &i in idxs.iter().take(k) {
            peaked[i] = share;
        }
    }

    for i in 0..n {
        let noise = 0.5 * broad[i] + 0.5 * peaked[i];
        children_info[i].2 = (1.0 - epsilon) * children_info[i].2 + epsilon * noise as f32;
    }
}

/// Sample from a symmetric Dirichlet distribution Dir(alpha, ..., alpha)
/// with `n` components.
///
/// Uses Gamma sampling via the Ahrens-Dieter method (suitable for alpha < 1)
/// combined with Marsaglia-Tsang for alpha >= 1.
fn sample_dirichlet<R: Rng + ?Sized>(rng: &mut R, n: usize, alpha: f64) -> Vec<f64> {
    let mut samples: Vec<f64> = (0..n).map(|_| sample_gamma(rng, alpha)).collect();
    let sum: f64 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        let uniform = 1.0 / n as f64;
        samples.fill(uniform);
    }
    samples
}

/// Sample from Gamma(alpha, 1) distribution.
///
/// For alpha < 1, uses the transformation: if X ~ Gamma(alpha+1, 1) then
/// X * U^(1/alpha) ~ Gamma(alpha, 1), where U ~ Uniform(0,1).
/// For alpha >= 1, uses Marsaglia-Tsang method.
fn sample_gamma<R: Rng + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
    if alpha < 1.0 {
        // Boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        let u: f64 = rng.random();
        sample_gamma_gt1(rng, alpha + 1.0) * u.powf(1.0 / alpha)
    } else {
        sample_gamma_gt1(rng, alpha)
    }
}

/// Marsaglia-Tsang method for Gamma(alpha, 1) where alpha >= 1.
fn sample_gamma_gt1<R: Rng + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Generate standard normal via Box-Muller.
        let (u1, u2): (f64, f64) = (rng.random(), rng.random());
        let x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u: f64 = rng.random();

        // Squeeze test.
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build an evaluator that returns uniform policy and zero value.
    fn random_evaluator() -> Box<dyn Evaluator> {
        Box::new(HeuristicEvaluator)
    }

    #[test]
    fn heuristic_evaluator_returns_valid_policy() {
        let evaluator = HeuristicEvaluator;
        let state = GameState::new();
        let (policy, wdl) = evaluator.evaluate(&state);

        // Starting position is symmetric — scalar W−L should be near zero
        // and the WDL distribution should sum to 1.
        let value = wdl[0] - wdl[2];
        assert!(
            value.abs() < 1e-4,
            "value should be ~0.0 for starting pos, got {}",
            value
        );
        let wdl_sum = wdl[0] + wdl[1] + wdl[2];
        assert!(
            (wdl_sum - 1.0).abs() < 1e-4,
            "wdl should sum to 1, got {wdl_sum}"
        );

        // Policy should sum to ~1 over legal moves.
        let sum: f32 = policy.iter().sum();
        let legal_count = state.legal_moves().len();
        if legal_count > 0 {
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "policy should sum to 1, got {}",
                sum
            );

            // All legal moves should have nonzero probability.
            let nonzero: Vec<f32> = policy.iter().copied().filter(|&p| p > 0.0).collect();
            assert_eq!(nonzero.len(), legal_count);
        }
    }

    #[test]
    fn mcts_completes_without_panic() {
        let mut search = MctsSearch::new(random_evaluator());
        let state = GameState::new();
        let result = search.search(&state, 50);

        // Should return a valid move.
        let legal = state.legal_moves();
        assert!(
            legal
                .iter()
                .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
            "best move must be a legal move",
        );
    }

    #[test]
    fn mcts_returns_valid_move() {
        let mut search = MctsSearch::new(random_evaluator());
        let state = GameState::new();
        let result = search.search(&state, 100);

        let legal = state.legal_moves();
        assert!(
            legal
                .iter()
                .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
            "search must return a legal move",
        );

        // Policy should be a valid distribution.
        let sum: f32 = result.policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "policy should sum to 1, got {}",
            sum,
        );

        assert!(result.nodes_searched > 0, "must have searched some nodes");
    }

    #[test]
    fn mcts_with_temperature_runs() {
        let mut search = MctsSearch::new(random_evaluator());
        let state = GameState::new();

        // Temperature = 1 should work (stochastic).
        let result = search.search_with_temperature(&state, 50, 1.0);
        let legal = state.legal_moves();
        assert!(
            legal
                .iter()
                .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
        );
    }

    #[test]
    fn mcts_with_dirichlet_noise_runs() {
        let mut search = MctsSearch::new(random_evaluator());
        search.set_dirichlet(Some(DirichletConfig::default()));
        let state = GameState::new();
        let result = search.search(&state, 50);

        let legal = state.legal_moves();
        assert!(
            legal
                .iter()
                .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
        );
    }

    #[test]
    fn mcts_more_simulations_gives_more_nodes() {
        let state = GameState::new();

        let mut search = MctsSearch::new(random_evaluator());
        let r1 = search.search(&state, 10);

        let mut search = MctsSearch::new(random_evaluator());
        let r2 = search.search(&state, 100);

        assert!(
            r2.nodes_searched > r1.nodes_searched,
            "more simulations should produce more nodes: {} vs {}",
            r2.nodes_searched,
            r1.nodes_searched,
        );
    }

    #[test]
    fn mcts_reset_clears_tree() {
        let mut search = MctsSearch::new(random_evaluator());
        let state = GameState::new();
        let _ = search.search(&state, 50);
        assert!(!search.nodes.is_empty());
        search.reset();
        assert!(search.nodes.is_empty());
    }

    // ---------------------------------------------------------------
    // Batched search tests
    // ---------------------------------------------------------------

    /// Evaluator that tracks how many times evaluate vs evaluate_batch is called.
    struct CountingEvaluator {
        single_calls: std::sync::atomic::AtomicU32,
        batch_calls: std::sync::atomic::AtomicU32,
    }

    impl CountingEvaluator {
        fn new() -> Self {
            Self {
                single_calls: std::sync::atomic::AtomicU32::new(0),
                batch_calls: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

    impl Evaluator for CountingEvaluator {
        fn evaluate(&self, state: &GameState) -> (Vec<f32>, [f32; 3]) {
            self.single_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            HeuristicEvaluator.evaluate(state)
        }

        fn evaluate_batch(&self, states: &[&GameState]) -> Vec<(Vec<f32>, [f32; 3])> {
            self.batch_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            states
                .iter()
                .map(|s| HeuristicEvaluator.evaluate(s))
                .collect()
        }
    }

    /// Wrapper to share a CountingEvaluator via Arc while implementing Evaluator.
    struct ArcEval(std::sync::Arc<CountingEvaluator>);
    impl Evaluator for ArcEval {
        fn evaluate(&self, state: &GameState) -> (Vec<f32>, [f32; 3]) {
            self.0.evaluate(state)
        }
        fn evaluate_batch(&self, states: &[&GameState]) -> Vec<(Vec<f32>, [f32; 3])> {
            self.0.evaluate_batch(states)
        }
    }

    #[test]
    fn batched_search_returns_valid_move() {
        let mut search = MctsSearch::new(random_evaluator());
        search.set_batch_size(8);
        let state = GameState::new();
        let result = search.search(&state, 100);

        let legal = state.legal_moves();
        assert!(
            legal
                .iter()
                .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
            "batched search must return a legal move",
        );

        let sum: f32 = result.policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "batched policy should sum to 1, got {}",
            sum,
        );

        assert!(result.nodes_searched > 0);
    }

    #[test]
    fn batched_search_uses_evaluate_batch() {
        let evaluator = std::sync::Arc::new(CountingEvaluator::new());
        let eval_clone = evaluator.clone();
        let mut search = MctsSearch::new(Box::new(ArcEval(eval_clone)));
        search.set_batch_size(4);
        let state = GameState::new();
        let _ = search.search(&state, 40);

        let single = evaluator
            .single_calls
            .load(std::sync::atomic::Ordering::Relaxed);
        let batch = evaluator
            .batch_calls
            .load(std::sync::atomic::Ordering::Relaxed);

        // Batched path should call evaluate_batch, not evaluate.
        assert_eq!(
            single, 0,
            "single evaluate should not be called in batched mode"
        );
        assert!(batch > 0, "evaluate_batch should have been called");
    }

    #[test]
    fn batched_search_with_temperature() {
        let mut search = MctsSearch::new(random_evaluator());
        search.set_batch_size(4);
        let state = GameState::new();

        let result = search.search_with_temperature(&state, 50, 1.0);
        let legal = state.legal_moves();
        assert!(
            legal
                .iter()
                .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
        );
    }

    #[test]
    fn batched_search_with_dirichlet() {
        let mut search = MctsSearch::new(random_evaluator());
        search.set_batch_size(4);
        search.set_dirichlet(Some(DirichletConfig::default()));
        let state = GameState::new();

        let result = search.search(&state, 50);
        let legal = state.legal_moves();
        assert!(
            legal
                .iter()
                .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
        );
    }

    #[test]
    fn batched_more_sims_gives_more_nodes() {
        let state = GameState::new();

        let mut s1 = MctsSearch::new(random_evaluator());
        s1.set_batch_size(4);
        let r1 = s1.search(&state, 10);

        let mut s2 = MctsSearch::new(random_evaluator());
        s2.set_batch_size(4);
        let r2 = s2.search(&state, 100);

        assert!(
            r2.nodes_searched > r1.nodes_searched,
            "batched: more sims should produce more nodes: {} vs {}",
            r2.nodes_searched,
            r1.nodes_searched,
        );
    }

    #[test]
    fn batch_size_1_uses_sequential_path() {
        let evaluator = std::sync::Arc::new(CountingEvaluator::new());
        let eval_clone = evaluator.clone();
        let mut search = MctsSearch::new(Box::new(ArcEval(eval_clone)));
        search.set_batch_size(1);
        let state = GameState::new();
        let _ = search.search(&state, 20);

        let single = evaluator
            .single_calls
            .load(std::sync::atomic::Ordering::Relaxed);
        let batch = evaluator
            .batch_calls
            .load(std::sync::atomic::Ordering::Relaxed);

        // Sequential path should call evaluate, not evaluate_batch.
        assert!(single > 0, "sequential path should call evaluate");
        assert_eq!(batch, 0, "sequential path should not call evaluate_batch");
    }

    /// Verify that sequential (batch=1) and batched (batch=8) produce
    /// structurally similar results: both return legal moves, valid policies,
    /// and comparable node counts.
    #[test]
    fn sequential_vs_batched_structural_equivalence() {
        let state = GameState::new();
        let sims = 200;

        let mut seq = MctsSearch::new(random_evaluator());
        seq.set_batch_size(1);
        let r_seq = seq.search(&state, sims);

        let mut bat = MctsSearch::new(random_evaluator());
        bat.set_batch_size(8);
        let r_bat = bat.search(&state, sims);

        let legal = state.legal_moves();

        // Both return legal moves.
        assert!(
            legal
                .iter()
                .any(|m| m.from == r_seq.best_move.from && m.to == r_seq.best_move.to)
        );
        assert!(
            legal
                .iter()
                .any(|m| m.from == r_bat.best_move.from && m.to == r_bat.best_move.to)
        );

        // Both have valid policies.
        let sum_seq: f32 = r_seq.policy.iter().sum();
        let sum_bat: f32 = r_bat.policy.iter().sum();
        assert!((sum_seq - 1.0).abs() < 1e-3);
        assert!((sum_bat - 1.0).abs() < 1e-3);

        // Node counts should be in the same ballpark (batched may differ
        // slightly due to TT hits in batch reducing actual evaluations,
        // but both should have expanded a comparable number of nodes).
        assert!(
            r_bat.nodes_searched > sims / 2,
            "batched should still build a substantial tree: got {}",
            r_bat.nodes_searched,
        );
    }

    /// Various batch sizes should all produce valid results.
    #[test]
    fn various_batch_sizes_all_valid() {
        let state = GameState::new();
        for bs in [2, 3, 4, 7, 16, 32] {
            let mut search = MctsSearch::new(random_evaluator());
            search.set_batch_size(bs);
            let result = search.search(&state, 64);

            let legal = state.legal_moves();
            assert!(
                legal
                    .iter()
                    .any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
                "batch_size={bs}: must return a legal move",
            );

            let sum: f32 = result.policy.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "batch_size={bs}: policy should sum to 1, got {sum}",
            );
        }
    }

    /// Batched search shouldn't corrupt game state — verify the root
    /// visit count matches num_simulations.
    #[test]
    fn batched_root_visit_count_matches_simulations() {
        let state = GameState::new();
        let sims = 100u32;

        let mut search = MctsSearch::new(random_evaluator());
        search.set_batch_size(8);
        let _ = search.search(&state, sims);

        // Root node should have exactly `sims` visits.
        assert_eq!(
            search.nodes[0].visit_count, sims,
            "root visit count should equal num_simulations",
        );
    }

    /// Virtual loss should be fully removed after search completes.
    /// Verify by checking that Q-values are reasonable (not poisoned by
    /// leaked virtual loss, which would push them far below -1.0).
    #[test]
    fn virtual_loss_fully_removed_after_search() {
        let state = GameState::new();
        let sims = 80u32;

        let mut search = MctsSearch::new(random_evaluator());
        search.set_batch_size(4);
        let _ = search.search(&state, sims);

        // Root visit count must equal num_simulations.
        assert_eq!(search.nodes[0].visit_count, sims);

        // Every node's Q-value should be in [-1, 1] (values are tanh-bounded).
        // If virtual loss leaked, Q would be far below -1.
        for (i, node) in search.nodes.iter().enumerate() {
            if node.visit_count > 0 {
                let q = node.q_value();
                assert!(
                    q >= -1.0 - 1e-6 && q <= 1.0 + 1e-6,
                    "node {i}: Q-value {q} out of [-1, 1] range — possible virtual loss leak",
                );
            }
        }

        // Children visits + initial root-only visits should equal root visits.
        let root = &search.nodes[0];
        let children_visits: u32 = root
            .children
            .iter()
            .map(|&i| search.nodes[i].visit_count)
            .sum();
        // The gap is from the first batch where root wasn't yet expanded —
        // up to batch_size leaves may all land on the unexpanded root.
        let gap = sims - children_visits;
        let bs = search.config().batch_size as u32;
        assert!(
            gap <= bs,
            "visit gap ({gap}) should be at most batch_size ({bs})",
        );
    }

    /// Virtual loss must *discourage* re-selection of a node within the same
    /// batch. This test catches sign errors: if VL is subtracted instead of
    /// added to `value_sum`, the negated q-value in PUCT actually *increases*,
    /// causing the batched search to funnel all visits into one child.
    #[test]
    fn virtual_loss_discourages_reselection() {
        // Set up a position and run a batched search.
        let state = GameState::new();
        let sims = 200u32;

        let mut search = MctsSearch::new(random_evaluator());
        search.set_batch_size(16);
        let _ = search.search(&state, sims);

        let root = &search.nodes[0];
        let child_visits: Vec<u32> = root
            .children
            .iter()
            .map(|&i| search.nodes[i].visit_count)
            .collect();
        let max_visits = *child_visits.iter().max().unwrap();
        let num_children = child_visits.len() as u32;
        let total: u32 = child_visits.iter().sum();

        // With correct virtual loss, visits should be spread across children.
        // No single child should hog more than 50% of total visits.
        // (With inverted VL, the first child gets nearly ALL visits.)
        assert!(
            max_visits < total / 2,
            "One child has {max_visits}/{total} visits — virtual loss is not \
             discouraging re-selection (expected spread across {num_children} children)",
        );

        // At least 60% of children should have been visited at least once.
        let visited = child_visits.iter().filter(|&&v| v > 0).count() as u32;
        assert!(
            visited * 5 >= num_children * 3,
            "Only {visited}/{num_children} children visited — virtual loss \
             may be funnelling visits into too few children",
        );
    }

    // ---------------------------------------------------------------
    // v2 rebuild: config, dynamic PUCT, shaped Dirichlet, PCR, PTP,
    // LCB, temperature decay, 2-fold draw, TT key
    // ---------------------------------------------------------------

    #[test]
    fn dynamic_puct_formula_known_points() {
        let c2 = 19652.0f32;
        let v0 = dynamic_c_puct(2.5, c2, 0);
        let expected0 = 2.5 + ((1.0 + 0.0 + c2) / c2).ln();
        assert!(
            (v0 - expected0).abs() < 1e-5,
            "N=0: got {v0}, want {expected0}"
        );
        assert!(v0 > 2.5 && v0 < 2.5001);

        let v1k = dynamic_c_puct(2.5, c2, 1_000);
        let expected1k = 2.5 + ((1.0 + 1000.0 + c2) / c2).ln();
        assert!((v1k - expected1k).abs() < 1e-5);

        let v_large = dynamic_c_puct(3.5, c2, 200_000);
        assert!(v_large > 3.5 + 1.5 && v_large < 3.5 + 3.0);
    }

    #[test]
    fn search_config_defaults_match_spec() {
        let t = SearchConfig::training();
        assert_eq!(t.c_puct, 2.5);
        assert_eq!(t.c_puct_root, 3.5);
        assert_eq!(t.c2, 19652.0);
        assert_eq!(t.fpu_reduction, 0.0);
        assert_eq!(t.virtual_loss, 1.0);
        assert_eq!(t.forced_playout_k, 2.0);
        assert!(t.two_fold_as_draw);
        assert!(t.dirichlet.is_some());
        assert!(!t.use_lcb);
        assert_eq!(t.draw_utility, 0.0);

        let e = SearchConfig::eval();
        assert_eq!(e.fpu_reduction, 0.2);
        assert!(e.dirichlet.is_none());
        assert_eq!(e.forced_playout_k, 0.0);
        assert!(e.use_lcb);
        assert_eq!(e.pcr.p_full, 1.0);
        assert_eq!(e.draw_utility, 0.0);
    }

    #[test]
    fn set_config_without_seed_clears_internal_rng() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.set_rng_seed(123);
        assert!(search.rng.is_some(), "set_rng_seed should initialize rng");

        // Default configs carry `rng_seed = None`, which should clear any
        // previously seeded RNG and return to thread-local randomness.
        search.set_config(SearchConfig::training());
        assert!(
            search.rng.is_none(),
            "set_config with rng_seed=None should clear internal rng",
        );
    }

    #[test]
    fn temperature_schedule_decays_and_clamps() {
        let ts = TemperatureSchedule::default();
        assert!((ts.temperature_at(0) - 1.0).abs() < 1e-5);
        let t10 = ts.temperature_at(10);
        assert!(t10 > 0.1 && t10 < 1.0);
        let t100 = ts.temperature_at(50);
        assert!(t100 >= ts.tau_min - 1e-6);
        assert_eq!(ts.temperature_at(60), 0.0);
        assert_eq!(ts.temperature_at(999), 0.0);
    }

    #[test]
    fn shaped_dirichlet_returns_valid_distribution() {
        let legal = GameState::new().legal_moves();
        assert!(!legal.is_empty());
        let uniform_prior = 1.0 / legal.len() as f32;
        let mut ci: Vec<(Move, usize, f32)> = legal
            .iter()
            .enumerate()
            .map(|(i, m)| (*m, i, uniform_prior))
            .collect();
        let mut trng = rand::rng();
        apply_shaped_dirichlet_noise(&mut ci, 0.25, 0.25, 10, &mut trng);
        let sum: f32 = ci.iter().map(|e| e.2).sum();
        assert!((sum - 1.0).abs() < 1e-3, "shaped dirichlet sum={sum}");
        for (_, _, p) in &ci {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }

    #[test]
    fn forced_playout_count_formula() {
        let k = 2.0f64;
        let p = 0.25f64;
        let n = 400u32;
        let expected = (k * p * (n as f64).sqrt()).ceil() as u32;
        assert_eq!(expected, 10);
        let p2 = 0.01f64;
        let expected2 = (k * p2 * (n as f64).sqrt()).ceil() as u32;
        assert_eq!(expected2, 1);
    }

    #[test]
    fn pcr_branches_on_coin() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let state = GameState::new();
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.config_mut().pcr = PcrConfig {
            p_full: 0.5,
            n_full: 40,
            n_fast: 20,
        };
        search.config_mut().dirichlet = None;

        let mut rng_always_full = StdRng::seed_from_u64(0);
        let mut full_count = 0;
        let mut fast_count = 0;
        for _ in 0..20 {
            let outcome = search.run_pcr(&state, 0, &mut rng_always_full);
            if outcome.was_full_search {
                full_count += 1;
                assert!(outcome.policy_target.is_some());
            } else {
                fast_count += 1;
                assert!(outcome.policy_target.is_none());
            }
        }
        assert!(
            full_count > 0 && fast_count > 0,
            "both branches should fire"
        );
    }

    /// Regression test: fast PCR moves must use greedy selection (temp=0),
    /// not the temperature schedule. With greedy selection, the same position
    /// should always produce the same move regardless of RNG seed.
    #[test]
    fn fast_pcr_uses_greedy_selection() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let state = GameState::new();

        // Force PCR to always take the fast branch (p_full = 0).
        let mut search_a = MctsSearch::new(Box::new(HeuristicEvaluator));
        search_a.config_mut().pcr = PcrConfig {
            p_full: 0.0,
            n_full: 100,
            n_fast: 50,
        };
        search_a.config_mut().dirichlet = None;
        // High temperature schedule that would cause variance if used.
        search_a.config_mut().temperature = TemperatureSchedule {
            tau_max: 1.5,
            tau_min: 1.0,
            halflife: 10.0,
            hard_greedy_ply: 100,
        };

        let mut search_b = MctsSearch::new(Box::new(HeuristicEvaluator));
        search_b.config_mut().pcr = search_a.config().pcr.clone();
        search_b.config_mut().dirichlet = None;
        search_b.config_mut().temperature = search_a.config().temperature.clone();

        // Different RNG seeds — if temperature were applied, moves could differ.
        let mut rng_a = StdRng::seed_from_u64(111);
        let mut rng_b = StdRng::seed_from_u64(999);

        let outcome_a = search_a.run_pcr(&state, 0, &mut rng_a);
        let outcome_b = search_b.run_pcr(&state, 0, &mut rng_b);

        assert!(!outcome_a.was_full_search);
        assert!(!outcome_b.was_full_search);
        // Compare moves field by field since Move doesn't derive PartialEq.
        assert_eq!(
            outcome_a.best_move.from, outcome_b.best_move.from,
            "fast PCR should be greedy (deterministic), not temperature-sampled"
        );
        assert_eq!(
            outcome_a.best_move.to, outcome_b.best_move.to,
            "fast PCR should be greedy (deterministic), not temperature-sampled"
        );
        assert_eq!(
            outcome_a.best_move.promotion, outcome_b.best_move.promotion,
            "fast PCR should be greedy (deterministic), not temperature-sampled"
        );
    }

    /// Regression test: aux_opponent_policy must report grandchildren of the
    /// child actually selected by extract_result (which may be temperature-
    /// sampled), not an independent visit-max recomputation.
    #[test]
    fn aux_opponent_policy_follows_temperature_sampled_child() {
        let state = GameState::new();
        let legal = state.legal_moves();
        assert!(legal.len() >= 2);

        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.config_mut().use_lcb = false;

        // Build a synthetic tree: root with two children, each with disjoint
        // grandchildren. We'll manually set last_selected_child_idx to the
        // lower-visit child (simulating temperature sampling) and verify
        // aux_opponent_policy reports that child's grandchildren.
        let mut root = MctsNode::new(None, None, 1.0);
        root.is_expanded = true;
        search.nodes.push(root);

        // c_a: high visits (visit-max would pick this)
        let mut c_a = MctsNode::new(Some(legal[0]), Some(0), 0.5);
        c_a.visit_count = 200;
        c_a.is_expanded = true;
        search.nodes.push(c_a);
        search.nodes[0].children.push(1);

        // c_b: lower visits (temperature sampling picked this)
        let mut c_b = MctsNode::new(Some(legal[1]), Some(1), 0.5);
        c_b.visit_count = 50;
        c_b.is_expanded = true;
        search.nodes.push(c_b);
        search.nodes[0].children.push(2);

        // Grandchildren with disjoint action indices so we can tell them apart.
        let idx_gc_a = 100usize;
        let idx_gc_b = 200usize;
        let root_idx_gc_a = serialization::mirror_move_index(idx_gc_a);
        let root_idx_gc_b = serialization::mirror_move_index(idx_gc_b);
        assert_ne!(root_idx_gc_a, root_idx_gc_b);

        let mut gc_a = MctsNode::new(Some(legal[0]), Some(idx_gc_a), 1.0);
        gc_a.visit_count = 80;
        search.nodes.push(gc_a);
        search.nodes[1].children.push(3); // gc_a under c_a

        let mut gc_b = MctsNode::new(Some(legal[1]), Some(idx_gc_b), 1.0);
        gc_b.visit_count = 30;
        search.nodes.push(gc_b);
        search.nodes[2].children.push(4); // gc_b under c_b

        // Simulate temperature sampling having picked c_b (index 2).
        search.last_selected_child_idx = Some(2);

        let aux = search.aux_opponent_policy().expect("should have aux");

        // Aux should report c_b's grandchildren, not c_a's.
        assert!(
            aux[root_idx_gc_b] > 0.0,
            "aux should report grandchildren of the temperature-sampled child (c_b)"
        );
        assert!(
            aux[root_idx_gc_a] == 0.0,
            "aux should NOT report grandchildren of the visit-max child (c_a)"
        );
    }

    #[test]
    fn seeded_temperature_search_is_reproducible() {
        let state = GameState::new();
        let mut a = MctsSearch::new(Box::new(HeuristicEvaluator));
        let mut b = MctsSearch::new(Box::new(HeuristicEvaluator));
        a.set_rng_seed(7);
        b.set_rng_seed(7);

        let ra = a.search_with_temperature(&state, 64, 1.0);
        let rb = b.search_with_temperature(&state, 64, 1.0);

        assert_eq!(ra.best_move.from, rb.best_move.from);
        assert_eq!(ra.best_move.to, rb.best_move.to);
        assert_eq!(ra.best_move.promotion, rb.best_move.promotion);
        assert_eq!(
            ra.policy, rb.policy,
            "same seed should produce identical visit policy"
        );
    }

    #[test]
    fn ptp_hand_computed_example() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.config_mut().forced_playout_k = 2.0;

        let mut root = MctsNode::new(None, None, 1.0);
        root.visit_count = 100;
        root.is_expanded = true;
        search.nodes.push(root);

        let mut make_child = |action_idx: usize, visits: u32, prior: f32| {
            let mut c = MctsNode::new(None, Some(action_idx), prior);
            c.visit_count = visits;
            search.nodes.push(c);
            let ci = search.nodes.len() - 1;
            search.nodes[0].children.push(ci);
        };
        make_child(0, 70, 0.6);
        make_child(1, 20, 0.3);
        make_child(2, 10, 0.1);

        let policy = search.policy_target_pruned();
        let p0 = policy[0];
        let p1 = policy[1];
        let p2 = policy[2];

        let sqrt_n = 10.0f64;
        let n_forced_1 = (2.0 * 0.3f32 as f64 * sqrt_n).ceil() as i64;
        let n_forced_2 = (2.0 * 0.1f32 as f64 * sqrt_n).ceil() as i64;
        let adj1 = (20 - n_forced_1).max(0);
        let adj2 = (10 - n_forced_2).max(0);

        let total = 70.0 + adj1 as f64 + adj2 as f64;
        let exp0 = 70.0 / total;
        let exp1 = adj1 as f64 / total;
        let exp2 = adj2 as f64 / total;
        assert!(
            (p0 as f64 - exp0).abs() < 1e-5,
            "p0={p0} exp={exp0} p1={p1} exp1={exp1} p2={p2} exp2={exp2}"
        );
        assert!((p1 as f64 - exp1).abs() < 1e-5);
        assert!((p2 as f64 - exp2).abs() < 1e-5);
        assert!(p0 > p1 && p0 > p2, "argmax must be preserved");
    }

    #[test]
    fn lcb_picks_lower_variance_child() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        let legal = GameState::new().legal_moves();
        assert!(legal.len() >= 2);

        let mut root = MctsNode::new(None, None, 1.0);
        root.is_expanded = true;
        search.nodes.push(root);

        let mut c_a = MctsNode::new(Some(legal[0]), Some(0), 0.5);
        c_a.visit_count = 100;
        // Equivalent to the legacy scalar `value_sum = -60`: all loss mass,
        // no draws, Q = (0 − 60) / 100 = −0.6.
        c_a.value_sum = [0.0, 0.0, 60.0];
        c_a.m2 = 1.0;
        search.nodes.push(c_a);
        search.nodes[0].children.push(1);

        let mut c_b = MctsNode::new(Some(legal[1]), Some(1), 0.5);
        c_b.visit_count = 100;
        c_b.value_sum = [0.0, 0.0, 62.0];
        c_b.m2 = 0.01;
        search.nodes.push(c_b);
        search.nodes[0].children.push(2);

        let best = search.lcb_best_move().unwrap();
        assert_eq!(best.from, legal[1].from);
        assert_eq!(best.to, legal[1].to);
    }

    /// Regression: `SearchConfig::eval()` sets `use_lcb = true`, but until we
    /// fixed extract_result, greedy selection still went through
    /// `max_by_key(visit_count)`. Drives extract_result end-to-end and
    /// asserts the LCB winner is played even when visit-max would not choose
    /// it — and that the policy target still tracks visit counts.
    #[test]
    fn extract_result_honors_lcb_when_enabled() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.set_config(SearchConfig::eval());
        assert!(search.config().use_lcb);

        let state = GameState::new();
        let legal = state.legal_moves();
        assert!(legal.len() >= 2);

        // Construct a fake expanded root. c_a has more visits but a wider
        // Q distribution; c_b has fewer visits but much tighter variance.
        // Visit-max picks c_a; LCB picks c_b.
        let mut root = MctsNode::new(None, None, 1.0);
        root.is_expanded = true;
        search.nodes.push(root);

        let mut c_a = MctsNode::new(Some(legal[0]), Some(0), 0.5);
        c_a.visit_count = 200;
        // value_sum in WDL form: Q = (0 − 100)/200 = −0.5 → root-side q = 0.5.
        c_a.value_sum = [0.0, 0.0, 100.0];
        c_a.m2 = 50.0; // var ≈ 0.2513
        search.nodes.push(c_a);
        search.nodes[0].children.push(1);

        let mut c_b = MctsNode::new(Some(legal[1]), Some(1), 0.5);
        c_b.visit_count = 100;
        // Q = (0 − 48)/100 = −0.48 → root-side q = 0.48.
        c_b.value_sum = [0.0, 0.0, 48.0];
        c_b.m2 = 0.5; // var ≈ 0.00505
        search.nodes.push(c_b);
        search.nodes[0].children.push(2);

        // Sanity: visit-max would pick c_a (more visits).
        let visit_max_idx = *search.nodes[0]
            .children
            .iter()
            .max_by_key(|&&i| search.nodes[i].visit_count)
            .unwrap();
        assert_eq!(visit_max_idx, 1, "visit-max baseline should be c_a");

        // Greedy extract in eval mode should pick LCB winner (c_b).
        let result = search.extract_result(&state, 0.0);
        assert_eq!(result.best_move.from, legal[1].from);
        assert_eq!(result.best_move.to, legal[1].to);

        // Policy target must still reflect visit counts, not LCB — so c_a
        // (more visits) should get strictly more mass than c_b.
        let policy_a = result.policy[0];
        let policy_b = result.policy[1];
        assert!(
            policy_a > policy_b,
            "policy target should follow visits (a={policy_a}, b={policy_b})"
        );
    }

    /// Training mode (`use_lcb = false`) must keep the visit-max behaviour
    /// unchanged — LCB is strictly an eval-path selection rule.
    #[test]
    fn extract_result_uses_visit_max_when_lcb_disabled() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        assert!(!search.config().use_lcb);

        let state = GameState::new();
        let legal = state.legal_moves();
        assert!(legal.len() >= 2);

        let mut root = MctsNode::new(None, None, 1.0);
        root.is_expanded = true;
        search.nodes.push(root);

        // Same values as the LCB test — LCB would pick c_b, visit-max picks c_a.
        let mut c_a = MctsNode::new(Some(legal[0]), Some(0), 0.5);
        c_a.visit_count = 200;
        c_a.value_sum = [0.0, 0.0, 100.0];
        c_a.m2 = 50.0;
        search.nodes.push(c_a);
        search.nodes[0].children.push(1);

        let mut c_b = MctsNode::new(Some(legal[1]), Some(1), 0.5);
        c_b.visit_count = 100;
        c_b.value_sum = [0.0, 0.0, 48.0];
        c_b.m2 = 0.5;
        search.nodes.push(c_b);
        search.nodes[0].children.push(2);

        let result = search.extract_result(&state, 0.0);
        assert_eq!(result.best_move.from, legal[0].from);
        assert_eq!(result.best_move.to, legal[0].to);
    }

    #[test]
    fn tt_key_discriminates_by_repetition() {
        let state = GameState::new();
        let k0 = TtKey::from_state(&state);
        let k0_again = TtKey::from_state(&state);
        assert_eq!(k0, k0_again);
        let k_rep = TtKey {
            repetition: 1,
            ..k0
        };
        assert_ne!(k0, k_rep);
        let mut map: HashMap<TtKey, i32> = HashMap::new();
        map.insert(k0, 1);
        map.insert(k_rep, 2);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn two_fold_as_draw_runs_without_hang() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.config_mut().two_fold_as_draw = true;
        search.config_mut().batch_size = 1;
        let state = GameState::new();
        let result = search.search(&state, 100);
        assert!(result.nodes_searched > 0);
    }

    #[test]
    fn resignation_triggers_after_k_low_values() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.config_mut().resign = ResignConfig {
            enabled: true,
            v_resign: 0.05,
            k: 5,
        };
        for _ in 0..4 {
            assert!(!search.record_and_check_resign(Color::White, 0.01));
        }
        assert!(search.record_and_check_resign(Color::White, 0.01));
    }

    #[test]
    fn resignation_resets_on_high_value() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.config_mut().resign = ResignConfig {
            enabled: true,
            v_resign: 0.05,
            k: 5,
        };
        for _ in 0..4 {
            assert!(!search.record_and_check_resign(Color::White, 0.01));
        }
        assert!(!search.record_and_check_resign(Color::White, 0.5));
        assert!(!search.record_and_check_resign(Color::White, 0.01));
    }

    #[test]
    fn rng_seed_yields_deterministic_visits() {
        // Two searches with the same seed on the same position must produce
        // identical visit counts across all root children.
        let state = GameState::new();

        let mut run_one = || {
            let mut s = MctsSearch::new(Box::new(HeuristicEvaluator));
            s.config_mut().batch_size = 1;
            s.set_rng_seed(42);
            let _ = s.search(&state, 64);
            let root = &s.nodes[0];
            root.children
                .iter()
                .map(|&i| s.nodes[i].visit_count)
                .collect::<Vec<_>>()
        };

        let a = run_one();
        let b = run_one();
        assert_eq!(a, b, "seeded searches must be deterministic");
    }

    #[test]
    fn aux_opponent_policy_sums_to_one() {
        let state = GameState::new();
        let mut s = MctsSearch::new(Box::new(HeuristicEvaluator));
        s.config_mut().batch_size = 1;
        s.set_rng_seed(7);
        let _ = s.search(&state, 128);
        let aux = s.aux_opponent_policy().expect("should have aux");
        let sum: f32 = aux.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "aux sum = {sum}");
    }

    /// Invariant: under eval mode, `aux_opponent_policy` must describe the
    /// opponent replies under the child that `extract_result` actually plays
    /// — not the visit-max child, which can differ under LCB. Regression
    /// guard for the divergence Codex flagged on PR #106.
    #[test]
    fn aux_opponent_policy_matches_lcb_played_child_in_eval_mode() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.set_config(SearchConfig::eval());

        let state = GameState::new();
        let legal = state.legal_moves();
        assert!(legal.len() >= 2);

        // Root with two children. c_a: more visits but high variance.
        // c_b: fewer visits but tight variance. Visit-max → c_a, LCB → c_b.
        let mut root = MctsNode::new(None, None, 1.0);
        root.is_expanded = true;
        search.nodes.push(root);

        let mut c_a = MctsNode::new(Some(legal[0]), Some(0), 0.5);
        c_a.visit_count = 200;
        c_a.value_sum = [0.0, 0.0, 100.0];
        c_a.m2 = 50.0;
        c_a.is_expanded = true;
        search.nodes.push(c_a);
        search.nodes[0].children.push(1);

        let mut c_b = MctsNode::new(Some(legal[1]), Some(1), 0.5);
        c_b.visit_count = 100;
        c_b.value_sum = [0.0, 0.0, 48.0];
        c_b.m2 = 0.5;
        c_b.is_expanded = true;
        search.nodes.push(c_b);
        search.nodes[0].children.push(2);

        // Give c_a and c_b disjoint sets of grandchildren so the aux
        // distribution uniquely identifies which child was selected.
        // Grandchild action_index is stored in the opponent's STM frame
        // (see expand_nodes / stm_policy_index), and aux_opponent_policy
        // remaps it back to the root's frame via `mirror_move_index`.
        // We therefore pick two indices a/b that stay distinct after the
        // remap, and assert aux mass lands at `mirror_move_index(b)`.
        let idx_a = 10usize;
        let idx_b = 20usize;
        let root_idx_a = serialization::mirror_move_index(idx_a);
        let root_idx_b = serialization::mirror_move_index(idx_b);
        assert_ne!(root_idx_a, root_idx_b);

        let mut gc_a = MctsNode::new(Some(legal[0]), Some(idx_a), 1.0);
        gc_a.visit_count = 50;
        search.nodes.push(gc_a);
        search.nodes[1].children.push(3);

        let mut gc_b = MctsNode::new(Some(legal[1]), Some(idx_b), 1.0);
        gc_b.visit_count = 50;
        search.nodes.push(gc_b);
        search.nodes[2].children.push(4);

        // extract_result should pick c_b (LCB winner).
        let result = search.extract_result(&state, 0.0);
        assert_eq!(result.best_move.from, legal[1].from);
        assert_eq!(result.best_move.to, legal[1].to);

        // aux_opponent_policy must expose grandchildren of c_b, so mass
        // belongs at `mirror_move_index(idx_b)` (root STM frame), not at
        // `mirror_move_index(idx_a)`.
        let aux = search
            .aux_opponent_policy()
            .expect("aux should be available");
        assert_eq!(
            aux[root_idx_a], 0.0,
            "aux should NOT describe visit-max (c_a) grandchildren"
        );
        assert!(
            aux[root_idx_b] > 0.0,
            "aux should describe LCB-selected (c_b) grandchildren"
        );
    }

    /// Training mode keeps `aux_opponent_policy` on visit-max — confirms the
    /// fix doesn't accidentally change the self-play auxiliary-head target.
    #[test]
    fn aux_opponent_policy_uses_visit_max_when_lcb_disabled() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        assert!(!search.config().use_lcb);

        let state = GameState::new();
        let legal = state.legal_moves();
        assert!(legal.len() >= 2);

        let mut root = MctsNode::new(None, None, 1.0);
        root.is_expanded = true;
        search.nodes.push(root);

        let mut c_a = MctsNode::new(Some(legal[0]), Some(0), 0.5);
        c_a.visit_count = 200;
        c_a.value_sum = [0.0, 0.0, 100.0];
        c_a.m2 = 50.0;
        c_a.is_expanded = true;
        search.nodes.push(c_a);
        search.nodes[0].children.push(1);

        let mut c_b = MctsNode::new(Some(legal[1]), Some(1), 0.5);
        c_b.visit_count = 100;
        c_b.value_sum = [0.0, 0.0, 48.0];
        c_b.m2 = 0.5;
        c_b.is_expanded = true;
        search.nodes.push(c_b);
        search.nodes[0].children.push(2);

        // Same disjoint-grandchildren setup as the LCB test. Aux remaps
        // the stored opp-STM indices to root-STM via `mirror_move_index`,
        // so we look up mass at the remapped slots.
        let idx_a = 10usize;
        let idx_b = 20usize;
        let root_idx_a = serialization::mirror_move_index(idx_a);
        let root_idx_b = serialization::mirror_move_index(idx_b);
        assert_ne!(root_idx_a, root_idx_b);

        let mut gc_a = MctsNode::new(Some(legal[0]), Some(idx_a), 1.0);
        gc_a.visit_count = 50;
        search.nodes.push(gc_a);
        search.nodes[1].children.push(3);

        let mut gc_b = MctsNode::new(Some(legal[1]), Some(idx_b), 1.0);
        gc_b.visit_count = 50;
        search.nodes.push(gc_b);
        search.nodes[2].children.push(4);

        // Training mode: visit-max → c_a, and aux tracks c_a's grandchildren.
        let _ = search.extract_result(&state, 0.0);
        let aux = search
            .aux_opponent_policy()
            .expect("aux should be available");
        assert!(
            aux[root_idx_a] > 0.0,
            "training mode should expose visit-max (c_a) grandchildren"
        );
        assert_eq!(aux[root_idx_b], 0.0, "c_b grandchildren must not appear");
    }

    #[test]
    fn aux_opponent_policy_is_in_root_stm_frame() {
        // The aux target must live in the ROOT's STM frame, not the
        // opponent's. For every visited grandchild, verify that aux
        // places the *exact* normalized visit mass (visits / total_visits)
        // at the root-STM index for its move. Also assert that slots at
        // indices which correspond to no visited grandchild are zero,
        // catching "wrong-frame mass leaked into a different slot" bugs
        // that a positivity-only check would miss.
        let state = GameState::new();
        let root_stm = state.board.side_to_move;

        let mut s = MctsSearch::new(Box::new(HeuristicEvaluator));
        s.config_mut().batch_size = 1;
        s.set_rng_seed(13);
        let _ = s.search(&state, 256);

        let aux = s.aux_opponent_policy().expect("should have aux");

        let root_children = s.nodes[0].children.clone();
        let best_child_idx = *root_children
            .iter()
            .max_by_key(|&&i| s.nodes[i].visit_count)
            .unwrap();
        let best_child = &s.nodes[best_child_idx];

        let total_visits: u32 = best_child
            .children
            .iter()
            .map(|&gi| s.nodes[gi].visit_count)
            .sum();
        assert!(total_visits > 0, "expected visited grandchildren");

        let mut expected_mass_by_idx: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();
        for &gi in &best_child.children {
            let gc = &s.nodes[gi];
            if gc.visit_count == 0 {
                continue;
            }
            let mv = gc.action.expect("grandchild must have a move");
            let root_idx = serialization::stm_policy_index(&mv, root_stm)
                .expect("grandchild move must be in the table");
            *expected_mass_by_idx.entry(root_idx).or_insert(0.0) +=
                gc.visit_count as f32 / total_visits as f32;
        }

        // Every index we expect must match within fp tolerance.
        for (&idx, &expected) in &expected_mass_by_idx {
            assert!(
                (aux[idx] - expected).abs() < 1e-5,
                "aux[{idx}] = {} but expected {} (root-STM visit mass)",
                aux[idx],
                expected
            );
        }
        // And every aux entry must be zero unless it's one of the expected
        // indices — if a wrong-frame impl parked mass at a different slot,
        // this catches it.
        for (i, &p) in aux.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            assert!(
                expected_mass_by_idx.contains_key(&i),
                "aux[{i}] = {p} but no visited grandchild maps to that root-STM index"
            );
        }
    }

    #[test]
    fn mcts_policy_frames_match_side_to_move() {
        // Run a short search from a black-to-move position and verify
        // EXACT normalized visit mass at every expected STM index for
        // both main and aux policies. This exercises the root-black case
        // which under the old absolute-frame code was indistinguishable
        // from root-white — in STM frame the indices actually differ via
        // MIRROR_INDEX, so a "forgot to remap" regression would surface
        // as mass landing at the wrong slot.
        let mut state = GameState::new();
        let white_move = state.legal_moves()[0];
        state.apply_move(white_move);
        let root_stm = state.board.side_to_move;
        assert_eq!(root_stm, Color::Black);

        let mut s = MctsSearch::new(Box::new(HeuristicEvaluator));
        s.config_mut().batch_size = 1;
        s.set_rng_seed(21);
        let result = s.search_with_temperature(&state, 128, 0.0);

        // ---- Main policy: exact mass check on every root child. ----
        let root_children = s.nodes[0].children.clone();
        let total_root_visits: u32 = root_children
            .iter()
            .map(|&ci| s.nodes[ci].visit_count)
            .sum();
        assert!(total_root_visits > 0);

        // Under temperature = 0, extract_result normalizes raw visit counts
        // directly (see extract_result). Build the expected vector the same
        // way and assert exact match — this both validates the frame AND
        // catches any re-normalization drift.
        let mut expected_main: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();
        for &ci in &root_children {
            let child = &s.nodes[ci];
            if child.visit_count == 0 {
                continue;
            }
            let mv = child.action.expect("child must have a move");
            let idx =
                serialization::stm_policy_index(&mv, root_stm).expect("move must be in the table");
            *expected_main.entry(idx).or_insert(0.0) +=
                child.visit_count as f32 / total_root_visits as f32;
        }
        for (&idx, &expected) in &expected_main {
            assert!(
                (result.policy[idx] - expected).abs() < 1e-5,
                "main policy[{idx}] = {} expected {} (root-STM black)",
                result.policy[idx],
                expected
            );
        }
        for (i, &p) in result.policy.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            assert!(
                expected_main.contains_key(&i),
                "main policy[{i}] = {p} but no root child maps to that STM index"
            );
        }

        // ---- Aux policy: exact mass check on every visited grandchild. ----
        let aux = s.aux_opponent_policy().expect("should have aux");
        let best_child_idx = *root_children
            .iter()
            .max_by_key(|&&i| s.nodes[i].visit_count)
            .unwrap();
        let best_child = &s.nodes[best_child_idx];
        let total_grandchild_visits: u32 = best_child
            .children
            .iter()
            .map(|&gi| s.nodes[gi].visit_count)
            .sum();
        assert!(total_grandchild_visits > 0);

        let mut expected_aux: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();
        for &gi in &best_child.children {
            let gc = &s.nodes[gi];
            if gc.visit_count == 0 {
                continue;
            }
            let mv = gc.action.expect("grandchild must have a move");
            let idx =
                serialization::stm_policy_index(&mv, root_stm).expect("move must be in the table");
            *expected_aux.entry(idx).or_insert(0.0) +=
                gc.visit_count as f32 / total_grandchild_visits as f32;
        }
        for (&idx, &expected) in &expected_aux {
            assert!(
                (aux[idx] - expected).abs() < 1e-5,
                "aux[{idx}] = {} expected {} (root-STM black)",
                aux[idx],
                expected
            );
        }
        for (i, &p) in aux.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            assert!(
                expected_aux.contains_key(&i),
                "aux[{i}] = {p} but no visited grandchild maps to that STM index"
            );
        }
    }

    #[test]
    fn resignation_disabled_never_fires() {
        let mut search = MctsSearch::new(Box::new(HeuristicEvaluator));
        search.config_mut().resign.enabled = false;
        for _ in 0..20 {
            assert!(!search.record_and_check_resign(Color::White, 0.0));
        }
    }
}
