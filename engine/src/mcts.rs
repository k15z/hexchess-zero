//! Monte Carlo Tree Search (AlphaZero variant) for hexagonal chess.
//!
//! Uses arena-style allocation with nodes stored in a `Vec<MctsNode>` and
//! referenced by index, avoiding pointer overhead and improving cache locality.

use std::collections::HashMap;

use crate::eval;
use crate::game::GameState;
use crate::movegen::Move;
use crate::serialization;

use rand::Rng;

// ---------------------------------------------------------------------------
// Evaluator trait
// ---------------------------------------------------------------------------

/// Trait for position evaluation, returning a policy vector and scalar value.
///
/// The policy vector is a probability distribution over move indices (length
/// `serialization::num_move_indices()`). The value is in [-1, 1] where
/// positive means good for the side to move.
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, state: &GameState) -> (Vec<f32>, f32);
}

// ---------------------------------------------------------------------------
// HeuristicEvaluator
// ---------------------------------------------------------------------------

/// Uniform policy over legal moves with material-aware value estimate.
pub struct HeuristicEvaluator;

impl Evaluator for HeuristicEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<f32>, f32) {
        let moves = state.legal_moves();
        let num_indices = serialization::num_move_indices();
        let mut policy = vec![0.0f32; num_indices];

        if !moves.is_empty() {
            let prob = 1.0 / moves.len() as f32;
            for mv in &moves {
                if let Some(idx) = serialization::move_to_index(mv) {
                    policy[idx] = prob;
                }
            }
        }

        // Material + terminal-aware value mapped to [-1, 1].
        // Terminal states (±10000) saturate to ±1.0, material diffs scale smoothly.
        let cp = eval::evaluate(state) as f32;
        let value = (cp / 400.0).tanh();

        (policy, value)
    }
}

// ---------------------------------------------------------------------------
// MCTS Node (arena-allocated)
// ---------------------------------------------------------------------------

struct MctsNode {
    /// Move that led to this node (None for root).
    action: Option<Move>,
    /// Index of action in the policy vector.
    action_index: Option<usize>,
    /// Parent node index in the arena (retained for tree reuse in future).
    #[allow(dead_code)]
    parent: Option<usize>,
    /// Children node indices in the arena.
    children: Vec<usize>,
    /// Number of times this node was visited.
    visit_count: u32,
    /// Sum of value estimates through this node.
    value_sum: f64,
    /// Prior probability from the policy network.
    prior: f32,
    /// Whether this node has been expanded.
    is_expanded: bool,
}

impl MctsNode {
    fn new(action: Option<Move>, action_index: Option<usize>, parent: Option<usize>, prior: f32) -> Self {
        Self {
            action,
            action_index,
            parent,
            children: Vec::new(),
            visit_count: 0,
            value_sum: 0.0,
            prior,
            is_expanded: false,
        }
    }

    /// Mean value Q(node) = value_sum / visit_count.
    fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------

pub struct SearchResult {
    /// Best move chosen by the search.
    pub best_move: Move,
    /// Visit-count distribution over all move indices (normalized to sum to 1).
    pub policy: Vec<f32>,
    /// Value estimate of the root position.
    pub value: f32,
    /// Total nodes in the search tree.
    pub nodes_searched: u32,
}

// ---------------------------------------------------------------------------
// MCTS configuration
// ---------------------------------------------------------------------------

/// Optional Dirichlet noise parameters for root exploration.
#[derive(Clone, Debug)]
pub struct DirichletConfig {
    /// Mixing weight for noise (typically 0.25).
    pub epsilon: f32,
    /// Dirichlet concentration parameter (typically 0.3 for chess).
    pub alpha: f64,
}

impl Default for DirichletConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.25,
            alpha: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// MCTS Search
// ---------------------------------------------------------------------------

pub struct MctsSearch {
    nodes: Vec<MctsNode>,
    evaluator: Box<dyn Evaluator>,
    /// Exploration constant for PUCT (default ~1.5).
    c_puct: f32,
    /// Optional Dirichlet noise config for root priors.
    dirichlet: Option<DirichletConfig>,
    /// Transposition table: zobrist_hash -> (policy, value).
    /// Avoids re-running NN inference for positions seen before.
    tt: HashMap<u64, (Vec<f32>, f32)>,
}

impl MctsSearch {
    pub fn new(evaluator: Box<dyn Evaluator>) -> Self {
        Self {
            nodes: Vec::new(),
            evaluator,
            c_puct: 1.5,
            dirichlet: None,
            tt: HashMap::new(),
        }
    }

    /// Set the PUCT exploration constant.
    pub fn set_c_puct(&mut self, c_puct: f32) {
        self.c_puct = c_puct;
    }

    /// Enable Dirichlet noise at the root.
    pub fn set_dirichlet(&mut self, config: Option<DirichletConfig>) {
        self.dirichlet = config;
    }

    /// Clear the tree for a new search. Keeps the transposition table.
    pub fn reset(&mut self) {
        self.nodes.clear();
    }

    /// Clear everything including the transposition table.
    pub fn reset_all(&mut self) {
        self.nodes.clear();
        self.tt.clear();
    }

    /// Run MCTS for `num_simulations` iterations (greedy, temperature = 0).
    pub fn search(&mut self, state: &GameState, num_simulations: u32) -> SearchResult {
        self.search_with_temperature(state, num_simulations, 0.0)
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
        self.reset();

        // Create root node.
        let root = MctsNode::new(None, None, None, 1.0);
        self.nodes.push(root);

        // Run simulations.
        for _ in 0..num_simulations {
            let mut current_state = state.clone();
            self.simulate(0, &mut current_state);
        }

        self.extract_result(state, temperature)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// One simulation: select -> expand/evaluate -> backpropagate.
    fn simulate(&mut self, root_idx: usize, state: &mut GameState) {
        // Phase 1: SELECT — walk down tree using PUCT.
        let mut node_idx = root_idx;
        let mut path = vec![node_idx];

        while self.nodes[node_idx].is_expanded && !self.nodes[node_idx].children.is_empty() {
            node_idx = self.select_child(node_idx);
            // Apply the child's action to advance the game state.
            let action = self.nodes[node_idx]
                .action
                .expect("non-root node must have an action");
            state.apply_move(action);
            path.push(node_idx);
        }

        // Phase 2: EXPAND + EVALUATE.
        let value = if state.is_game_over() {
            state.outcome_value().unwrap_or(0.0) as f64
        } else {
            // Check transposition table before calling the (expensive) evaluator
            let hash = state.board.zobrist_hash;
            let (policy, value) = if let Some(cached) = self.tt.get(&hash) {
                cached.clone()
            } else {
                let result = self.evaluator.evaluate(state);
                self.tt.insert(hash, result.clone());
                result
            };
            self.expand(node_idx, state, policy);
            value as f64
        };

        // Phase 3: BACKPROPAGATE.
        // The value is from the perspective of the side to move at `node_idx`.
        // As we walk back up, we negate at each level.
        self.backpropagate(&path, value);
    }

    /// Select the child of `node_idx` with the highest PUCT score.
    fn select_child(&self, node_idx: usize) -> usize {
        let parent = &self.nodes[node_idx];
        let parent_visits_sqrt = (parent.visit_count as f64).sqrt();

        let mut best_idx = parent.children[0];
        let mut best_score = f64::NEG_INFINITY;

        for &child_idx in &parent.children {
            let child = &self.nodes[child_idx];
            // Negate: child stores value from its own side-to-move perspective,
            // but the parent wants to maximize from the parent's perspective.
            let q = -child.q_value();
            let u = self.c_puct as f64
                * child.prior as f64
                * parent_visits_sqrt
                / (1.0 + child.visit_count as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }

        best_idx
    }

    /// Expand `node_idx` by creating children for all legal moves.
    fn expand(&mut self, node_idx: usize, state: &GameState, policy: Vec<f32>) {
        let legal_moves = state.legal_moves();
        if legal_moves.is_empty() {
            return;
        }

        // Map legal moves to (move, index, raw_prior).
        let mut children_info: Vec<(Move, usize, f32)> = Vec::with_capacity(legal_moves.len());
        let mut prior_sum = 0.0f32;

        for mv in &legal_moves {
            if let Some(idx) = serialization::move_to_index(mv) {
                let p = if idx < policy.len() { policy[idx] } else { 0.0 };
                prior_sum += p;
                children_info.push((*mv, idx, p));
            } else {
                // Move not in index table — give it a small prior.
                children_info.push((*mv, usize::MAX, 0.0));
            }
        }

        // Renormalize priors over legal moves.
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

        // Apply Dirichlet noise at the root (node_idx == 0).
        if node_idx == 0
            && let Some(ref config) = self.dirichlet {
                self.apply_dirichlet_noise(&mut children_info, config.epsilon, config.alpha);
            }

        // Create child nodes.
        for (mv, mv_idx, prior) in children_info {
            let action_index = if mv_idx == usize::MAX { None } else { Some(mv_idx) };
            let child = MctsNode::new(Some(mv), action_index, Some(node_idx), prior);
            let child_idx = self.nodes.len();
            self.nodes.push(child);
            self.nodes[node_idx].children.push(child_idx);
        }

        self.nodes[node_idx].is_expanded = true;
    }

    /// Apply Dirichlet noise to the priors in children_info.
    fn apply_dirichlet_noise(
        &self,
        children_info: &mut [(Move, usize, f32)],
        epsilon: f32,
        alpha: f64,
    ) {
        let n = children_info.len();
        if n == 0 {
            return;
        }

        let noise = sample_dirichlet(n, alpha);
        for (i, info) in children_info.iter_mut().enumerate() {
            info.2 = (1.0 - epsilon) * info.2 + epsilon * noise[i] as f32;
        }
    }

    /// Backpropagate value up the path, negating at each level.
    fn backpropagate(&mut self, path: &[usize], leaf_value: f64) {
        // `leaf_value` is from the perspective of the side to move at the leaf.
        // As we go up, alternate the sign.
        let mut value = leaf_value;
        for &node_idx in path.iter().rev() {
            self.nodes[node_idx].visit_count += 1;
            self.nodes[node_idx].value_sum += value;
            value = -value;
        }
    }

    /// Extract the search result after all simulations.
    fn extract_result(&self, _state: &GameState, temperature: f32) -> SearchResult {
        let root = &self.nodes[0];
        let num_indices = serialization::num_move_indices();
        let mut policy = vec![0.0f32; num_indices];

        // Build visit-count distribution.
        let total_child_visits: u32 = root.children.iter().map(|&i| self.nodes[i].visit_count).sum();

        // Populate policy with raw visit counts (will normalize later).
        for &child_idx in &root.children {
            let child = &self.nodes[child_idx];
            if let Some(mv_idx) = child.action_index
                && mv_idx < num_indices {
                    policy[mv_idx] = child.visit_count as f32;
                }
        }

        // Select best move.
        let best_child_idx = if temperature < 1e-4 {
            // Greedy: highest visit count.
            *root
                .children
                .iter()
                .max_by_key(|&&i| self.nodes[i].visit_count)
                .expect("root must have children")
        } else {
            // Temperature-based selection.
            self.select_by_temperature(root, temperature)
        };

        let best_move = self.nodes[best_child_idx]
            .action
            .expect("child must have an action");

        // Normalize policy to sum to 1.
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
                        if lp > max_log { max_log = lp; }
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

        let value = root.q_value() as f32;

        SearchResult {
            best_move,
            policy,
            value,
            nodes_searched: self.nodes.len() as u32,
        }
    }

    /// Select a child of root proportionally to visit_count^(1/temperature).
    fn select_by_temperature(&self, root: &MctsNode, temperature: f32) -> usize {
        let inv_temp = 1.0 / temperature;
        let weights: Vec<f64> = root
            .children
            .iter()
            .map(|&i| (self.nodes[i].visit_count as f64).powf(inv_temp as f64))
            .collect();

        let total: f64 = weights.iter().sum();
        if total == 0.0 {
            return root.children[0];
        }

        // Sample from the categorical distribution defined by weights.
        let mut rng = rand::rng();
        let threshold: f64 = rng.random::<f64>() * total;
        let mut cumulative = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                return root.children[i];
            }
        }
        // Fallback (rounding).
        *root.children.last().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Dirichlet sampling
// ---------------------------------------------------------------------------

/// Sample from a symmetric Dirichlet distribution Dir(alpha, ..., alpha)
/// with `n` components.
///
/// Uses Gamma sampling via the Ahrens-Dieter method (suitable for alpha < 1)
/// combined with Marsaglia-Tsang for alpha >= 1.
fn sample_dirichlet(n: usize, alpha: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut samples: Vec<f64> = (0..n).map(|_| sample_gamma(&mut rng, alpha)).collect();
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
fn sample_gamma(rng: &mut impl Rng, alpha: f64) -> f64 {
    if alpha < 1.0 {
        // Boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        let u: f64 = rng.random();
        sample_gamma_gt1(rng, alpha + 1.0) * u.powf(1.0 / alpha)
    } else {
        sample_gamma_gt1(rng, alpha)
    }
}

/// Marsaglia-Tsang method for Gamma(alpha, 1) where alpha >= 1.
fn sample_gamma_gt1(rng: &mut impl Rng, alpha: f64) -> f64 {
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
        let (policy, value) = evaluator.evaluate(&state);

        // Starting position is symmetric — value should be near zero.
        assert!(value.abs() < 1e-4, "value should be ~0.0 for starting pos, got {}", value);

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
            legal.iter().any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
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
            legal.iter().any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
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
            legal.iter().any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
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
            legal.iter().any(|m| m.from == result.best_move.from && m.to == result.best_move.to),
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
}
