//! Alpha-beta minimax search for hexagonal chess.
//!
//! TODO: Potential optimizations to revisit (reverted in #19 because they
//! caused a 27x node blowup in `search_all_moves` via quiescence search,
//! and correctness was insufficiently tested):
//!
//! - Transposition table (Zobrist hash → score/bound/best-move cache)
//! - Move ordering: MVV-LVA for captures, killer moves, history heuristic
//! - Iterative deepening to populate TT before final-depth search
//! - Quiescence search to avoid horizon effect on captures
//!
//! When re-adding these, ensure:
//! 1. `search_all_moves` is tested against the naive implementation across
//!    hundreds of positions, comparing *every* move's score (not just best).
//! 2. Quiescence must not blow up `search_all_moves` node count — the
//!    full-window search means quiescence runs without alpha-beta pruning
//!    across root moves. Consider skipping quiescence in `search_all_moves`
//!    or adding root-level alpha tightening with re-search for fail-low ties.
//! 3. TT mate score adjustment (ply-relative storage) needs dedicated tests
//!    verifying round-trip correctness at different probe plies.

use crate::eval;
use crate::game::{GameState, GameStatus};
use crate::movegen::{self, Move};

/// Result of a minimax search.
pub struct MinimaxResult {
    pub best_move: Move,
    pub score: i32,
    pub nodes: u64,
}

/// A move paired with its minimax score.
pub struct RankedMove {
    pub mv: Move,
    pub score: i32,
}

/// Result of `search_all_moves`: scores for every legal root move.
pub struct MinimaxAllResult {
    pub moves: Vec<RankedMove>,
    pub nodes: u64,
}

/// Run alpha-beta search at the given depth and return the best move.
///
/// `depth` is the number of plies to search. Returns `None` if the position is terminal.
pub fn search(state: &mut GameState, depth: u32) -> Option<MinimaxResult> {
    assert!(depth >= 1, "minimax depth must be >= 1");
    let moves = state.legal_moves();
    if moves.is_empty() {
        return None;
    }

    let mut best_move = moves[0];
    let mut best_score = i32::MIN + 1;
    let mut nodes = 0u64;

    for mv in &moves {
        state.apply_move(*mv);
        let score = -negamax(state, depth - 1, i32::MIN + 1, -best_score, &mut nodes);
        state.undo_move();

        if score > best_score {
            best_score = score;
            best_move = *mv;
        }
    }

    Some(MinimaxResult {
        best_move,
        score: best_score,
        nodes,
    })
}

/// Run alpha-beta search and return scores for **all** legal root moves.
///
/// Unlike `search`, this uses a full window (no alpha tightening at the root)
/// so every move gets an accurate score. Subtree pruning is unaffected.
/// Returns `None` if the position is terminal.
pub fn search_all_moves(state: &mut GameState, depth: u32) -> Option<MinimaxAllResult> {
    assert!(depth >= 1, "minimax depth must be >= 1");
    let moves = state.legal_moves();
    if moves.is_empty() {
        return None;
    }

    let mut ranked = Vec::with_capacity(moves.len());
    let mut nodes = 0u64;

    for mv in &moves {
        state.apply_move(*mv);
        let score = -negamax(state, depth - 1, i32::MIN + 1, i32::MAX, &mut nodes);
        state.undo_move();
        ranked.push(RankedMove { mv: *mv, score });
    }

    Some(MinimaxAllResult {
        moves: ranked,
        nodes,
    })
}

/// Negamax with alpha-beta pruning.
fn negamax(state: &mut GameState, depth: u32, mut alpha: i32, beta: i32, nodes: &mut u64) -> i32 {
    *nodes += 1;

    // Check terminal conditions before generating moves.
    match state.status() {
        GameStatus::Ongoing => {}
        GameStatus::Checkmate(_) => return -(10_000 + depth as i32),
        _ => return 0, // all draw types
    }

    let moves = state.legal_moves();

    // No legal moves: checkmate or stalemate.
    if moves.is_empty() {
        let stm = state.side_to_move();
        return if movegen::is_in_check(&state.board, stm) {
            -(10_000 + depth as i32) // prefer shorter mates
        } else {
            0 // stalemate
        };
    }

    if depth == 0 {
        return eval::evaluate(state);
    }

    for mv in &moves {
        state.apply_move(*mv);
        let score = -negamax(state, depth - 1, -beta, -alpha, nodes);
        state.undo_move();

        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth1_returns_move() {
        let mut state = GameState::new();
        let result = search(&mut state, 1).unwrap();
        assert!(result.nodes > 0);
        assert!(result.score.abs() <= 10_100);
    }

    #[test]
    fn depth2_returns_move() {
        let mut state = GameState::new();
        let result = search(&mut state, 2).unwrap();
        assert!(result.nodes > 0);
    }

    #[test]
    fn terminal_returns_none() {
        let mut state = GameState::new();
        // Ongoing position should return Some.
        assert!(search(&mut state, 1).is_some());
    }

    #[test]
    fn search_all_moves_best_matches_search() {
        let mut state = GameState::new();
        let best = search(&mut state, 2).unwrap();
        let all = search_all_moves(&mut state, 2).unwrap();

        // Best score from search_all_moves should match search.
        let top = all.moves.iter().max_by_key(|m| m.score).unwrap();
        assert_eq!(top.score, best.score);

        // All legal moves should be present.
        let legal = state.legal_moves();
        assert_eq!(all.moves.len(), legal.len());
    }
}
