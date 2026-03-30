//! Alpha-beta minimax search for hexagonal chess.

use crate::eval;
use crate::game::{GameState, GameStatus};
use crate::movegen::Move;

/// Result of a minimax search.
pub struct MinimaxResult {
    pub best_move: Move,
    pub score: i32,
    pub nodes: u64,
}

/// Run alpha-beta search at the given depth and return the best move.
///
/// `depth` is the number of plies to search. Panics if the position is terminal.
pub fn search(state: &mut GameState, depth: u32) -> MinimaxResult {
    assert!(depth >= 1, "minimax depth must be >= 1");
    let moves = state.legal_moves();
    assert!(!moves.is_empty(), "minimax called on terminal position");

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

    MinimaxResult {
        best_move,
        score: best_score,
        nodes,
    }
}

/// Negamax with alpha-beta pruning.
fn negamax(state: &mut GameState, depth: u32, mut alpha: i32, beta: i32, nodes: &mut u64) -> i32 {
    *nodes += 1;

    // Terminal check
    match state.status() {
        GameStatus::Ongoing => {}
        GameStatus::Checkmate(winner) => {
            return if winner == state.side_to_move() {
                10_000
            } else {
                -10_000
            };
        }
        _ => return 0, // draws
    }

    if depth == 0 {
        return eval::evaluate(state);
    }

    let moves = state.legal_moves();
    // Should not be empty since we already checked for terminal above,
    // but guard just in case.
    if moves.is_empty() {
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
        let result = search(&mut state, 1);
        assert!(result.nodes > 0);
        assert!(result.score.abs() <= 10_000);
    }

    #[test]
    fn depth2_returns_move() {
        let mut state = GameState::new();
        let result = search(&mut state, 2);
        assert!(result.nodes > 0);
    }
}
