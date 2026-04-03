//! Reference (naive) negamax for comparison — no TT, no ordering, no qsearch.
//!
//! Used by minimax_benchmark.rs and minimax_strength.rs to verify the optimized
//! search produces equivalent or better results.

use hexchess_engine::eval;
use hexchess_engine::game::{GameState, GameStatus};
use hexchess_engine::movegen;

pub fn naive_negamax(
    state: &mut GameState,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    nodes: &mut u64,
) -> i32 {
    *nodes += 1;

    match state.status() {
        GameStatus::Ongoing => {}
        GameStatus::Checkmate(_) => return -(10_000 + depth as i32),
        _ => return 0,
    }

    let moves = state.legal_moves();
    if moves.is_empty() {
        let stm = state.side_to_move();
        return if movegen::is_in_check(&state.board, stm) {
            -(10_000 + depth as i32)
        } else {
            0
        };
    }

    if depth == 0 {
        return eval::evaluate(state);
    }

    for mv in &moves {
        state.apply_move(*mv);
        let score = -naive_negamax(state, depth - 1, -beta, -alpha, nodes);
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

pub fn naive_search(state: &mut GameState, depth: u32) -> Option<(movegen::Move, i32, u64)> {
    let moves = state.legal_moves();
    if moves.is_empty() {
        return None;
    }

    let mut best_move = moves[0];
    let mut best_score = i32::MIN + 1;
    let mut nodes = 0u64;

    for mv in &moves {
        state.apply_move(*mv);
        let score = -naive_negamax(state, depth - 1, i32::MIN + 1, -best_score, &mut nodes);
        state.undo_move();
        if score > best_score {
            best_score = score;
            best_move = *mv;
        }
    }

    Some((best_move, best_score, nodes))
}
