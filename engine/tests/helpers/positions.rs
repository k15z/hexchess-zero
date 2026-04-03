//! Shared test position helpers.

use hexchess_engine::eval::EvalWeights;
use hexchess_engine::game::GameState;
use hexchess_engine::minimax;

/// Play `n` moves from the starting position using depth-1 search.
/// Returns `None` if the game ends before `n` moves.
pub fn play_n_moves(n: usize) -> Option<GameState> {
    let mut state = GameState::new();
    for _ in 0..n {
        if state.is_game_over() {
            return None;
        }
        match minimax::search(&mut state, 1, &EvalWeights::material_only()) {
            Some(r) => state.apply_move(r.best_move),
            None => return None,
        }
    }
    if state.is_game_over() {
        None
    } else {
        Some(state)
    }
}
