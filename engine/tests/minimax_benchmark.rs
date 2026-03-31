//! Benchmark: node counts at depth 4 for naive vs optimized minimax.
//! Run with: cargo test --release --test minimax_benchmark -- --nocapture

use hexchess_engine::eval;
use hexchess_engine::game::GameState;
use hexchess_engine::minimax;
use hexchess_engine::movegen;

fn naive_negamax(
    state: &mut GameState,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    nodes: &mut u64,
) -> i32 {
    *nodes += 1;

    match state.status() {
        hexchess_engine::game::GameStatus::Ongoing => {}
        hexchess_engine::game::GameStatus::Checkmate(_) => return -(10_000 + depth as i32),
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

fn naive_search(state: &mut GameState, depth: u32) -> Option<(movegen::Move, i32, u64)> {
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

#[test]
fn depth4_node_comparison() {
    let mut state = GameState::new();

    let t0 = std::time::Instant::now();
    let (_, old_score, old_nodes) = naive_search(&mut state, 4).unwrap();
    let old_time = t0.elapsed();

    let t0 = std::time::Instant::now();
    let new_result = minimax::search(&mut state, 4).unwrap();
    let new_time = t0.elapsed();

    println!("\n=== Depth 4 from starting position ===");
    println!(
        "Naive:     {} nodes in {:.2}s (score={})",
        old_nodes,
        old_time.as_secs_f64(),
        old_score
    );
    println!(
        "Optimized: {} nodes in {:.2}s (score={})",
        new_result.nodes,
        new_time.as_secs_f64(),
        new_result.score
    );
    println!(
        "Node reduction: {:.2}x",
        old_nodes as f64 / new_result.nodes as f64
    );
    println!(
        "Speedup:        {:.2}x",
        old_time.as_secs_f64() / new_time.as_secs_f64()
    );
}

/// Test from a mid-game position (after 10 moves).
#[test]
fn midgame_node_comparison() {
    let mut state = GameState::new();

    // Play 10 moves to reach mid-game.
    for _ in 0..10 {
        if state.is_game_over() {
            break;
        }
        match minimax::search(&mut state, 1) {
            Some(r) => state.apply_move(r.best_move),
            None => break,
        }
    }

    if state.is_game_over() {
        println!("Game ended before mid-game, skipping");
        return;
    }

    let t0 = std::time::Instant::now();
    let (_, old_score, old_nodes) = naive_search(&mut state, 3).unwrap();
    let old_time = t0.elapsed();

    let t0 = std::time::Instant::now();
    let new_result = minimax::search(&mut state, 3).unwrap();
    let new_time = t0.elapsed();

    println!("\n=== Depth 3 from mid-game position ===");
    println!(
        "Naive:     {} nodes in {:.3}s (score={})",
        old_nodes,
        old_time.as_secs_f64(),
        old_score
    );
    println!(
        "Optimized: {} nodes in {:.3}s (score={})",
        new_result.nodes,
        new_time.as_secs_f64(),
        new_result.score
    );
    println!(
        "Node reduction: {:.2}x",
        old_nodes as f64 / new_result.nodes as f64
    );
    println!(
        "Speedup:        {:.2}x",
        old_time.as_secs_f64() / new_time.as_secs_f64()
    );
}
