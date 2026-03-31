//! Benchmark: node counts at depth 4 for naive vs optimized minimax.
//! Run with: cargo test --release --test minimax_benchmark -- --nocapture

mod helpers;

use helpers::naive_minimax::naive_search;
use hexchess_engine::game::GameState;
use hexchess_engine::minimax;

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
