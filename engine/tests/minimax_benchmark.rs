//! Benchmark: node counts for naive vs optimized minimax and search_with_policy.
//! Run with: cargo test --release --test minimax_benchmark -- --nocapture

mod helpers;

use helpers::naive_minimax::naive_search;
use helpers::positions::play_n_moves;
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

#[test]
fn midgame_node_comparison() {
    let mut state = match play_n_moves(10) {
        Some(s) => s,
        None => {
            println!("Game ended before mid-game, skipping");
            return;
        }
    };

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

#[test]
fn search_with_policy_benchmark() {
    let mut state = GameState::new();

    let t0 = std::time::Instant::now();
    let search_result = minimax::search(&mut state, 3).unwrap();
    let search_time = t0.elapsed();

    let t0 = std::time::Instant::now();
    let policy_result = minimax::search_with_policy(&mut state, 3).unwrap();
    let policy_time = t0.elapsed();

    println!("\n=== search_with_policy depth 3 from starting position ===");
    println!(
        "search():             {} nodes in {:.3}s",
        search_result.nodes,
        search_time.as_secs_f64()
    );
    println!(
        "search_with_policy(): {} nodes in {:.3}s ({} move scores)",
        policy_result.nodes,
        policy_time.as_secs_f64(),
        policy_result.move_scores.len()
    );
    println!(
        "Overhead: {:.2}x nodes, {:.2}x time",
        policy_result.nodes as f64 / search_result.nodes as f64,
        policy_time.as_secs_f64() / search_time.as_secs_f64()
    );
}
