//! Benchmark comparing CoreML vs CPU-only ONNX inference for MCTS.
//!
//! Usage: cargo run --release --features onnx --bin bench_coreml -- <model.onnx>

#[cfg(feature = "onnx")]
fn main() {
    use hexchess_engine::game::GameState;
    use hexchess_engine::inference::OnnxEvaluator;
    use hexchess_engine::mcts::{Evaluator, MctsSearch};
    use std::time::Instant;

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: bench_coreml <model.onnx>");

    let n_evals = 200;
    let n_sims = 500;

    let state = GameState::new();

    // --- Raw inference benchmark ---
    println!("=== Raw inference ({n_evals} evaluations) ===\n");

    // CPU-only
    let cpu_eval =
        OnnxEvaluator::from_path_cpu_only(&model_path, 0).expect("failed to load model (CPU)");
    // Warmup
    for _ in 0..10 {
        cpu_eval.evaluate(&state);
    }
    let t = Instant::now();
    for _ in 0..n_evals {
        cpu_eval.evaluate(&state);
    }
    let cpu_dur = t.elapsed();
    println!(
        "CPU:     {:>7.1}ms total, {:>6.2}ms/eval",
        cpu_dur.as_secs_f64() * 1000.0,
        cpu_dur.as_secs_f64() * 1000.0 / n_evals as f64
    );

    // CoreML (default — will use CoreML on Apple, CPU elsewhere)
    let coreml_eval =
        OnnxEvaluator::from_path_with_threads(&model_path, 0).expect("failed to load model (CoreML)");
    // Warmup
    for _ in 0..10 {
        coreml_eval.evaluate(&state);
    }
    let t = Instant::now();
    for _ in 0..n_evals {
        coreml_eval.evaluate(&state);
    }
    let coreml_dur = t.elapsed();
    println!(
        "CoreML:  {:>7.1}ms total, {:>6.2}ms/eval",
        coreml_dur.as_secs_f64() * 1000.0,
        coreml_dur.as_secs_f64() * 1000.0 / n_evals as f64
    );

    let speedup = cpu_dur.as_secs_f64() / coreml_dur.as_secs_f64();
    println!("Speedup: {speedup:.2}x\n");

    // --- Batched inference benchmark ---
    let batch_size = 32;
    println!("=== Batched inference (batch_size={batch_size}, {n_evals} batches) ===\n");
    let states: Vec<&GameState> = vec![&state; batch_size];

    // CPU batched
    for _ in 0..5 {
        cpu_eval.evaluate_batch(&states);
    }
    let t = Instant::now();
    for _ in 0..n_evals {
        cpu_eval.evaluate_batch(&states);
    }
    let cpu_batch_dur = t.elapsed();
    println!(
        "CPU:     {:>7.1}ms total, {:>6.2}ms/batch",
        cpu_batch_dur.as_secs_f64() * 1000.0,
        cpu_batch_dur.as_secs_f64() * 1000.0 / n_evals as f64
    );

    // CoreML batched
    for _ in 0..5 {
        coreml_eval.evaluate_batch(&states);
    }
    let t = Instant::now();
    for _ in 0..n_evals {
        coreml_eval.evaluate_batch(&states);
    }
    let coreml_batch_dur = t.elapsed();
    println!(
        "CoreML:  {:>7.1}ms total, {:>6.2}ms/batch",
        coreml_batch_dur.as_secs_f64() * 1000.0,
        coreml_batch_dur.as_secs_f64() * 1000.0 / n_evals as f64
    );

    let batch_speedup = cpu_batch_dur.as_secs_f64() / coreml_batch_dur.as_secs_f64();
    println!("Speedup: {batch_speedup:.2}x\n");

    // --- MCTS benchmark ---
    println!("=== MCTS search ({n_sims} simulations) ===\n");

    // CPU MCTS
    drop(cpu_eval);
    let cpu_eval =
        OnnxEvaluator::from_path_cpu_only(&model_path, 0).expect("failed to load model (CPU)");
    let mut cpu_mcts = MctsSearch::new(Box::new(cpu_eval));
    // Warmup
    cpu_mcts.search(&state, 50);
    cpu_mcts.reset();
    let t = Instant::now();
    cpu_mcts.search(&state, n_sims);
    let cpu_mcts_dur = t.elapsed();
    println!(
        "CPU:     {:>7.1}ms ({:.2}ms/sim)",
        cpu_mcts_dur.as_secs_f64() * 1000.0,
        cpu_mcts_dur.as_secs_f64() * 1000.0 / n_sims as f64
    );

    // CoreML MCTS
    drop(coreml_eval);
    let coreml_eval =
        OnnxEvaluator::from_path_with_threads(&model_path, 0).expect("failed to load model (CoreML)");
    let mut coreml_mcts = MctsSearch::new(Box::new(coreml_eval));
    // Warmup
    coreml_mcts.search(&state, 50);
    coreml_mcts.reset();
    let t = Instant::now();
    coreml_mcts.search(&state, n_sims);
    let coreml_mcts_dur = t.elapsed();
    println!(
        "CoreML:  {:>7.1}ms ({:.2}ms/sim)",
        coreml_mcts_dur.as_secs_f64() * 1000.0,
        coreml_mcts_dur.as_secs_f64() * 1000.0 / n_sims as f64
    );

    let mcts_speedup = cpu_mcts_dur.as_secs_f64() / coreml_mcts_dur.as_secs_f64();
    println!("Speedup: {mcts_speedup:.2}x");
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("This benchmark requires the `onnx` feature: cargo run --features onnx --bin bench_coreml");
    std::process::exit(1);
}
