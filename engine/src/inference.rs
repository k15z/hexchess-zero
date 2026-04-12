//! ONNX-based neural network evaluator for MCTS.
//!
//! Requires the `onnx` feature flag: `cargo build --features onnx`

/// Softmax the 3-dim WDL logits into a probability vector `[W, D, L]`.
///
/// This is the real output of the value head: MCTS stores, averages, and
/// backpropagates these distributions rather than the scalar collapse so the
/// draw mass is preserved for callers that use WDL directly and for contempt shaping.
pub fn softmax_wdl(wdl_logits: &[f32]) -> [f32; 3] {
    assert_eq!(wdl_logits.len(), 3);
    let w = wdl_logits[0];
    let d = wdl_logits[1];
    let l = wdl_logits[2];
    let max = w.max(d).max(l);
    let ew = (w - max).exp();
    let ed = (d - max).exp();
    let el = (l - max).exp();
    let sum = ew + ed + el;
    [ew / sum, ed / sum, el / sum]
}

/// Collapse WDL logits to a scalar value in `[-1, 1]` via `W_prob - L_prob`.
///
/// Kept for callers that only need the scalar projection (tests, legacy
/// tooling). MCTS itself now consumes the full distribution from
/// [`softmax_wdl`].
pub fn wdl_to_value(wdl_logits: &[f32]) -> f32 {
    let p = softmax_wdl(wdl_logits);
    p[0] - p[2]
}

#[cfg(feature = "onnx")]
mod onnx_impl {
    use std::path::Path;
    use std::sync::Mutex;

    use ort::session::Session;

    use crate::game::GameState;
    use crate::mcts::{EvalResult, Evaluator};
    use crate::serialization;

    /// Neural network evaluator backed by ONNX Runtime.
    pub struct OnnxEvaluator {
        session: Mutex<Session>,
    }

    impl OnnxEvaluator {
        /// Load an ONNX model from the given file path.
        ///
        /// `intra_threads` controls the number of threads ONNX Runtime uses for
        /// intra-op parallelism (e.g. matrix multiplications). Pass `0` to let
        /// ORT auto-detect based on available cores, or a specific number to pin.
        pub fn from_path_with_threads(
            path: impl AsRef<Path>,
            intra_threads: usize,
        ) -> Result<Self, ort::Error> {
            let session = Session::builder()?
                .with_intra_threads(intra_threads)?
                .commit_from_file(path)?;
            Ok(Self {
                session: Mutex::new(session),
            })
        }

        /// Load an ONNX model with default thread settings (auto-detect cores).
        pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ort::Error> {
            Self::from_path_with_threads(path, 0)
        }
    }

    impl Evaluator for OnnxEvaluator {
        fn evaluate(&self, state: &GameState) -> EvalResult {
            self.evaluate_batch(&[state]).into_iter().next().unwrap()
        }

        fn evaluate_batch(&self, states: &[&GameState]) -> Vec<EvalResult> {
            let n = states.len();
            assert!(n > 0, "evaluate_batch called with empty slice");

            let c = serialization::NUM_CHANNELS;
            let d = serialization::BOARD_DIM;
            let per_board = c * d * d;

            // Build a single flat buffer for the whole batch: [N, C, H, W].
            let mut data = Vec::with_capacity(n * per_board);
            for state in states {
                let flat = serialization::encode_board(state);
                data.extend_from_slice(&flat);
            }

            let shape: Vec<i64> = vec![n as i64, c as i64, d as i64, d as i64];
            let input = ort::value::Tensor::from_array((shape, data.into_boxed_slice()))
                .expect("failed to create tensor");
            let mut session = self.session.lock().unwrap();
            let mut outputs = session
                .run(ort::inputs![input])
                .expect("ONNX inference failed");

            // Extract policy logits: shape [N, num_move_indices]
            let policy_value = outputs.remove("policy").expect("no 'policy' output");
            let policy_array = policy_value
                .try_extract_array::<f32>()
                .expect("failed to extract policy");

            // Extract WDL logits: shape [N, 3]
            let value_value = outputs.remove("value").expect("no 'value' output");
            let value_array = value_value
                .try_extract_array::<f32>()
                .expect("failed to extract value");

            // Extract MLH (moves-left head): shape [N, 1]
            let mlh_value = outputs.remove("mlh").expect("no 'mlh' output");
            let mlh_array = mlh_value
                .try_extract_array::<f32>()
                .expect("failed to extract mlh");

            // Extract STV (short-term value) logits: shape [N, 3]
            let stv_value = outputs.remove("stv").expect("no 'stv' output");
            let stv_array = stv_value
                .try_extract_array::<f32>()
                .expect("failed to extract stv");

            // Extract aux_policy logits: shape [N, num_move_indices]
            let aux_policy_value = outputs
                .remove("aux_policy")
                .expect("no 'aux_policy' output");
            let aux_policy_array = aux_policy_value
                .try_extract_array::<f32>()
                .expect("failed to extract aux_policy");

            let policy_size = serialization::num_move_indices();
            let policy_flat = policy_array.as_slice().expect("non-contiguous policy");
            let value_flat = value_array.as_slice().expect("non-contiguous value");
            let mlh_flat = mlh_array.as_slice().expect("non-contiguous mlh");
            let stv_flat = stv_array.as_slice().expect("non-contiguous stv");
            let aux_policy_flat = aux_policy_array
                .as_slice()
                .expect("non-contiguous aux_policy");

            let mut results = Vec::with_capacity(n);
            for i in 0..n {
                let logits = &policy_flat[i * policy_size..(i + 1) * policy_size];

                // Softmax over policy logits.
                let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut policy: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum: f32 = policy.iter().sum();
                if sum > 0.0 {
                    for p in &mut policy {
                        *p /= sum;
                    }
                }

                // Full WDL distribution is carried through MCTS; the scalar
                // collapse happens lazily via `q_value()` when contempt is 0.
                let wdl = super::softmax_wdl(&value_flat[i * 3..(i + 1) * 3]);

                // MLH: raw normalized value from network (multiply by mlh_scale=100 to get plies)
                let mlh = mlh_flat[i];

                // STV: short-term WDL softmaxed
                let stv = super::softmax_wdl(&stv_flat[i * 3..(i + 1) * 3]);

                // Aux policy: softmax over opponent's expected reply distribution
                let aux_logits = &aux_policy_flat[i * policy_size..(i + 1) * policy_size];
                let aux_max = aux_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut aux_policy: Vec<f32> =
                    aux_logits.iter().map(|&x| (x - aux_max).exp()).collect();
                let aux_sum: f32 = aux_policy.iter().sum();
                if aux_sum > 0.0 {
                    for p in &mut aux_policy {
                        *p /= aux_sum;
                    }
                }

                results.push(EvalResult {
                    policy,
                    wdl,
                    mlh,
                    stv,
                    aux_policy,
                });
            }

            results
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_impl::OnnxEvaluator;

#[cfg(feature = "tract")]
mod tract_impl {
    use std::path::Path;
    use std::sync::Mutex;

    use tract_onnx::prelude::*;

    use crate::game::GameState;
    use crate::mcts::{EvalResult, Evaluator};
    use crate::serialization;

    type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

    /// Neural network evaluator backed by tract (pure Rust ONNX runtime).
    ///
    /// Works in both native and WASM environments, unlike `OnnxEvaluator` which
    /// requires the C++ ONNX Runtime.
    ///
    /// Note: unlike `OnnxEvaluator`, this does not override `evaluate_batch` —
    /// tract's `SimplePlan` is not designed for batched execution. This is fine
    /// for WASM (single-threaded), but means lower throughput than ORT for
    /// native batch inference.
    pub struct TractEvaluator {
        model: Mutex<TractModel>,
        policy_idx: usize,
        value_idx: usize,
        mlh_idx: usize,
        stv_idx: usize,
        aux_policy_idx: usize,
    }

    fn find_output_index(graph: &TypedModel, name: &str) -> Option<usize> {
        let outlets = graph.output_outlets().ok()?;
        outlets.iter().enumerate().find_map(|(i, &outlet)| {
            let label = graph.outlet_label(outlet)?;
            if label == name { Some(i) } else { None }
        })
    }

    impl TractEvaluator {
        fn from_typed_model(optimized: TypedModel) -> TractResult<Self> {
            let policy_idx = find_output_index(&optimized, "policy")
                .expect("model has no output named 'policy'");
            let value_idx =
                find_output_index(&optimized, "value").expect("model has no output named 'value'");
            let mlh_idx =
                find_output_index(&optimized, "mlh").expect("model has no output named 'mlh'");
            let stv_idx =
                find_output_index(&optimized, "stv").expect("model has no output named 'stv'");
            let aux_policy_idx = find_output_index(&optimized, "aux_policy")
                .expect("model has no output named 'aux_policy'");
            let model = optimized.into_runnable()?;
            Ok(Self {
                model: Mutex::new(model),
                policy_idx,
                value_idx,
                mlh_idx,
                stv_idx,
                aux_policy_idx,
            })
        }

        /// Load an ONNX model from a file path.
        pub fn from_path(path: impl AsRef<Path>) -> TractResult<Self> {
            let optimized = tract_onnx::onnx().model_for_path(path)?.into_optimized()?;
            Self::from_typed_model(optimized)
        }

        /// Load an ONNX model from in-memory bytes.
        pub fn from_bytes(bytes: &[u8]) -> TractResult<Self> {
            let mut cursor = std::io::Cursor::new(bytes);
            let optimized = tract_onnx::onnx()
                .model_for_read(&mut cursor)?
                .into_optimized()?;
            Self::from_typed_model(optimized)
        }

        fn run_inference(&self, state: &GameState) -> EvalResult {
            let c = serialization::NUM_CHANNELS;
            let d = serialization::BOARD_DIM;

            let flat = serialization::encode_board(state);
            let input: Tensor = tract_ndarray::Array4::from_shape_vec((1, c, d, d), flat.to_vec())
                .expect("shape mismatch")
                .into();

            let outputs = self
                .model
                .lock()
                .unwrap()
                .run(tvec![input.into()])
                .expect("tract inference failed");

            let policy_tensor = outputs[self.policy_idx]
                .to_array_view::<f32>()
                .expect("failed to extract policy");
            let value_tensor = outputs[self.value_idx]
                .to_array_view::<f32>()
                .expect("failed to extract value");
            let mlh_tensor = outputs[self.mlh_idx]
                .to_array_view::<f32>()
                .expect("failed to extract mlh");
            let stv_tensor = outputs[self.stv_idx]
                .to_array_view::<f32>()
                .expect("failed to extract stv");
            let aux_policy_tensor = outputs[self.aux_policy_idx]
                .to_array_view::<f32>()
                .expect("failed to extract aux_policy");

            let logits: &[f32] = policy_tensor.as_slice().unwrap();

            // Softmax over policy logits.
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut policy: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum: f32 = policy.iter().sum();
            if sum > 0.0 {
                for p in &mut policy {
                    *p /= sum;
                }
            }

            let wdl = crate::inference::softmax_wdl(value_tensor.as_slice().unwrap());
            let mlh = mlh_tensor.as_slice().unwrap()[0];
            let stv = crate::inference::softmax_wdl(stv_tensor.as_slice().unwrap());

            // Softmax over aux_policy logits
            let aux_logits: &[f32] = aux_policy_tensor.as_slice().unwrap();
            let aux_max = aux_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut aux_policy: Vec<f32> =
                aux_logits.iter().map(|&x| (x - aux_max).exp()).collect();
            let aux_sum: f32 = aux_policy.iter().sum();
            if aux_sum > 0.0 {
                for p in &mut aux_policy {
                    *p /= aux_sum;
                }
            }

            EvalResult {
                policy,
                wdl,
                mlh,
                stv,
                aux_policy,
            }
        }
    }

    impl Evaluator for TractEvaluator {
        fn evaluate(&self, state: &GameState) -> EvalResult {
            self.run_inference(state)
        }
    }
}

#[cfg(feature = "tract")]
pub use tract_impl::TractEvaluator;

#[cfg(test)]
mod tests {
    use crate::game::GameState;
    use crate::mcts::Evaluator;
    use crate::serialization;

    #[test]
    fn test_wdl_to_value_clear_win() {
        // Large W logit → value ≈ 1.0
        let v = super::wdl_to_value(&[10.0, -10.0, -10.0]);
        assert!((v - 1.0).abs() < 1e-4, "expected ~1.0, got {v}");
    }

    #[test]
    fn test_wdl_to_value_clear_loss() {
        // Large L logit → value ≈ -1.0
        let v = super::wdl_to_value(&[-10.0, -10.0, 10.0]);
        assert!((v - (-1.0)).abs() < 1e-4, "expected ~-1.0, got {v}");
    }

    #[test]
    fn test_wdl_to_value_clear_draw() {
        // Large D logit → value ≈ 0.0
        let v = super::wdl_to_value(&[-10.0, 10.0, -10.0]);
        assert!(v.abs() < 1e-4, "expected ~0.0, got {v}");
    }

    #[test]
    fn test_wdl_to_value_uniform() {
        // Equal logits → value = 0.0
        let v = super::wdl_to_value(&[0.0, 0.0, 0.0]);
        assert!(v.abs() < 1e-6, "expected 0.0, got {v}");
    }

    #[test]
    fn test_wdl_to_value_positive_when_w_dominates() {
        // W > L → positive value (value inversion canary)
        let v = super::wdl_to_value(&[5.0, 0.0, 0.0]);
        assert!(
            v > 0.0,
            "W-dominant logits should give positive value, got {v}"
        );
    }

    /// Requires .data/gen1/model/best.onnx — run with `cargo test --features tract -- --ignored`
    #[test]
    #[ignore]
    #[cfg(feature = "tract")]
    fn test_tract_evaluator_loads_and_infers() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../.data/gen1/model/best.onnx");
        let eval = super::TractEvaluator::from_path(model_path).expect("load failed");
        let state = GameState::new();
        let result = eval.evaluate(&state);

        assert_eq!(result.policy.len(), serialization::num_move_indices());
        let wdl_sum = result.wdl[0] + result.wdl[1] + result.wdl[2];
        assert!((wdl_sum - 1.0).abs() < 1e-4, "wdl sum {wdl_sum} != 1.0");
        for &p in &result.wdl {
            assert!((0.0..=1.0).contains(&p), "wdl probability {p} out of range");
        }
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "policy sum {sum} != 1.0");
    }

    /// Requires .data/gen1/model/best.onnx — run with `cargo test --features tract,onnx -- --ignored`
    #[test]
    #[ignore]
    #[cfg(all(feature = "tract", feature = "onnx"))]
    fn test_tract_matches_onnx() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../.data/gen1/model/best.onnx");
        let tract = super::TractEvaluator::from_path(model_path).expect("tract load failed");
        let onnx = super::OnnxEvaluator::from_path(model_path).expect("onnx load failed");
        let state = GameState::new();

        let tract_result = tract.evaluate(&state);
        let onnx_result = onnx.evaluate(&state);

        for i in 0..3 {
            assert!(
                (tract_result.wdl[i] - onnx_result.wdl[i]).abs() < 0.01,
                "wdl[{i}] mismatch: tract={}, onnx={}",
                tract_result.wdl[i],
                onnx_result.wdl[i],
            );
        }

        let tract_top = tract_result
            .policy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let onnx_top = onnx_result
            .policy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(tract_top, onnx_top, "top move index mismatch");
    }
}
