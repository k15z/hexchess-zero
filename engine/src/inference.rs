//! ONNX-based neural network evaluator for MCTS.
//!
//! Requires the `onnx` feature flag: `cargo build --features onnx`

#[cfg(feature = "onnx")]
mod onnx_impl {
    use std::path::Path;
    use std::sync::Mutex;

    use ort::session::Session;

    use crate::game::GameState;
    use crate::mcts::Evaluator;
    use crate::serialization;

    /// Neural network evaluator backed by ONNX Runtime.
    pub struct OnnxEvaluator {
        session: Mutex<Session>,
    }

    impl OnnxEvaluator {
        /// Load an ONNX model from the given file path.
        pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ort::Error> {
            let session = Session::builder()?
                .with_intra_threads(1)?
                .commit_from_file(path)?;
            Ok(Self {
                session: Mutex::new(session),
            })
        }
    }

    impl Evaluator for OnnxEvaluator {
        fn evaluate(&self, state: &GameState) -> (Vec<f32>, f32) {
            self.evaluate_batch(&[state]).into_iter().next().unwrap()
        }

        fn evaluate_batch(&self, states: &[&GameState]) -> Vec<(Vec<f32>, f32)> {
            let n = states.len();
            assert!(n > 0, "evaluate_batch called with empty slice");

            let c = serialization::NUM_CHANNELS;
            let d = serialization::BOARD_DIM;
            let per_board = c * d * d;

            // Build a single flat buffer for the whole batch: [N, C, H, W].
            let mut data = Vec::with_capacity(n * per_board);
            for state in states {
                let flat = serialization::encode_board(&state.board);
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

            // Extract values: shape [N, 1] or [N]
            let value_value = outputs.remove("value").expect("no 'value' output");
            let value_array = value_value
                .try_extract_array::<f32>()
                .expect("failed to extract value");

            let policy_size = serialization::num_move_indices();
            let policy_flat = policy_array.as_slice().expect("non-contiguous policy");
            let value_flat = value_array.as_slice().expect("non-contiguous value");

            let mut results = Vec::with_capacity(n);
            for i in 0..n {
                let logits = &policy_flat[i * policy_size..(i + 1) * policy_size];

                // Softmax over this sample's logits.
                let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut policy: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum: f32 = policy.iter().sum();
                if sum > 0.0 {
                    for p in &mut policy {
                        *p /= sum;
                    }
                }

                let value = value_flat[i];
                results.push((policy, value));
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
    use crate::mcts::Evaluator;
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
            let model = optimized.into_runnable()?;
            Ok(Self {
                model: Mutex::new(model),
                policy_idx,
                value_idx,
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

        fn run_inference(&self, state: &GameState) -> (Vec<f32>, f32) {
            let c = serialization::NUM_CHANNELS;
            let d = serialization::BOARD_DIM;

            let flat = serialization::encode_board(&state.board);
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

            let logits: &[f32] = policy_tensor.as_slice().unwrap();

            // Softmax over logits.
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut policy: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum: f32 = policy.iter().sum();
            if sum > 0.0 {
                for p in &mut policy {
                    *p /= sum;
                }
            }

            let value = value_tensor.as_slice().unwrap()[0];
            (policy, value)
        }
    }

    impl Evaluator for TractEvaluator {
        fn evaluate(&self, state: &GameState) -> (Vec<f32>, f32) {
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

    /// Requires .data/gen1/model/best.onnx — run with `cargo test --features tract -- --ignored`
    #[test]
    #[ignore]
    #[cfg(feature = "tract")]
    fn test_tract_evaluator_loads_and_infers() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../.data/gen1/model/best.onnx");
        let eval = super::TractEvaluator::from_path(model_path).expect("load failed");
        let state = GameState::new();
        let (policy, value) = eval.evaluate(&state);

        assert_eq!(policy.len(), serialization::num_move_indices());
        assert!(value > -1.0 && value < 1.0, "value {value} out of range");
        let sum: f32 = policy.iter().sum();
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

        let (tract_policy, tract_value) = tract.evaluate(&state);
        let (onnx_policy, onnx_value) = onnx.evaluate(&state);

        assert!(
            (tract_value - onnx_value).abs() < 0.01,
            "value mismatch: tract={tract_value}, onnx={onnx_value}"
        );

        let tract_top = tract_policy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let onnx_top = onnx_policy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(tract_top, onnx_top, "top move index mismatch");
    }
}
