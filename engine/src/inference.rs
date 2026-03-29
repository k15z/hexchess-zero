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
            Ok(Self { session: Mutex::new(session) })
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
