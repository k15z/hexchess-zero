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
            let flat = serialization::encode_board(&state.board);
            let c = serialization::NUM_CHANNELS;
            let d = serialization::BOARD_DIM;

            let shape: Vec<i64> = vec![1, c as i64, d as i64, d as i64];
            let input = ort::value::Tensor::from_array((shape, flat.to_vec().into_boxed_slice()))
                .expect("failed to create tensor");
            let mut session = self.session.lock().unwrap();
            let mut outputs = session
                .run(ort::inputs![input])
                .expect("ONNX inference failed");

            // Extract policy logits via ndarray view
            let policy_value = outputs.remove("policy").expect("no 'policy' output");
            let policy_array = policy_value
                .try_extract_array::<f32>()
                .expect("failed to extract policy");

            // Softmax
            let max_logit = policy_array.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut policy: Vec<f32> = policy_array.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum: f32 = policy.iter().sum();
            if sum > 0.0 {
                for p in &mut policy {
                    *p /= sum;
                }
            }

            // Extract value
            let value_value = outputs.remove("value").expect("no 'value' output");
            let value_array = value_value
                .try_extract_array::<f32>()
                .expect("failed to extract value");
            let value = *value_array.iter().next().expect("empty value");

            (policy, value)
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_impl::OnnxEvaluator;
