//! How a Prism forward failure reads to the shared Qwen 3.5 generation loop.
//!
//! MLX failures keep their runtime classification, so a forward that does
//! not fit under the memory ceiling still reaches the worker as a refused
//! request rather than a fatal one, as it does for the Qwen 3.5 models.

use runtime_integration::MlxRuntimeError;

use super::PrismHadamardModelError;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

impl From<PrismHadamardModelError> for Qwen3_5ExecutionError {
    fn from(prism_model_error: PrismHadamardModelError) -> Self {
        match prism_model_error {
            PrismHadamardModelError::Forward(captured_error) => {
                Self::Runtime(MlxRuntimeError::from(captured_error))
            }
            PrismHadamardModelError::ForwardRuntime(runtime_error) => Self::Runtime(runtime_error),
            PrismHadamardModelError::DecoderStateLayerMismatch { layer_index } => {
                Self::InvalidRequestDecoderState {
                    layer_index,
                    description: "the layer's state kind does not match its attention weights",
                }
            }
            other_prism_model_error => Self::TextModelForward {
                description: other_prism_model_error.to_string(),
            },
        }
    }
}

#[cfg(test)]
#[path = "execution_error_tests.rs"]
mod execution_error_tests;
