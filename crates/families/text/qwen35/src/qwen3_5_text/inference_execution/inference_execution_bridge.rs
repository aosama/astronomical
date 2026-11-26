//! The engine's side of the execution contract; see the macro.

use super::engine_state::{Qwen3_5EngineState, Qwen3_5InferenceExecution};
use crate::qwen3_5_core::neural_engine::neural_engine_prefill_lifecycle::qwen3_5_inference_execution_bridge;

qwen3_5_inference_execution_bridge!(Qwen3_5InferenceExecution, Qwen3_5EngineState);
