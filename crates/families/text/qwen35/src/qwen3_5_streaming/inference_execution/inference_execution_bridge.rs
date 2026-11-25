//! Trait bridge from the engine-neutral execution contract onto the state
//! methods; the methods both Qwen 3.5 engines share are expanded from
//! `shared_execution::inference_execution_bridge`.

use crate::{MlxInferenceExecution, TextInferenceRequest};

use super::engine_state::{Qwen3_5EngineState, Qwen3_5InferenceExecution};

impl MlxInferenceExecution for Qwen3_5InferenceExecution {
    type Request = TextInferenceRequest;

    text_generation::text_shared_inference_execution_methods!(Qwen3_5EngineState);
}
