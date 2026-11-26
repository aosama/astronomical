//! The resident engine's bridge from the engine-neutral execution contract
//! onto its state, written once for both Qwen 3.5 engines.

crate::qwen3_5_core::neural_engine::neural_engine_prefill_lifecycle::qwen3_5_inference_execution_bridge!(
    super::engine_state::Qwen3_5InferenceExecution,
    super::engine_state::Qwen3_5EngineState
);
