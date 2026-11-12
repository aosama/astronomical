//! Generation serving: request executors and their shared output contracts.

pub(crate) mod chat_generation_executor;
pub(crate) mod decisions_executor;
pub(crate) mod decisions_service;
pub(crate) mod embeddings_executor;
pub(crate) mod helper_model_resolution;
pub(crate) mod image_generation_executor;
pub(crate) mod neural_engine_session_totals;
pub(crate) mod request_seed;
pub(crate) mod serving_session_snapshot;
pub(crate) mod speech_executor;
pub(crate) mod speech_service;
pub(crate) mod structured_output;

pub(crate) use chat_generation_executor::ChatGenerationStartFuture;
pub(crate) use image_generation_executor::GenerationOutcomeStartFuture;
