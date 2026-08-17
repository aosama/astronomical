#![deny(unsafe_code)]

mod application;
mod application_build_identity;
mod cache_clear_endpoint;
mod chat_diagnostics;
mod chat_generation_executor;
mod config_reload;
mod config_reload_endpoint;
mod console_assets;
mod generation_output_ceiling;
mod generation_performance_log;
mod instance_lock;
mod maximum_mlx_memory_endpoint;
mod openai_chat_completion;
mod openai_chat_endpoint;
mod openai_chat_stream;
mod openai_chat_translation;
mod openai_models_endpoint;
mod openai_responses_assembly;
mod openai_responses_endpoint;
mod openai_responses_stream;
mod openai_responses_translation;
mod serving_session_snapshot;
mod shutdown_control;
mod status_endpoint;
mod system_telemetry;
mod worker;
mod worker_cache_clear;
mod worker_completion_event;
mod worker_containment;
mod worker_control_error;
mod worker_event_handler;
mod worker_generation_preparation;
mod worker_handle;
mod worker_health;
mod worker_loop_types;
mod worker_memory_limit;
mod worker_model_swap;
mod worker_prefill_progress;
mod worker_process;
mod worker_stderr_tail;

pub use application::{
    build_application, build_application_with_discovered_models,
    build_application_with_full_control, build_application_with_shutdown,
    build_development_application_with_reload,
};
pub use application_build_identity::ApplicationBuildIdentity;

/// Returns the rolling policy shared by the supervisor process log.
#[must_use]
pub fn astronomical_log_rotation() -> tracing_appender::rolling::Rotation {
    tracing_appender::rolling::Rotation::HOURLY
}
pub use chat_diagnostics::{
    OpenAiChatRequestDiagnosticSnapshot, OpenAiChatRequestInfoDiagnosticSnapshot,
    build_openai_chat_request_diagnostic_snapshot,
    build_openai_chat_request_info_diagnostic_snapshot,
};
pub use chat_generation_executor::{
    ChatGenerationExecutor, ChatGenerationStreamErrorCode, ChatGenerationStreamEvent,
    GenerationStartError,
};
pub use config_reload::{
    ConfigReloadDecision, ConfigReloadDiff, ResolvedRuntimeConfig, ResolvedRuntimeConfigError,
    ResolvedRuntimeConfigResolver,
};
pub use generation_performance_log::{GenerationPerformanceLog, GenerationPerformanceRecord};
pub use instance_lock::{AstronomicalInstanceLock, AstronomicalInstanceLockError};
pub use openai_chat_translation::{
    OpenAiChatTranslationError, translate_openai_chat_completion_request,
};
pub use openai_responses_assembly::{OpenAiResponsesAssemblyError, OpenAiResponsesCollector};
pub use openai_responses_stream::{
    OpenAiResponsesStreamEncoder, OpenAiResponsesStreamEncodingError,
};
pub use openai_responses_translation::{
    OpenAiResponsesTranslationError, translate_openai_responses_request,
};
pub use serving_session_snapshot::ServingSessionSnapshot;
pub use shutdown_control::ShutdownController;
pub use system_telemetry::parse_macos_memory_pressure_level;
pub use worker_cache_clear::PromptCacheClearOutcome;
pub use worker_control_error::WorkerControlError;
pub use worker_handle::{GenerationQueueDepth, WorkerHandle};
pub use worker_health::{
    ActiveRequestProgress, ExpertResidencySnapshot, PendingPromptCacheClear,
    PersistentPromptCacheSummary, WorkerActivity, WorkerHealthSnapshot, WorkerHealthStatus,
};
pub use worker_memory_limit::MlxMemoryLimitUpdateOutcome;
pub use worker_process::{WorkerProcess, WorkerTerminationOutcome};
