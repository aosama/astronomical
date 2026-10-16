#![deny(unsafe_code)]

mod application;
mod application_build_identity;
mod config;
mod console;
mod http;
mod instance_lock;
mod ipc;
mod library;
mod performance;
mod serving;
mod shutdown_control;
pub mod startup_failure;
mod system_telemetry;
mod worker;

// Re-exported at the crate root because `mod ipc` is private; daemon tests and
// external consumers reference `astronomical_daemon::daemon_ipc_chat_schema`
// directly, and this keeps that path stable across the source reorganization.
pub use ipc::daemon_ipc_chat_schema;

pub use application::{
    build_application, build_application_with_discovered_models,
    build_application_with_download_catalog, build_application_with_full_control,
    build_application_with_full_control_and_download_catalog, build_application_with_shutdown,
    build_development_application_with_reload,
};
pub use application_build_identity::ApplicationBuildIdentity;
pub use library::library_application::{
    build_application_with_full_control_and_library_download,
    build_application_with_library_download,
};

pub use config::config_reload::{
    ConfigReloadDecision, ConfigReloadDiff, ResolvedRuntimeConfig, ResolvedRuntimeConfigError,
    ResolvedRuntimeConfigResolver,
};
pub use config::resolved_configuration_generation::ResolvedConfigurationGeneration;
pub use config::runtime_model_policy::{RuntimeModelGenerationDefaults, RuntimeModelPolicy};
pub use http::chat_diagnostics::{
    OpenAiChatRequestDiagnosticSnapshot, OpenAiChatRequestInfoDiagnosticSnapshot,
    build_openai_chat_request_diagnostic_snapshot,
    build_openai_chat_request_info_diagnostic_snapshot,
};
pub use http::openai_chat_translation::{
    OpenAiChatTranslationError, translate_openai_chat_completion_request,
};
pub use http::openai_responses_assembly::{OpenAiResponsesAssemblyError, OpenAiResponsesCollector};
pub use http::openai_responses_stream::{
    OpenAiResponsesStreamEncoder, OpenAiResponsesStreamEncodingError,
};
pub use http::openai_responses_translation::{
    OpenAiResponsesTranslationError, translate_openai_responses_request,
};
pub use instance_lock::{AstronomicalInstanceLock, AstronomicalInstanceLockError};
pub use ipc::daemon_ipc::{DaemonIpcGenerationContext, DaemonIpcService, start_daemon_ipc_service};
pub use library::library_model_discovery_refresh::LibraryModelDiscoveryRefresh;
pub use library::{
    DiskCapacityQuery, DownloadCatalog, DownloadCatalogEntry, DownloadCatalogError,
    DownloadCatalogFamily, DownloadDiskCapacityCheck, DownloadDiskPreflight,
    DownloadDiskPreflightError, DownloadFileDigest, DownloadJob, DownloadJobError, DownloadJobFile,
    DownloadJobPublicErrorCode, DownloadJobState, DownloadJobStore, DownloadJobStoreError,
    DownloadManifestPreflight, DownloadManifestPreflightError, DownloadPayloadTransfer,
    DownloadPayloadTransferError, DownloadPayloadTransferOutcome, DownloadProgressSnapshot,
    DownloadPublication, DownloadPublicationError, DownloadPublicationRefresh,
    DownloadTransferControl, Fs4DiskCapacityQuery, HubHttpMethod, HubHttpRequest, HubHttpResponse,
    HubHttpResponseError, HubManifestFile, HubPayloadByteStream, HubPayloadFuture,
    HubPayloadRequest, HubPayloadResponse, HubPayloadTransport, HubTransport, HubTransportError,
    HubTransportFuture, HuggingFaceHub, HuggingFaceHubError, HuggingFaceHubLimits,
    HuggingFaceManifest, LibraryDownloadCoordinator, LibraryDownloadCoordinatorError,
    MAXIMUM_CONCURRENT_PAYLOAD_TRANSFERS, PAYLOAD_TRANSFER_SEGMENT_BYTES, ReqwestHubTransport,
    ReqwestHubTransportBuildError,
};
pub use performance::completion_attribution_log::{
    CompletedToolCall, CompletionArgumentsRecord, CompletionAttributionLog,
    CompletionAttributionRecord, CompletionToolCallRecord,
};
pub use performance::daemon_performance_attribution::{
    DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
};
pub use performance::generation_performance_log::{
    GenerationPerformanceLog, GenerationPerformanceRecord, ImageGenerationPerformanceRecord,
};
pub use serving::chat_generation_executor::{
    ChatGenerationExecutor, ChatGenerationStreamErrorCode, ChatGenerationStreamEvent,
    GenerationStartError,
};
pub use serving::embeddings_executor::{EmbeddingsExecutionError, EmbeddingsOutput};
pub use serving::image_generation_executor::{
    ImageGenerationExecutionError, ImageGenerationExecutor, ImageGenerationOutput,
    ImageGenerationTimeouts,
};
pub use serving::serving_session_snapshot::ServingSessionSnapshot;
pub use shutdown_control::ShutdownController;
pub use system_telemetry::parse_macos_memory_pressure_level;
pub use worker::memory::worker_memory_limit::MlxMemoryLimitUpdateOutcome;
pub use worker::process::worker_process::{WorkerProcess, WorkerTerminationOutcome};
pub use worker::worker_cache_clear::PromptCacheClearOutcome;
pub use worker::worker_control_error::WorkerControlError;
pub use worker::worker_handle::{GenerationQueueDepth, WorkerHandle};
pub use worker::worker_health::{
    ActiveRequestProgress, ExpertResidencySnapshot, PendingPromptCacheClear,
    PersistentPromptCacheSummary, WorkerActivity, WorkerHealthSnapshot, WorkerHealthStatus,
};
