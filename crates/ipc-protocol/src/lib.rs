#![forbid(unsafe_code)]

mod base64_bytes;
mod chat_generation;
#[cfg(test)]
mod chat_generation_tests;
mod chat_generation_validation;
#[cfg(test)]
mod chat_generation_validation_limits_tests;
#[cfg(test)]
mod chat_generation_validation_tests;
mod daemon_client;
mod daemon_listener;
mod daemon_protocol;
#[cfg(test)]
mod daemon_protocol_tests;
mod daemon_transport_error;
mod embeddings;
#[cfg(test)]
mod embeddings_tests;
mod expert_layer_seating;
mod image_generation;
#[cfg(test)]
mod image_generation_tests;
mod message_codec;
mod persistent_prompt_cache_diagnostics;
#[cfg(test)]
mod persistent_prompt_cache_diagnostics_tests;
#[cfg(test)]
mod prompt_processing_progress_protocol_tests;
mod protocol_error;
mod protocol_message;
#[cfg(test)]
mod protocol_message_tests;
mod protocol_reader;
mod protocol_writer;
mod worker_chunking_configuration;
#[cfg(test)]
mod worker_chunking_configuration_tests;
mod worker_event_diagnostics;
mod worker_model_configuration;
#[cfg(test)]
mod worker_model_configuration_tests;
mod worker_startup_configuration;
#[cfg(test)]
mod worker_startup_configuration_tests;

pub use chat_generation::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatGenerationSettings, ChatImageInput, ChatMessage, ChatModelCapabilities, ChatToolChoice,
    ChatToolDefinition, MAXIMUM_CHAT_SCHEMA_JSON_BYTES, StructuredGenerationConstraint,
    structured_regex_dfa_pattern,
};
pub use chat_generation_validation::ChatGenerationValidationError;
pub use daemon_client::DaemonIpcClient;
pub use daemon_listener::{DaemonIpcListener, StreamingResponseWriter};
pub use daemon_protocol::{
    DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonCatalogEntry, DaemonDownloadJob,
    DaemonListedModel, DaemonRequest, DaemonResponse, DaemonWorkerStatus,
};
pub use daemon_transport_error::DaemonTransportError;
pub use embeddings::{
    EmbeddingEncodingFormat, EmbeddingsCommand, EmbeddingsFailureReason, EmbeddingsValidationError,
};
pub use expert_layer_seating::{EXPERT_LAYER_SEATING_LIMIT, ExpertLayerSeatingCounts};
pub use image_generation::{
    GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationCompletionValidationError, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, ImageGenerationSettings, ImageGenerationValidationError,
    WorkerEmbeddingCapabilities, WorkerModelCapabilities, WorkerModelCapabilitiesValidationError,
};
pub use message_codec::{
    decode_command, decode_daemon_request, decode_daemon_response, decode_event, encode_command,
    encode_daemon_request, encode_daemon_response, encode_event,
};
pub use persistent_prompt_cache_diagnostics::{
    WorkerInMemoryPrefixAdoptionOutcome, WorkerPersistentPromptCacheExpectedBlockHashPrefix,
    WorkerPersistentPromptCacheLookupOutcome, WorkerPersistentPromptCacheMissReason,
    WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};
pub use protocol_error::ProtocolError;
pub use protocol_message::{
    ExpertMemoryMode, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, RequestId, WorkerCommand,
    WorkerEvent, WorkerExpertLayerTopology, WorkerExpertResidencySnapshot,
    WorkerExpertStreamingStats, WorkerMemoryCeilingUtilizationSnapshot, WorkerMlxMemorySnapshot,
    WorkerPromptProcessingPhase, WorkerPromptWorkReuse,
};
pub use protocol_reader::ProtocolReader;
pub use protocol_writer::ProtocolWriter;
pub use worker_chunking_configuration::{
    WorkerChunkingConfiguration, graph_submission_layer_interval,
};
pub use worker_model_configuration::{
    WorkerAutoregressiveModelConfiguration, WorkerEmbeddingModelConfiguration,
    WorkerEmbeddingModelFamily, WorkerFlux2KleinModelConfiguration,
    WorkerImageGenerationModelFamily, WorkerLoadedAutoregressiveModelRuntimeConfiguration,
    WorkerLoadedModelRuntimeConfiguration, WorkerModelConfiguration,
    WorkerQwenImage21ModelConfiguration,
};
pub use worker_startup_configuration::{
    WorkerDecodeExpertCachePolicy, WorkerDiagnosticQwen3_5EngineOverride, WorkerLogLevel,
    WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};
