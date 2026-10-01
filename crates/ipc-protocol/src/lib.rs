#![forbid(unsafe_code)]

mod base64_bytes;
mod chat_generation;
mod chat_generation_validation;
mod daemon_client;
mod daemon_listener;
mod daemon_protocol;
mod daemon_transport_error;
mod embeddings;
mod image_generation;
mod message_codec;
mod persistent_prompt_cache_diagnostics;
mod protocol_error;
mod protocol_message;
mod protocol_reader;
mod protocol_writer;
mod worker_chunking_configuration;
mod worker_event_diagnostics;
mod worker_model_configuration;
mod worker_startup_configuration;

pub use chat_generation::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatGenerationSettings, ChatImageInput, ChatMessage, ChatModelCapabilities, ChatToolChoice,
    ChatToolDefinition, MAX_QWEN_THINKING_CHANNEL_SEED_BYTES, MAXIMUM_CHAT_SCHEMA_JSON_BYTES,
    StructuredGenerationConstraint, structured_regex_dfa_pattern,
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
    WorkerPersistentPromptCacheExpectedBlockHashPrefix, WorkerPersistentPromptCacheLookupOutcome,
    WorkerPersistentPromptCacheMissReason, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};
pub use protocol_error::ProtocolError;
pub use protocol_message::{
    ExpertMemoryMode, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, MtpDepthResolutionReason,
    MtpDepthStatus, MtpRuntimeState, RequestId, SpeculativePrefillRuntimeState, WorkerCommand,
    WorkerEvent, WorkerExpertResidencySnapshot, WorkerMemoryCeilingUtilizationSnapshot,
    WorkerMlxMemorySnapshot, WorkerPromptProcessingPhase, WorkerPromptWorkReuse,
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
    WorkerQwenImage21ModelConfiguration, WorkerSpeculativePrefillRuntimeConfiguration,
};
pub use worker_startup_configuration::{
    WorkerLogLevel, WorkerRuntimeFeatureConfiguration, WorkerSpeculativePrefillConfiguration,
    WorkerStartupConfiguration,
};
