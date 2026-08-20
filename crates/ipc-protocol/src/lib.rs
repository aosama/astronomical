#![forbid(unsafe_code)]

mod base64_bytes;
mod chat_generation;
mod chat_generation_validation;
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
    ChatToolDefinition,
};
pub use chat_generation_validation::ChatGenerationValidationError;
pub use image_generation::{
    GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationCompletionValidationError, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, ImageGenerationSettings, ImageGenerationValidationError,
    WorkerModelCapabilities, WorkerModelCapabilitiesValidationError,
};
pub use message_codec::{decode_command, decode_event, encode_command, encode_event};
pub use persistent_prompt_cache_diagnostics::{
    WorkerPersistentPromptCacheExpectedBlockHashPrefix, WorkerPersistentPromptCacheLookupOutcome,
    WorkerPersistentPromptCacheMissReason, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};
pub use protocol_error::ProtocolError;
pub use protocol_message::{
    ExpertMemoryMode, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, MtpDepthStatus,
    MtpRuntimeState, RequestId, SpeculativePrefillRuntimeState, WorkerCommand, WorkerEvent,
    WorkerExpertResidencySnapshot, WorkerMlxMemorySnapshot, WorkerPromptProcessingPhase,
    WorkerPromptWorkReuse,
};
pub use protocol_reader::ProtocolReader;
pub use protocol_writer::ProtocolWriter;
pub use worker_chunking_configuration::{
    WorkerChunkingConfiguration, graph_submission_layer_interval,
};
pub use worker_model_configuration::{
    WorkerAutoregressiveModelConfiguration, WorkerAuxiliaryModelConfiguration,
    WorkerFlux2KleinModelConfiguration, WorkerImageGenerationModelFamily,
    WorkerLoadedAutoregressiveModelRuntimeConfiguration, WorkerLoadedModelRuntimeConfiguration,
    WorkerModelConfiguration, WorkerSpeculativePrefillRuntimeConfiguration,
};
pub use worker_startup_configuration::{
    WorkerLogLevel, WorkerRuntimeFeatureConfiguration, WorkerSpeculativePrefillConfiguration,
    WorkerStartupConfiguration,
};
