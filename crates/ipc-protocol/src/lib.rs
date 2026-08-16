#![forbid(unsafe_code)]

mod chat_generation;
mod chat_generation_validation;
mod message_codec;
mod persistent_prompt_cache_diagnostics;
mod prompt_processing_chunk_optimization;
mod protocol_error;
mod protocol_message;
mod protocol_reader;
mod protocol_writer;
mod worker_chunking_configuration;
mod worker_startup_configuration;

pub use chat_generation::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatGenerationSettings, ChatImageInput, ChatMessage, ChatModelCapabilities, ChatToolChoice,
    ChatToolDefinition,
};
pub use chat_generation_validation::ChatGenerationValidationError;
pub use message_codec::{decode_command, decode_event, encode_command, encode_event};
pub use persistent_prompt_cache_diagnostics::{
    WorkerPersistentPromptCacheExpectedBlockHashPrefix, WorkerPersistentPromptCacheLookupOutcome,
    WorkerPersistentPromptCacheMissReason, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};
pub use prompt_processing_chunk_optimization::{
    WorkerPromptProcessingChunkCandidateMeasurementSummary,
    WorkerPromptProcessingChunkMeasurementSource, WorkerPromptProcessingChunkOptimizationContext,
    WorkerPromptProcessingChunkOptimizationOutcome, WorkerPromptProcessingChunkSelectionReason,
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
    WorkerChunkingConfiguration, WorkerPromptProcessingChunkSizingPolicy,
    experimental_ssd_paging_graph_submission_layer_interval,
};
pub use worker_startup_configuration::{
    WorkerLogLevel, WorkerRuntimeFeatureConfiguration, WorkerSpeculativePrefillConfiguration,
    WorkerStartupConfiguration,
};
