#![forbid(unsafe_code)]

mod astro_ssd_prompt_cache_diagnostics;
#[cfg(test)]
mod astro_ssd_prompt_cache_diagnostics_tests;
mod base64_bytes;
mod chat_generation;
#[cfg(test)]
mod chat_generation_command_test_fixtures;
#[cfg(test)]
mod chat_generation_tests;
mod chat_generation_validation;
#[cfg(test)]
mod chat_generation_validation_limits_tests;
#[cfg(test)]
mod chat_generation_validation_tests;
mod custom_metal_kernel_report;
mod daemon_client;
mod daemon_listener;
mod daemon_protocol;
#[cfg(test)]
mod daemon_protocol_tests;
mod daemon_transport_error;
mod decisions;
#[cfg(test)]
mod decisions_tests;
mod draft_model_status;
mod embeddings;
#[cfg(test)]
mod embeddings_tests;
mod exhaustive_table;
mod expert_layer_seating;
mod image_generation;
#[cfg(test)]
mod image_generation_tests;
mod message_codec;
mod model_load_failure_identity;
mod model_load_in_flight;
mod neural_engine_chunk_report;
mod neural_engine_prefill_status;
#[cfg(test)]
mod prompt_processing_progress_protocol_tests;
mod protocol_error;
mod protocol_message;
#[cfg(test)]
mod protocol_message_tests;
mod protocol_reader;
#[cfg(test)]
mod protocol_round_trip_test_support;
mod protocol_writer;
mod reply_memory_record;
mod request_drafting_statistics;
mod speech;
mod worker_chunking_configuration;
#[cfg(test)]
mod worker_chunking_configuration_tests;
mod worker_event_diagnostics;
mod worker_memory_ledger_snapshot;
mod worker_model_capabilities;
mod worker_model_configuration;
#[cfg(test)]
mod worker_model_configuration_tests;
mod worker_startup_configuration;
#[cfg(test)]
mod worker_startup_configuration_tests;

pub use astro_ssd_prompt_cache_diagnostics::{
    WorkerAstroSsdPromptCacheExpectedBlockHashPrefix, WorkerAstroSsdPromptCacheLookupOutcome,
    WorkerAstroSsdPromptCacheMissReason, WorkerAstroSsdPromptCacheRequestDiagnostics,
    WorkerAstroSsdPromptCacheStartupCleanupCategory,
    WorkerAstroSsdPromptCacheStartupCleanupEvidence, WorkerInMemoryPrefixAdoptionOutcome,
};
pub use chat_generation::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatGenerationSettings, ChatImageInput, ChatMessage, ChatModelCapabilities, ChatToolChoice,
    ChatToolDefinition, MAXIMUM_CHAT_SCHEMA_JSON_BYTES, OUT_OF_GRAPHICS_MEMORY_ADVICE,
    StructuredGenerationConstraint, structured_regex_dfa_pattern,
};
pub use chat_generation_validation::ChatGenerationValidationError;
pub use custom_metal_kernel_report::{
    CustomMetalKernelProbeFailureKind, CustomMetalKernelProbeOutcome,
    CustomMetalKernelProbeVerdict, CustomMetalKernelReport,
    PROBE_FAILURE_SUMMARY_MAXIMUM_CHARACTERS, bounded_probe_failure_summary,
};
pub use daemon_client::DaemonIpcClient;
pub use daemon_listener::{AcceptedDaemonConnection, DaemonIpcListener, StreamingResponseWriter};
pub use daemon_protocol::{
    DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonCatalogEntry, DaemonDownloadJob,
    DaemonListedModel, DaemonModelUnloadStatus, DaemonRequest, DaemonResponse, DaemonWorkerStatus,
    ImageGenerateRequest, MAXIMUM_SPEECH_AUDIO_FRAME_BYTES, SpeakGenerateRequest,
};
pub use daemon_transport_error::DaemonTransportError;
pub use decisions::{
    DecisionTrainingCommand, DecisionsCommand, DecisionsValidationError,
    MINIMUM_DECISION_TOKENS_PER_QUESTION, WorkerDecisionCapabilities,
};
pub use draft_model_status::{
    BUNDLED_DRAFTER_FOLDER_NAMES, DraftModelGeneration, DraftModelLocation, DraftModelStatus,
};
pub use embeddings::{
    EmbeddingEncodingFormat, EmbeddingsCommand, EmbeddingsFailureReason, EmbeddingsValidationError,
};
pub use expert_layer_seating::{EXPERT_LAYER_SEATING_LIMIT, ExpertLayerSeatingCounts};
pub use image_generation::{
    GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationCompletionValidationError, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, ImageGenerationSettings, ImageGenerationValidationError,
    WorkerEmbeddingCapabilities,
};
pub use message_codec::{
    decode_command, decode_daemon_request, decode_daemon_response, decode_event, encode_command,
    encode_daemon_request, encode_daemon_response, encode_event,
};
pub use model_load_failure_identity::{
    ARTIFACT_VALIDATION_FAILED_PREFIX, ENGINE_INITIALIZATION_FAILED_REASON,
    MEMORY_GEOMETRY_FAILED_REASON, ModelLoadFailureIdentity,
    PROCESSOR_INITIALIZATION_FAILED_REASON, QWEN_ENGINE_INITIALIZATION_FAILED_REASON,
    RESIDENCY_ADMISSION_FAILED_REASON, identify_model_load_failure,
};
pub use model_load_in_flight::ModelLoadInFlight;
pub use neural_engine_chunk_report::{NeuralEngineChunkPlacement, NeuralEngineChunkReport};
pub use neural_engine_prefill_status::NeuralEnginePrefillStatus;
pub use protocol_error::ProtocolError;
pub use protocol_message::{
    ExpertMemoryMode, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, RequestId, WorkerCommand,
    WorkerDenseLayerResidency, WorkerEvent, WorkerExpertLayerTopology,
    WorkerExpertResidencySnapshot, WorkerExpertStreamingStats,
    WorkerMemoryCeilingUtilizationSnapshot, WorkerMlxMemorySnapshot, WorkerPromptLayerProgress,
    WorkerPromptProcessingPhase, WorkerPromptWorkReuse,
};
pub use protocol_reader::ProtocolReader;
pub use protocol_writer::ProtocolWriter;
pub use reply_memory_record::ReplyMemoryRecord;
pub use request_drafting_statistics::RequestDraftingStatistics;
pub use speech::{
    SpeechCommand, SpeechResultMetadata, SpokenAudioSegment, WorkerSpeechCapabilities,
};
pub use worker_chunking_configuration::{
    WorkerChunkingConfiguration, graph_submission_layer_interval,
};
pub use worker_memory_ledger_snapshot::{
    ActiveMemoryCeilingEnforcement, WorkerMemoryLedgerSnapshot,
};
pub use worker_model_capabilities::{
    WorkerModelCapabilities, WorkerModelCapabilitiesValidationError,
};
pub use worker_model_configuration::{
    WorkerAutoregressiveModelConfiguration, WorkerDecisionModelConfiguration,
    WorkerEmbeddingModelConfiguration, WorkerImageGenerationModelConfiguration,
    WorkerLoadedAutoregressiveModelRuntimeConfiguration, WorkerLoadedModelRuntimeConfiguration,
    WorkerModelConfiguration, WorkerSpeechModelConfiguration,
};
pub use worker_startup_configuration::{
    WorkerDecodeAstroExpertCachePolicy, WorkerLogLevel, WorkerRuntimeFeatureConfiguration,
    WorkerStartupConfiguration,
};
