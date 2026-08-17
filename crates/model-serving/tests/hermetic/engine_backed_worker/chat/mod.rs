use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationFailureReason,
    ChatGenerationOutput, ChatGenerationSettings, ChatMessage, ChatModelCapabilities,
    ChatToolChoice, ChatToolDefinition, ExpertMemoryMode, MAX_IPC_FRAME_BYTES,
    MlxMemorySnapshotSource, MtpRuntimeState, ProtocolReader, ProtocolWriter, RequestId,
    SpeculativePrefillRuntimeState, WorkerCommand, WorkerEvent, WorkerExpertResidencySnapshot,
    WorkerMlxMemorySnapshot, WorkerPromptProcessingPhase, WorkerPromptWorkReuse,
};
use astronomical_model_serving::{
    EngineBackedWorker, EngineGenerationStart, EngineLoadResult, ExpertResidencyTelemetry,
    GeneratedToken, GenerationFinalization, InferenceEngine, InferenceEngineError,
    MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry, ModelFactory,
    ModelGeneratedTokenTranslation, ModelGenerationOutputError, ModelGenerationProcessor,
    PreparedInferenceRequest, PreparedModelGeneration, WorkerRuntimeError,
};
use tokio::{
    io::{AsyncWrite, duplex, split},
    task::JoinHandle,
    time::timeout,
};

mod expert_memory_mode;
mod generation_lifecycle;
mod memory_breakdown;
mod model_visible_correction;
mod prefill_progress;
mod prompt_cache_stats;
mod ready_and_model_lifecycle;
mod scripted_chat_test_doubles;
mod scripted_model_factory_test_doubles;
mod support;
mod tracking_chat_engine;

struct ScriptedInferenceRequest {
    prompt_token_count: usize,
}

impl ScriptedInferenceRequest {
    const fn new(prompt_token_count: usize) -> Self {
        Self { prompt_token_count }
    }
}

impl PreparedInferenceRequest for ScriptedInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        self.prompt_token_count
    }
}

use scripted_chat_test_doubles::{
    CorrectionRequestingProcessor, MalformedFinishProcessor, ScriptedChatEngine,
    ScriptedChatProcessor,
};
use scripted_model_factory_test_doubles::{
    FirstCreationFailsScriptedModelFactory, LazyScriptedModelFactory,
};
use support::{
    chat_command, close_worker_transport, next_event, ready_event, ready_event_with_load_details,
    ready_event_with_speculative_prefill_load_details,
};
use tracking_chat_engine::TrackingChatEngine;
