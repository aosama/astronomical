use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, EmbeddingsFailureReason, ImageGenerationCommand,
    ImageGenerationFailureReason, ImageGenerationPhase, RequestId,
    WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::time::Instant;

use crate::serving::decisions_executor::DecisionsResultSender;
use crate::serving::speech_executor::SpeechResultSender;
use crate::worker::memory::limit::PendingMlxMemoryLimitUpdate;
use crate::worker::process::failure_relaunch::ContainedFailureRelaunch;
use crate::{
    ChatGenerationStreamEvent, CompletedToolCall, CompletionAttributionLog, DecisionsOutput,
    EmbeddingsExecutionError, EmbeddingsOutput, GenerationPerformanceLog, GenerationStartError,
    ImageGenerationExecutionError, ImageGenerationOutput, MlxMemoryLimitUpdateOutcome,
    PromptCacheClearOutcome, RuntimeModelPolicy, WorkerControlError, WorkerHealthSnapshot,
    WorkerProcess, WorkerTerminationOutcome,
};

/// The owned, loop-carried state of the worker event loop.
///
/// `run_worker` constructs this once; every select! branch and command arm
/// then takes `&mut WorkerLoopRuntime` instead of a dozen individual `&mut`
/// references, which is what let the loop's decomposition stay readable.
/// Field-for-field these are exactly the locals the loop used to declare.
pub(in crate::worker) struct WorkerLoopRuntime {
    pub worker_process: WorkerProcess,
    pub health_snapshot: Arc<RwLock<WorkerHealthSnapshot>>,
    pub is_ready: bool,
    pub model_load_deadline: Option<Instant>,
    pub active_generation: Option<ActiveWorkerRequest>,
    pub performance_log: GenerationPerformanceLog,
    pub completion_log: CompletionAttributionLog,
    pub model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    pub pending_mlx_memory_limit_update: Option<PendingMlxMemoryLimitUpdate>,
    pub pending_prompt_cache_clear: Option<crate::PendingPromptCacheClear>,
    pub active_generation_permits: Arc<Semaphore>,
    pub generation_queue_permits: Arc<Semaphore>,
    pub model_load_timeout: Duration,
    pub cancellation_acknowledgement_timeout: Duration,
    pub image_generation_timeouts: crate::ImageGenerationTimeouts,
    pub speech_timeouts: crate::SpeechTimeouts,
    pub contained_failure_relaunch: ContainedFailureRelaunch,
    pub idle_model_unload: crate::worker::idle_model_unload::IdleModelUnloadClock,
}

/// The work kind is defined beside the executor contract it belongs to; the
/// worker loop carries it from admission to dispatch.
pub(crate) use crate::serving::chat_generation_executor::ChatWork;

/// One chat command together with the work it asks for.
pub(crate) struct QueuedChatRequest {
    pub(crate) generation_command: ChatGenerationCommand,
    pub(crate) chat_work: ChatWork,
}

pub(crate) enum WorkerLoopCommand {
    Generate {
        active_generation_permit: OwnedSemaphorePermit,
        queued_chat_request: QueuedChatRequest,
        start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
        stream_event_sender: mpsc::Sender<ChatGenerationStreamEvent>,
    },
    GenerateImage {
        active_generation_permit: OwnedSemaphorePermit,
        generation_command: ImageGenerationCommand,
        start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
        image_result_sender:
            mpsc::Sender<Result<ImageGenerationOutput, ImageGenerationExecutionError>>,
        admitted_at: Instant,
        queue_wait_elapsed: std::time::Duration,
    },
    GenerateEmbeddings {
        active_generation_permit: OwnedSemaphorePermit,
        embeddings_command: astronomical_ipc_protocol::EmbeddingsCommand,
        start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
        embeddings_result_sender: mpsc::Sender<Result<EmbeddingsOutput, EmbeddingsExecutionError>>,
        admitted_at: Instant,
        queue_wait_elapsed: std::time::Duration,
    },
    GenerateDecisions {
        active_generation_permit: OwnedSemaphorePermit,
        decisions_command: astronomical_ipc_protocol::DecisionsCommand,
        start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
        decisions_result_sender: DecisionsResultSender,
    },
    GenerateSpeech {
        active_generation_permit: OwnedSemaphorePermit,
        speech_command: astronomical_ipc_protocol::SpeechCommand,
        start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
        speech_result_sender: SpeechResultSender,
    },
    Shutdown {
        shutdown_sender: oneshot::Sender<Result<WorkerTerminationOutcome, WorkerControlError>>,
    },
    RestartWorker {
        worker_executable_path: PathBuf,
        model_policy_catalog: Arc<std::collections::HashMap<String, RuntimeModelPolicy>>,
        worker_startup_configuration: Option<WorkerStartupConfiguration>,
        expected_configuration_generation: String,
        restart_sender:
            oneshot::Sender<Result<WorkerRuntimeFeatureConfiguration, WorkerControlError>>,
    },
    UpdateMlxMemoryLimit {
        effective_mlx_memory_ceiling_bytes: u64,
        configuration_generation: String,
        update_sender: oneshot::Sender<Result<MlxMemoryLimitUpdateOutcome, WorkerControlError>>,
    },
    RestartWorkerForMemoryDemotion {
        worker_executable_path: PathBuf,
        model_policy_catalog: Arc<std::collections::HashMap<String, RuntimeModelPolicy>>,
        worker_startup_configuration: WorkerStartupConfiguration,
        expected_configuration_generation: String,
        demotion_sender:
            oneshot::Sender<Result<WorkerRuntimeFeatureConfiguration, WorkerControlError>>,
    },
    UpdateModelPolicyCatalog {
        model_policy_catalog: Arc<std::collections::HashMap<String, RuntimeModelPolicy>>,
        update_sender: oneshot::Sender<Result<(), WorkerControlError>>,
    },
    ClearPromptCache {
        model_id: Option<String>,
        clear_sender: oneshot::Sender<Result<PromptCacheClearOutcome, WorkerControlError>>,
    },
}

/// The worker loop can own exactly one request modality at a time.
pub(crate) enum ActiveWorkerRequest {
    Chat(ActiveGeneration),
    Image(ActiveImageGeneration),
    Embeddings(ActiveEmbeddingsGeneration),
    Decisions(ActiveDecisionsGeneration),
    Speech(ActiveSpeechGeneration),
}

impl ActiveWorkerRequest {
    pub(crate) fn request_id(&self) -> RequestId {
        astronomical_config::exhaustive_table!(
            self,
            request => request.request_id,
            [
                Self::Chat,
                Self::Image,
                Self::Embeddings,
                Self::Decisions,
                Self::Speech
            ]
        )
    }

    pub(crate) fn chat(&self) -> Option<&ActiveGeneration> {
        match self {
            Self::Chat(request) => Some(request),
            Self::Image(_) | Self::Embeddings(_) | Self::Decisions(_) | Self::Speech(_) => None,
        }
    }

    pub(crate) fn chat_mut(&mut self) -> Option<&mut ActiveGeneration> {
        match self {
            Self::Chat(request) => Some(request),
            Self::Image(_) | Self::Embeddings(_) | Self::Decisions(_) | Self::Speech(_) => None,
        }
    }

    pub(crate) fn image_mut(&mut self) -> Option<&mut ActiveImageGeneration> {
        match self {
            Self::Image(request) => Some(request),
            Self::Chat(_) | Self::Embeddings(_) | Self::Decisions(_) | Self::Speech(_) => None,
        }
    }

    pub(crate) fn embeddings_mut(&mut self) -> Option<&mut ActiveEmbeddingsGeneration> {
        match self {
            Self::Embeddings(request) => Some(request),
            Self::Chat(_) | Self::Image(_) | Self::Decisions(_) | Self::Speech(_) => None,
        }
    }

    /// The active chat request's stream-event channel, or `None` for any other
    /// modality. Cloned for the worker loop's `tokio::select!` so the branch
    /// owns its own handle to the same channel.
    pub(crate) fn chat_stream_event_sender(
        &self,
    ) -> Option<mpsc::Sender<ChatGenerationStreamEvent>> {
        match self {
            Self::Chat(request) => Some(request.stream_event_sender.clone()),
            Self::Image(_) | Self::Embeddings(_) | Self::Decisions(_) | Self::Speech(_) => None,
        }
    }

    /// The active image request's result channel, or `None` for any other
    /// modality.
    pub(crate) fn image_result_sender(
        &self,
    ) -> Option<mpsc::Sender<Result<ImageGenerationOutput, ImageGenerationExecutionError>>> {
        match self {
            Self::Image(request) => Some(request.image_result_sender.clone()),
            Self::Chat(_) | Self::Embeddings(_) | Self::Decisions(_) | Self::Speech(_) => None,
        }
    }

    /// The active embeddings request's result channel, or `None` for any other
    /// modality.
    pub(crate) fn embeddings_result_sender(
        &self,
    ) -> Option<mpsc::Sender<Result<EmbeddingsOutput, EmbeddingsExecutionError>>> {
        match self {
            Self::Embeddings(request) => Some(request.embeddings_result_sender.clone()),
            Self::Chat(_) | Self::Image(_) | Self::Decisions(_) | Self::Speech(_) => None,
        }
    }
}

impl ActiveWorkerRequest {
    pub(crate) fn decisions_mut(&mut self) -> Option<&mut ActiveDecisionsGeneration> {
        match self {
            Self::Decisions(request) => Some(request),
            Self::Chat(_) | Self::Image(_) | Self::Embeddings(_) | Self::Speech(_) => None,
        }
    }

    /// The active decisions request's result channel, or `None` for any other
    /// modality.
    pub(crate) fn decisions_result_sender(&self) -> Option<DecisionsResultSender> {
        match self {
            Self::Decisions(request) => Some(request.decisions_result_sender.clone()),
            Self::Chat(_) | Self::Image(_) | Self::Embeddings(_) | Self::Speech(_) => None,
        }
    }

    /// When a request that runs as one forward pass must have finished, or
    /// `None` for a request that reports progress instead.
    pub(crate) fn single_shot_execution_deadline(&self) -> Option<Instant> {
        match self {
            Self::Embeddings(request) => Some(request.execution_deadline),
            Self::Decisions(request) => Some(request.execution_deadline),
            Self::Chat(_) | Self::Image(_) | Self::Speech(_) => None,
        }
    }
}

impl ActiveWorkerRequest {
    pub(crate) fn speech_mut(&mut self) -> Option<&mut ActiveSpeechGeneration> {
        match self {
            Self::Speech(request) => Some(request),
            _ => None,
        }
    }

    /// The active speech request's result channel, or `None` for any other
    /// modality.
    pub(crate) fn speech_result_sender(&self) -> Option<SpeechResultSender> {
        match self {
            Self::Speech(request) => Some(request.speech_result_sender.clone()),
            _ => None,
        }
    }
}

/// One speech request the worker is speaking: its audio so far, segment by
/// segment, and the two bounds a stuck engine is caught by.
pub(crate) struct ActiveSpeechGeneration {
    pub(crate) _active_generation_permit: OwnedSemaphorePermit,
    pub(crate) request_id: RequestId,
    pub(crate) execution_deadline: Instant,
    pub(crate) progress_stall_deadline: Instant,
    pub(crate) progress_stall_timeout: Duration,
    /// 16-bit samples of every segment spoken so far, in order.
    pub(crate) pcm16_bytes: Vec<u8>,
    /// The most audio this request may collect; it scales with the pace
    /// asked for, since slow speech of the same text is longer.
    pub(crate) maximum_audio_bytes: usize,
    /// The worker spoke more than `maximum_audio_bytes`; the rest is
    /// dropped and the request fails when it completes.
    pub(crate) has_exceeded_audio_limit: bool,
    pub(crate) completed_segments: u32,
    pub(crate) terminal_received_at: Option<Instant>,
    pub(crate) speech_result_sender: SpeechResultSender,
    pub(crate) terminal_outcome:
        Option<Result<crate::SpeechOutput, astronomical_speech_contract::SpeechFailure>>,
}

pub(crate) struct ActiveDecisionsGeneration {
    pub(crate) _active_generation_permit: OwnedSemaphorePermit,
    pub(crate) request_id: RequestId,
    pub(crate) terminal_received_at: Option<tokio::time::Instant>,
    pub(crate) execution_deadline: Instant,
    pub(crate) decisions_result_sender: DecisionsResultSender,
    pub(crate) terminal_outcome:
        Option<Result<DecisionsOutput, astronomical_decision_contract::DecisionFailure>>,
}

pub(crate) struct ActiveImageGeneration {
    pub(crate) _active_generation_permit: OwnedSemaphorePermit,
    pub(crate) request_id: RequestId,
    pub(crate) model_id: String,
    pub(crate) settings: astronomical_ipc_protocol::ImageGenerationSettings,
    pub(crate) admitted_at: Instant,
    pub(crate) queue_wait_elapsed: std::time::Duration,
    pub(crate) swap_load_elapsed: std::time::Duration,
    pub(crate) execution_started_at: Instant,
    pub(crate) execution_deadline: Instant,
    pub(crate) progress_stall_deadline: Instant,
    pub(crate) progress_stall_timeout: std::time::Duration,
    pub(crate) latest_phase: Option<ImageGenerationPhase>,
    pub(crate) latest_completed_steps: u16,
    pub(crate) latest_elapsed_millis: u64,
    pub(crate) terminal_received_at: Option<Instant>,
    pub(crate) image_result_sender:
        mpsc::Sender<Result<ImageGenerationOutput, ImageGenerationExecutionError>>,
    pub(crate) terminal_outcome:
        Option<Result<ImageGenerationOutput, ImageGenerationFailureReason>>,
}

pub(crate) struct ActiveEmbeddingsGeneration {
    pub(crate) _active_generation_permit: OwnedSemaphorePermit,
    pub(crate) request_id: RequestId,
    pub(crate) terminal_received_at: Option<tokio::time::Instant>,
    pub(crate) execution_deadline: Instant,
    pub(crate) embeddings_result_sender:
        mpsc::Sender<Result<EmbeddingsOutput, EmbeddingsExecutionError>>,
    pub(crate) terminal_outcome: Option<Result<EmbeddingsOutput, EmbeddingsFailureReason>>,
}

pub(crate) struct ActiveGeneration {
    pub(crate) _active_generation_permit: OwnedSemaphorePermit,
    pub(crate) generated_token_count: u16,
    pub(crate) generation_started_at: Option<Instant>,
    pub(crate) generation_preparation_started_at: Option<Instant>,
    pub(crate) generation_preparation_elapsed_millis: Option<u64>,
    pub(crate) first_decode_forward_elapsed_millis: Option<u64>,
    pub(crate) time_to_first_output_millis: Option<u64>,
    pub(crate) final_resident_expert_count: Option<u32>,
    pub(crate) final_resident_expert_payload_bytes: Option<u64>,
    pub(crate) latest_generation_progress_token_count: u16,
    pub(crate) max_output_tokens: u16,
    pub(crate) next_sequence_number: u16,
    pub(crate) next_tool_call_index: u16,
    /// Tool calls emitted so far, accumulated for completion attribution.
    /// Bounded at write time so the completion log never grows without bound.
    pub(crate) completed_tool_calls: Vec<CompletedToolCall>,
    pub(crate) request_started_at: Instant,
    pub(crate) prefill_elapsed_millis: u64,
    pub(crate) maximum_mlx_peak_memory_bytes: Option<u64>,
    pub(crate) last_mlx_active_memory_bytes: Option<u64>,
    pub(crate) request_id: RequestId,
    pub(crate) stream_event_sender: mpsc::Sender<ChatGenerationStreamEvent>,
    /// Whether this request generates or only counts its prompt tokens; a
    /// count completes through the same path but stays out of the
    /// performance and attribution logs.
    pub(crate) chat_work: ChatWork,
}

impl ActiveGeneration {
    /// Folds one observed MLX peak sample into the request's durable peak
    /// record.
    ///
    /// The MLX peak counter resets at allocator-cache cleanups, so a later
    /// sample can read a lower window peak than an earlier one; the durable
    /// claim is a request maximum, so it must retain the highest window seen.
    /// Every event that carries a worker memory snapshot must route through
    /// here so the durable record never under-reports a transient the status
    /// endpoint visibly published.
    pub(crate) fn fold_observed_mlx_peak_memory_bytes(&mut self, observed_peak_memory_bytes: u64) {
        self.maximum_mlx_peak_memory_bytes = Some(
            self.maximum_mlx_peak_memory_bytes
                .unwrap_or(0)
                .max(observed_peak_memory_bytes),
        );
    }
}

/// The mutable worker-loop state that generation, memory, and process
/// handling all operate on: the supervised process, the readiness/model-load
/// deadline pair, the request that may be active, and the two telemetry
/// logs. Giving the repeating cluster one name keeps function signatures at
/// the loop boundary readable; every consumer destructures it back into
/// locals, so field borrows stay disjoint exactly as with individual
/// parameters.
pub(crate) struct WorkerLoopState<'a> {
    pub worker_process: &'a mut WorkerProcess,
    pub health_snapshot: &'a Arc<RwLock<WorkerHealthSnapshot>>,
    pub is_ready: &'a mut bool,
    pub model_load_deadline: &'a mut Option<Instant>,
    pub active_request: &'a mut Option<ActiveWorkerRequest>,
    pub performance_log: &'a mut GenerationPerformanceLog,
    pub completion_log: &'a mut CompletionAttributionLog,
    pub model_load_timeout: Duration,
}
