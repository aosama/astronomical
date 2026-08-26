use std::path::PathBuf;
use std::sync::Arc;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ImageGenerationCommand, ImageGenerationFailureReason,
    ImageGenerationPhase, RequestId, WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot};
use tokio::time::Instant;

use crate::{
    ChatGenerationStreamEvent, GenerationStartError, ImageGenerationExecutionError,
    ImageGenerationOutput, MlxMemoryLimitUpdateOutcome, PromptCacheClearOutcome,
    RuntimeModelPolicy, WorkerControlError, WorkerTerminationOutcome,
};

pub(crate) enum WorkerLoopCommand {
    Generate {
        active_generation_permit: OwnedSemaphorePermit,
        generation_command: ChatGenerationCommand,
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
}

impl ActiveWorkerRequest {
    pub(crate) fn request_id(&self) -> RequestId {
        match self {
            Self::Chat(request) => request.request_id,
            Self::Image(request) => request.request_id,
        }
    }

    pub(crate) fn chat(&self) -> Option<&ActiveGeneration> {
        match self {
            Self::Chat(request) => Some(request),
            Self::Image(_) => None,
        }
    }

    pub(crate) fn chat_mut(&mut self) -> Option<&mut ActiveGeneration> {
        match self {
            Self::Chat(request) => Some(request),
            Self::Image(_) => None,
        }
    }

    pub(crate) fn image_mut(&mut self) -> Option<&mut ActiveImageGeneration> {
        match self {
            Self::Image(request) => Some(request),
            Self::Chat(_) => None,
        }
    }
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
    pub(crate) request_started_at: Instant,
    pub(crate) prefill_elapsed_millis: u64,
    pub(crate) maximum_mlx_peak_memory_bytes: Option<u64>,
    pub(crate) last_mlx_active_memory_bytes: Option<u64>,
    pub(crate) request_id: RequestId,
    pub(crate) stream_event_sender: mpsc::Sender<ChatGenerationStreamEvent>,
}
