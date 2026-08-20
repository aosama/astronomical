use std::path::PathBuf;
use std::sync::Arc;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, RequestId, WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot};
use tokio::time::Instant;

use crate::{
    ChatGenerationStreamEvent, GenerationStartError, MlxMemoryLimitUpdateOutcome,
    PromptCacheClearOutcome, RuntimeModelPolicy, WorkerControlError, WorkerTerminationOutcome,
};

pub(crate) enum WorkerLoopCommand {
    Generate {
        active_generation_permit: OwnedSemaphorePermit,
        generation_command: ChatGenerationCommand,
        start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
        stream_event_sender: mpsc::Sender<ChatGenerationStreamEvent>,
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
        update_sender: oneshot::Sender<Result<MlxMemoryLimitUpdateOutcome, WorkerControlError>>,
    },
    ClearPromptCache {
        model_id: Option<String>,
        clear_sender: oneshot::Sender<Result<PromptCacheClearOutcome, WorkerControlError>>,
    },
}

pub(crate) struct ActiveGeneration {
    pub(crate) _active_generation_permit: OwnedSemaphorePermit,
    pub(crate) generated_token_count: u16,
    pub(crate) generation_started_at: Option<Instant>,
    pub(crate) generation_preparation_started_at: Option<Instant>,
    pub(crate) generation_preparation_elapsed_millis: Option<u64>,
    pub(crate) first_decode_forward_elapsed_millis: Option<u64>,
    pub(crate) time_to_first_output_millis: Option<u64>,
    pub(crate) final_complete_expert_layer_count: Option<u32>,
    pub(crate) final_complete_expert_payload_bytes: Option<u64>,
    pub(crate) final_partial_expert_layer_count: Option<u32>,
    pub(crate) final_partial_expert_payload_bytes: Option<u64>,
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
