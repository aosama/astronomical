#![allow(dead_code)]

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};

use astronomical_ipc_protocol::MtpRuntimeState;
use astronomical_ipc_protocol::{ChatGenerationCommand, ChatModelCapabilities};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerActivity,
    WorkerHealthSnapshot, WorkerHealthStatus,
};
use tokio::sync::mpsc;

pub(crate) mod supervisor;

pub const MODEL_ID: &str = "astronomical/application-test-model";

pub struct ScriptedExecutor {
    pub health_snapshot: WorkerHealthSnapshot,
    pub stream_events: Vec<ChatGenerationStreamEvent>,
    received_generation_commands: Arc<Mutex<Vec<ChatGenerationCommand>>>,
    /// Test override: when true, the executor reports a busy worker so the
    /// config-reload endpoint returns HTTP 409.
    pub is_busy_override: bool,
}

impl ScriptedExecutor {
    pub fn ready(stream_events: Vec<ChatGenerationStreamEvent>) -> Self {
        Self {
            health_snapshot: WorkerHealthSnapshot::ready_with_model(
                MODEL_ID.to_owned(),
                ChatModelCapabilities {
                    supports_reasoning: true,
                    supports_tool_calls: true,
                    has_vision: true,
                    max_input_tokens: 241_664,
                    max_output_tokens: 20_480,
                    context_window: 262_144,
                },
                MtpRuntimeState::Disabled,
                None,
            ),
            stream_events,
            received_generation_commands: Arc::new(Mutex::new(Vec::new())),
            is_busy_override: false,
        }
    }

    pub fn received_generation_commands(&self) -> Arc<Mutex<Vec<ChatGenerationCommand>>> {
        Arc::clone(&self.received_generation_commands)
    }

    pub fn unavailable() -> Self {
        let mut executor = Self::ready(Vec::new());
        executor.health_snapshot =
            WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable);
        executor
    }
}

impl ChatGenerationExecutor for ScriptedExecutor {
    fn start_chat_generation(
        &self,
        generation_command: ChatGenerationCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<ChatGenerationStreamEvent>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            self.received_generation_commands
                .lock()
                .expect("the scripted executor command log should not be poisoned")
                .push(generation_command);
            let (stream_event_sender, stream_event_receiver) =
                mpsc::channel(self.stream_events.len().max(1));
            for stream_event in &self.stream_events {
                stream_event_sender
                    .send(stream_event.clone())
                    .await
                    .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            }
            Ok(stream_event_receiver)
        })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        let mut worker_health_snapshot = self.health_snapshot.clone();
        if self.is_busy_override {
            worker_health_snapshot.activity = WorkerActivity::Generating;
        }
        worker_health_snapshot
    }
}
