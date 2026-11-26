use std::{error::Error, process::ExitCode, time::Duration};

use ipc_protocol::{
    ChatGenerationCompletionReason, ChatModelCapabilities, ExpertMemoryMode, ProtocolReader,
    ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};

mod scripted_worker_cancellation;
mod scripted_worker_chat;
mod scripted_worker_decisions;
mod scripted_worker_image;
mod scripted_worker_output_fixtures;
mod scripted_worker_single_shot;
mod scripted_worker_speech;

use scripted_worker_cancellation::{
    CancellationMlxTelemetry, EXPERT_STREAMING_TELEMETRY_FIXTURE_MODEL_ID,
    cancellation_memory_snapshot, emit_cancellation_telemetry,
};
use scripted_worker_chat::send_completion;
use scripted_worker_image::{
    ScriptedImageCommandOutcome, handle_image_command, image_capabilities, send_failed_image,
};
use scripted_worker_output_fixtures::send_scripted_chat_output;

const READY_MODEL_ID_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_TEST_WORKER_READY_MODEL_ID";
const DEFAULT_READY_MODEL_ID: &str = "astronomical/test-worker";

#[tokio::main]
async fn main() -> ExitCode {
    match run_fixture().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(fixture_error) => {
            eprintln!("test worker failed: {fixture_error}");
            ExitCode::FAILURE
        }
    }
}

/// The cancellation script the Generate arms set and the Cancel arm plays
/// back, plus the session fixture's request counter; one bundle keeps the
/// handler signatures readable.
struct ScriptedWorkerState {
    active_request_id: Option<RequestId>,
    cancellation_acknowledgement_delay: Duration,
    should_acknowledge_cancellation: bool,
    should_emit_unexpected_cancellation_event: bool,
    should_emit_astro_ssd_prompt_cache_stats_before_cancellation_acknowledgement: bool,
    should_emit_expert_streaming_stats_before_cancellation_acknowledgement: bool,
    expert_streaming_session_request_count: u64,
    cancellation_mlx_telemetry: CancellationMlxTelemetry,
}

impl ScriptedWorkerState {
    fn new() -> Self {
        ScriptedWorkerState {
            active_request_id: None,
            cancellation_acknowledgement_delay: Duration::ZERO,
            should_acknowledge_cancellation: true,
            should_emit_unexpected_cancellation_event: false,
            should_emit_astro_ssd_prompt_cache_stats_before_cancellation_acknowledgement: false,
            should_emit_expert_streaming_stats_before_cancellation_acknowledgement: false,
            expert_streaming_session_request_count: 0,
            cancellation_mlx_telemetry: CancellationMlxTelemetry::None,
        }
    }

    /// Makes `request_id` the generation a later Cancel stops, answered or
    /// left unanswered as the fixture scripts.
    fn begin_cancellable_request(
        &mut self,
        request_id: RequestId,
        should_acknowledge_cancellation: bool,
    ) {
        self.active_request_id = Some(request_id);
        self.should_acknowledge_cancellation = should_acknowledge_cancellation;
    }
}

async fn run_fixture() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut command_reader = ProtocolReader::new(tokio::io::stdin());
    let mut event_writer = ProtocolWriter::new(tokio::io::stdout());
    let mut worker_state = ScriptedWorkerState::new();
    send_ready_event(&mut event_writer).await?;

    while let Some(worker_command) = command_reader.next_command().await? {
        let should_exit_worker =
            answer_worker_command(&mut event_writer, &mut worker_state, worker_command).await?;
        if should_exit_worker {
            return Ok(());
        }
    }
    Ok(())
}

async fn send_ready_event(
    event_writer: &mut ProtocolWriter<tokio::io::Stdout>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let ready_model_id = std::env::var(READY_MODEL_ID_ENVIRONMENT_VARIABLE)
        .unwrap_or_else(|_| DEFAULT_READY_MODEL_ID.to_owned());
    event_writer
        .send_event(&WorkerEvent::Ready {
            model_id: ready_model_id,
            capabilities: ipc_protocol::WorkerModelCapabilities::chat_and_image(
                ChatModelCapabilities {
                    supports_reasoning: true,
                    supports_tool_calls: true,
                    has_vision: true,
                    max_input_tokens: 241_664,
                    max_output_tokens: 20_480,
                    context_window: 262_144,
                },
                image_capabilities(),
            ),
        })
        .await?;
    Ok(())
}

/// Answers one worker command; returns true when the worker should exit.
async fn answer_worker_command(
    event_writer: &mut ProtocolWriter<tokio::io::Stdout>,
    worker_state: &mut ScriptedWorkerState,
    worker_command: WorkerCommand,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    match worker_command {
        WorkerCommand::InitializeWorker(_) => {}
        single_shot_command @ (WorkerCommand::GenerateDecisions(_)
        | WorkerCommand::TrainDecisionModel(_)
        | WorkerCommand::GenerateEmbeddings(_)
        | WorkerCommand::GenerateSpeech(_)) => {
            scripted_worker_single_shot::answer_single_shot_command(
                event_writer,
                single_shot_command,
            )
            .await?;
        }
        WorkerCommand::CountChatTokens(count_command) => {
            answer_count_chat_tokens(event_writer, count_command).await?;
        }
        WorkerCommand::GenerateImage(generation_command) => {
            let request_id = generation_command.request_id;
            if let ScriptedImageCommandOutcome::CancellationPending { should_acknowledge } =
                handle_image_command(generation_command, event_writer).await?
            {
                worker_state.begin_cancellable_request(request_id, should_acknowledge);
            }
        }
        WorkerCommand::Generate(generation_command) => {
            return answer_generation(event_writer, worker_state, generation_command).await;
        }
        WorkerCommand::Cancel { request_id } => {
            answer_cancellation(event_writer, worker_state, request_id).await?;
        }
        WorkerCommand::SwapModel {
            model_directory, ..
        } => {
            tracing::warn!(
                model_directory,
                "test scripted worker received SwapModel; ignoring"
            );
        }
        WorkerCommand::SampleMlxMemory => {}
        WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes,
            configuration_generation: _,
        } => {
            answer_mlx_memory_limit_update(effective_mlx_memory_ceiling_bytes, event_writer)
                .await?;
        }
        WorkerCommand::ClearAstroSsdPromptCache { model_id } => {
            event_writer
                .send_event(&WorkerEvent::AstroSsdPromptCacheCleared {
                    model_id,
                    blocks_removed: 0,
                    bytes_freed: 0,
                })
                .await?;
        }
    }
    Ok(false)
}

async fn answer_count_chat_tokens(
    event_writer: &mut ProtocolWriter<tokio::io::Stdout>,
    count_command: ipc_protocol::ChatGenerationCommand,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Seven tokens per message: distinct from every `Generate`
    // reply, so a journey can prove the daemon sent a count.
    let message_count = u32::try_from(count_command.messages.len()).unwrap_or(u32::MAX);
    send_completion(
        count_command.request_id,
        message_count.saturating_mul(7),
        0,
        ChatGenerationCompletionReason::EndOfSequence,
        event_writer,
    )
    .await?;
    Ok(())
}

/// Answers a chat generation; returns true when the worker should exit.
async fn answer_generation(
    event_writer: &mut ProtocolWriter<tokio::io::Stdout>,
    worker_state: &mut ScriptedWorkerState,
    generation_command: ipc_protocol::ChatGenerationCommand,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let request_id = generation_command.request_id;
    if generation_command.structured_generation.is_some() {
        scripted_worker_chat::send_structured_json_reply(request_id, event_writer).await?;
        return Ok(false);
    }
    dispatch_unstructured_chat(generation_command, request_id, event_writer, worker_state).await
}

async fn answer_cancellation(
    event_writer: &mut ProtocolWriter<tokio::io::Stdout>,
    worker_state: &mut ScriptedWorkerState,
    request_id: RequestId,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if worker_state.active_request_id == Some(request_id) {
        worker_state.active_request_id = None;
    }
    if !worker_state.should_acknowledge_cancellation {
        return Ok(());
    }
    tokio::time::sleep(worker_state.cancellation_acknowledgement_delay).await;
    worker_state.cancellation_acknowledgement_delay = Duration::ZERO;
    if worker_state.should_emit_unexpected_cancellation_event {
        worker_state.should_emit_unexpected_cancellation_event = false;
        send_completion(
            RequestId::new(2),
            1,
            0,
            ChatGenerationCompletionReason::Cancelled,
            event_writer,
        )
        .await?;
        return Ok(());
    }
    // Telemetry a real worker can still flush while a cancelled
    // generation winds down must pass through the drain
    // without failing it; the helper fires each flag once.
    emit_cancellation_telemetry(
        event_writer,
        &mut worker_state
            .should_emit_astro_ssd_prompt_cache_stats_before_cancellation_acknowledgement,
        &mut worker_state.should_emit_expert_streaming_stats_before_cancellation_acknowledgement,
        &mut worker_state.cancellation_mlx_telemetry,
    )
    .await?;
    if worker_state.should_acknowledge_cancellation && request_id.value() >= 100 {
        send_failed_image(request_id, event_writer).await?;
        return Ok(());
    }
    send_completion(
        request_id,
        1,
        0,
        ChatGenerationCompletionReason::Cancelled,
        event_writer,
    )
    .await?;
    Ok(())
}

/// Both fixtures answer with an accepted chat; the session fixture also
/// publishes the engine-load counters accumulated through this request, so
/// the daemon-side observatory must show session totals that keep growing
/// across replies. Sharing one dispatch arm keeps the dispatcher's
/// complexity flat.
async fn send_accepted_or_expert_streaming_session_chat(
    model_id: &str,
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
    expert_streaming_session_request_count: &mut u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if model_id != "astronomical/expert-streaming-session-fixture" {
        scripted_worker_chat::send_accepted_chat(request_id, event_writer).await?;
        return Ok(());
    }
    *expert_streaming_session_request_count += 1;
    scripted_worker_chat::send_expert_streaming_session_chat(
        request_id,
        event_writer,
        *expert_streaming_session_request_count,
    )
    .await?;
    Ok(())
}

/// Dispatches the schema-free chat fixture arms: the accepted-chat pair, the
/// ones that script the shared cancellation state, and the exit fixture are
/// handled here, and every other model goes to the event-sequence fixtures.
/// Returns true when the worker should exit.
async fn dispatch_unstructured_chat(
    generation_command: ipc_protocol::ChatGenerationCommand,
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
    worker_state: &mut ScriptedWorkerState,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let model_id = generation_command.model.as_str();
    match model_id {
        "astronomical/accepted-chat-fixture" | "astronomical/expert-streaming-session-fixture" => {
            send_accepted_or_expert_streaming_session_chat(
                model_id,
                request_id,
                event_writer,
                &mut worker_state.expert_streaming_session_request_count,
            )
            .await?;
        }
        "astronomical/deadline-expiring-chat-fixture"
        | "astronomical/delayed-fragment-chat-fixture" => {
            worker_state.begin_cancellable_request(request_id, true);
        }
        "astronomical/unacknowledged-cancellation-fixture" => {
            worker_state.begin_cancellable_request(request_id, false);
        }
        "astronomical/delayed-cancellation-acknowledgement-fixture" => {
            worker_state.begin_cancellable_request(request_id, true);
            worker_state.cancellation_acknowledgement_delay = Duration::from_secs(4);
        }
        "astronomical/unexpected-cancellation-event-fixture" => {
            worker_state.begin_cancellable_request(request_id, true);
            worker_state.should_emit_unexpected_cancellation_event = true;
        }
        "astronomical/astro-ssd-prompt-cache-stats-during-cancellation-fixture"
        | EXPERT_STREAMING_TELEMETRY_FIXTURE_MODEL_ID => {
            worker_state.begin_cancellable_request(request_id, true);
            // Both fixtures drain exactly one telemetry event before the
            // acknowledgement; only the published kind differs.
            let emits_expert_streaming_stats =
                model_id == EXPERT_STREAMING_TELEMETRY_FIXTURE_MODEL_ID;
            worker_state
                .should_emit_astro_ssd_prompt_cache_stats_before_cancellation_acknowledgement =
                !emits_expert_streaming_stats;
            worker_state.should_emit_expert_streaming_stats_before_cancellation_acknowledgement =
                emits_expert_streaming_stats;
        }
        "astronomical/mlx-memory-during-cancellation-fixture" => {
            worker_state.begin_cancellable_request(request_id, true);
            worker_state.cancellation_mlx_telemetry = CancellationMlxTelemetry::Publish;
        }
        "astronomical/mlx-memory-clear-during-cancellation-fixture" => {
            worker_state.begin_cancellable_request(request_id, true);
            // Seed a visible snapshot so the cancellation event can
            // prove that `None` clears previously published memory.
            event_writer
                .send_event(&WorkerEvent::MlxMemorySample {
                    mlx_memory_snapshot: Some(cancellation_memory_snapshot(33_000)),
                    expert_residency: None,
                })
                .await?;
            worker_state.cancellation_mlx_telemetry = CancellationMlxTelemetry::Clear;
        }
        "astronomical/exit-after-chat-admission-fixture" => return Ok(true),
        _ => {
            send_scripted_chat_output(
                model_id,
                request_id,
                generation_command.settings.max_output_tokens,
                event_writer,
            )
            .await?;
        }
    }
    Ok(false)
}

/// A ceiling of this many bytes makes the fixture exit the way a real worker
/// does when an engine fails fatally while adopting a new ceiling (#1900).
/// The integration test that drives it repeats the value, since this fixture
/// is a separate binary.
const EXIT_ON_MLX_MEMORY_LIMIT_BYTES: u64 = 13;

/// Applies any other ceiling as the scripted worker always has. The exit lives
/// here, not in the command loop, so the loop's branch count stays unchanged.
async fn answer_mlx_memory_limit_update(
    effective_mlx_memory_ceiling_bytes: u64,
    event_writer: &mut ProtocolWriter<tokio::io::Stdout>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if effective_mlx_memory_ceiling_bytes == EXIT_ON_MLX_MEMORY_LIMIT_BYTES {
        eprintln!("model worker stopped with an error: scripted fatal memory-limit failure");
        std::process::exit(1);
    }
    event_writer
        .send_event(&WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes: 1,
            expert_memory_mode: ExpertMemoryMode::Resident,
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    Ok(())
}
