//! Chat generation serving over the daemon IPC: schema validation and
//! instruction pairing, worker-health gating, default filling, and stream
//! relaying from the worker executor to the CLI client.

use std::sync::{Arc, RwLock};

use astronomical_config::AstronomicalInstancePaths;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationFailureReason, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, DaemonResponse, DaemonTransportError, RequestId, StreamingResponseWriter,
};
use tokio::sync::mpsc;

use super::daemon_ipc::{
    DaemonIpcGenerationContext, generation_start_rejection_reason,
    send_terminal_streaming_response, unknown_model_rejection_reason,
};
use crate::{
    ChatGenerationStreamErrorCode, ChatGenerationStreamEvent,
    application::allocate_chat_request_id,
    config_reload::ResolvedRuntimeConfig,
    daemon_ipc_chat_schema::validated_chat_schema_constraint,
    load_configured_qwen_thinking_channel_seed,
    request_generation_defaults::{RequestGenerationSettingsPresence, apply_generation_defaults},
    structured_output::{enforced_schema_output_instruction, insert_json_output_instruction},
    supervisor_performance_attribution::SupervisorPerformanceAttributionLog,
    worker_health::{WorkerHealthSnapshot, WorkerHealthStatus},
};

/// One IPC chat request payload, grouped so the streaming plumbing takes the
/// request as a unit instead of four loose fields.
pub(super) struct DaemonIpcChatRequest {
    pub(super) model: String,
    pub(super) messages: Vec<ChatMessage>,
    pub(super) settings: ChatGenerationSettings,
    pub(super) schema_json: Option<String>,
}

/// Runs one IPC chat generation: gates on worker readiness, fills defaults,
/// and streams events to the client until a terminal frame.
pub(super) async fn stream_chat_generation(
    generation_context: &DaemonIpcGenerationContext,
    seed_instance_paths: &AstronomicalInstancePaths,
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
    chat_request: DaemonIpcChatRequest,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let DaemonIpcChatRequest {
        model,
        messages,
        settings,
        schema_json,
    } = chat_request;
    // A malformed schema is a request-shape error, so it rejects before the
    // worker-health gate: the request cannot be served regardless of state.
    let structured_generation_constraint = match schema_json.as_deref() {
        Some(schema_text) => match validated_chat_schema_constraint(schema_text) {
            Ok(validated_constraint) => Some(validated_constraint),
            Err(schema_rejection_reason) => {
                let rejection_response = DaemonResponse::GenerationRejected {
                    reason: schema_rejection_reason,
                };
                return send_terminal_streaming_response(
                    streaming_response_writer,
                    &rejection_response,
                )
                .await;
            }
        },
        None => None,
    };
    // The token mask clamps the visible channel but never tells the model
    // what shape to plan, so the enforced schema constraint pairs with a
    // system instruction (observed live: masking alone made the model answer
    // `{"error": "Invalid JSON"}` once the brace was forced). The validated
    // constraint above exists exactly when the schema text does, so the
    // schema text drives the injection.
    let mut messages = messages;
    if let Some(schema_text) = schema_json.as_deref() {
        insert_json_output_instruction(
            &mut messages,
            enforced_schema_output_instruction(schema_text),
        );
    }
    let health_snapshot = generation_context.executor.worker_health_snapshot();
    if let Some(rejection_reason) = ipc_generation_rejection_reason(
        &health_snapshot,
        &generation_context.reloadable_config,
        &model,
    ) {
        let rejection_response = DaemonResponse::GenerationRejected {
            reason: rejection_reason,
        };
        return send_terminal_streaming_response(streaming_response_writer, &rejection_response)
            .await;
    }
    let Some(request_id_raw) = allocate_chat_request_id(&generation_context.next_chat_request_id)
    else {
        let rejection_response = DaemonResponse::GenerationRejected {
            reason: "the local request identifier space is exhausted".to_owned(),
        };
        return send_terminal_streaming_response(streaming_response_writer, &rejection_response)
            .await;
    };
    let mut settings = settings;
    apply_ipc_generation_defaults(generation_context, &health_snapshot, &model, &mut settings);
    let generation_command = ChatGenerationCommand {
        request_id: RequestId::new(request_id_raw),
        model: model.clone(),
        messages,
        tools: vec![],
        tool_choice: ChatToolChoice::Auto,
        settings,
        qwen_thinking_channel_seed: load_configured_qwen_thinking_channel_seed(
            generation_context.reloadable_config.as_ref(),
            Some(seed_instance_paths),
            supervisor_attribution_log,
            &model,
        )
        .await,
        structured_generation: structured_generation_constraint,
    };
    let stream_event_receiver = match generation_context
        .executor
        .start_chat_generation(generation_command)
        .await
    {
        Ok(stream_event_receiver) => stream_event_receiver,
        Err(start_error) => {
            let rejection_response = DaemonResponse::GenerationRejected {
                reason: generation_start_rejection_reason(start_error),
            };
            return send_terminal_streaming_response(
                streaming_response_writer,
                &rejection_response,
            )
            .await;
        }
    };
    relay_stream_events(stream_event_receiver, streaming_response_writer).await
}

/// Rejects a generation the daemon cannot serve: an unavailable worker, or a
/// requested model that is unknown to the live model policy catalog (checked
/// as either a full catalog key or a leaf alias). A known model that is not
/// resident is admitted on purpose: the worker loop swaps or loads it on
/// demand, which is how the CLI auto-loads from a cold daemon.
fn ipc_generation_rejection_reason(
    health_snapshot: &WorkerHealthSnapshot,
    reloadable_config: &Option<Arc<RwLock<ResolvedRuntimeConfig>>>,
    requested_model_id: &str,
) -> Option<String> {
    if health_snapshot.status == WorkerHealthStatus::Unavailable {
        return Some("the daemon worker is not ready to serve chat generation".to_owned());
    }
    let Some(reloadable_config) = reloadable_config else {
        // No policy catalog wired: the worker loop applies its own unknown-model guard.
        return None;
    };
    let Ok(live_config) = reloadable_config.read() else {
        return Some("the daemon configuration is temporarily unavailable".to_owned());
    };
    let known_model_ids: Vec<&str> = live_config
        .model_policy_catalog
        .keys()
        .map(String::as_str)
        .collect();
    let requested_is_known = known_model_ids.contains(&requested_model_id)
        || known_model_ids.contains(&astronomical_config::leaf_model_id(requested_model_id));
    if !requested_is_known {
        let suggested_model_ids =
            astronomical_config::near_model_matches(requested_model_id, &known_model_ids);
        return Some(unknown_model_rejection_reason(
            requested_model_id,
            &suggested_model_ids,
            "the daemon knows no such model; run `astronomical models list` to see installed models or `astronomical models supported` to see downloadable ones",
        ));
    }
    None
}
fn apply_ipc_generation_defaults(
    generation_context: &DaemonIpcGenerationContext,
    health_snapshot: &WorkerHealthSnapshot,
    model_id: &str,
    generation_settings: &mut ChatGenerationSettings,
) {
    let settings_presence = RequestGenerationSettingsPresence {
        maximum_output_tokens: generation_settings.max_output_tokens != 0,
        temperature: generation_settings.temperature_thousandths.is_some(),
        top_p: generation_settings.top_p_thousandths.is_some(),
    };
    apply_generation_defaults(
        generation_context.reloadable_config.as_ref(),
        model_id,
        settings_presence,
        generation_settings,
    );
    if generation_settings.max_output_tokens == 0
        && let Some(ready_model_capabilities) = &health_snapshot.ready_model_capabilities
        && let Some(chat_capabilities) = &ready_model_capabilities.chat
    {
        // The worker advertises u32 token counts; the wire settings field is u16.
        generation_settings.max_output_tokens =
            u16::try_from(chat_capabilities.max_output_tokens).unwrap_or(u16::MAX);
    }
}
async fn relay_stream_events(
    mut stream_event_receiver: mpsc::Receiver<ChatGenerationStreamEvent>,
    mut streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    while let Some(stream_event) = stream_event_receiver.recv().await {
        let Some(daemon_response) = stream_event_to_daemon_response(stream_event) else {
            // PrefillProgress is worker-internal progress with no CLI presentation.
            continue;
        };
        streaming_response_writer
            .send_response(&daemon_response)
            .await?;
        let is_terminal_frame = matches!(
            daemon_response,
            DaemonResponse::ChatGenerationCompleted { .. }
                | DaemonResponse::ChatGenerationFailed { .. }
                | DaemonResponse::GenerationRejected { .. }
        );
        if is_terminal_frame {
            streaming_response_writer.close().await?;
            return Ok(());
        }
    }
    let eof_failure_response = DaemonResponse::ChatGenerationFailed {
        reason: ChatGenerationFailureReason::FatalExecution {
            reason: "the worker stream ended before completing the generation".to_owned(),
        },
    };
    send_terminal_streaming_response(streaming_response_writer, &eof_failure_response).await
}

fn stream_event_to_daemon_response(
    stream_event: ChatGenerationStreamEvent,
) -> Option<DaemonResponse> {
    match stream_event {
        ChatGenerationStreamEvent::TextFragment(text) => {
            Some(DaemonResponse::ChatGenerationText { text })
        }
        ChatGenerationStreamEvent::ReasoningFragment(text) => {
            Some(DaemonResponse::ChatGenerationReasoning { text })
        }
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index,
            function_name,
            arguments_json,
        } => Some(DaemonResponse::ChatGenerationToolCall {
            tool_call_index,
            function_name,
            arguments_json,
        }),
        ChatGenerationStreamEvent::PrefillProgress { .. } => None,
        ChatGenerationStreamEvent::Completed {
            prompt_token_count,
            generated_token_count,
            reasoning_token_count,
            cached_token_count,
            reason,
        } => Some(DaemonResponse::ChatGenerationCompleted {
            prompt_token_count,
            generated_token_count,
            reasoning_token_count,
            cached_token_count,
            reason,
        }),
        ChatGenerationStreamEvent::Failed { reason } => {
            Some(DaemonResponse::ChatGenerationFailed { reason })
        }
        ChatGenerationStreamEvent::Error(stream_error_code) => {
            Some(DaemonResponse::ChatGenerationFailed {
                reason: ChatGenerationFailureReason::FatalExecution {
                    reason: stream_error_code_to_failure_reason(stream_error_code),
                },
            })
        }
    }
}

fn stream_error_code_to_failure_reason(stream_error_code: ChatGenerationStreamErrorCode) -> String {
    match stream_error_code {
        ChatGenerationStreamErrorCode::WorkerUnavailable => {
            "the worker became unavailable during the generation".to_owned()
        }
    }
}
