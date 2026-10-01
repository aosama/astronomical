//! The one-shot `respond` journey: prompt in, streamed answer out, exit.
//!
//! The CLI never touches the REST surface. It speaks the framed daemon IPC
//! protocol over the instance's unix socket: the shared model lifecycle
//! resolves which model to use (flag, daemon default, or built-in) and
//! downloads it when the Mac lacks it, then one connection carries the
//! streamed generation — the daemon loads or swaps the resident model
//! itself.

use std::{io::Write, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationFailureReason, ChatGenerationSettings, ChatImageInput, ChatMessage,
    DaemonIpcClient, DaemonRequest, DaemonResponse,
};

use crate::{
    DaemonProbe, ModelLifecycle, RequiredCapability, RespondArguments, errors::RespondError,
    respond_image::read_image_inputs, respond_schema::read_schema_input,
};

/// Collaborators the respond journey needs, injected so tests can stub them.
pub struct RespondDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<std::path::PathBuf>,
    /// Where the answer payload goes.
    pub stdout: &'a mut dyn Write,
    /// Where progress, reasoning, and errors go.
    pub stderr: &'a mut dyn Write,
    /// Bound for each protocol stage of the journey.
    pub request_timeout: Duration,
    /// Bound for the whole download-wait stage, if the model must download.
    pub download_stage_bound: Duration,
    /// Wait between download status polls.
    pub download_poll_interval: Duration,
}

/// Runs the whole respond journey against the resident daemon: resolve the
/// model (flag, daemon default, or built-in), let the lifecycle download it
/// when the Mac does not have it yet, then stream the answer.
pub async fn run_respond(
    respond_arguments: &RespondArguments,
    respond_dependencies: &mut RespondDependencies<'_>,
) -> Result<(), RespondError> {
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: respond_dependencies.candidate_socket_paths.clone(),
        request_timeout: respond_dependencies.request_timeout,
    };
    let model_lifecycle = ModelLifecycle {
        daemon_probe,
        download_stage_bound: respond_dependencies.download_stage_bound,
        download_poll_interval: respond_dependencies.download_poll_interval,
    };
    // Local inputs fail before any daemon work: a missing schema file or an
    // unreadable image must never start a model download.
    let schema_json = match &respond_arguments.schema_path {
        Some(schema_path) => Some(read_schema_input(schema_path)?),
        None => None,
    };
    let images = read_image_inputs(&respond_arguments.images)?;
    let mut progress_started = false;
    let chat_model_id = model_lifecycle
        .prepare_model_id(
            respond_arguments.model_id.as_deref(),
            RequiredCapability::Chat,
            &mut |progress_line| {
                progress_started = true;
                let _ = write!(respond_dependencies.stderr, "\r{progress_line}");
                let _ = respond_dependencies.stderr.flush();
            },
        )
        .await?;
    if progress_started {
        // Finalize the live progress line before the answer owns stderr.
        let _ = writeln!(respond_dependencies.stderr);
    }
    stream_answer(
        respond_arguments,
        &model_lifecycle.daemon_probe,
        respond_dependencies,
        &chat_model_id,
        schema_json,
        images,
    )
    .await
}

/// Streams one chat generation, rendering frames until a terminal frame.
async fn stream_answer(
    respond_arguments: &RespondArguments,
    daemon_probe: &DaemonProbe,
    respond_dependencies: &mut RespondDependencies<'_>,
    chat_model_id: &str,
    schema_json: Option<String>,
    images: Vec<ChatImageInput>,
) -> Result<(), RespondError> {
    let mut daemon_client = daemon_probe.connect().await?;
    // The instructions, when given, become the initial system message the
    // worker renders before the user's prompt; the daemon forwards messages
    // to the worker unchanged.
    let mut messages = Vec::new();
    if let Some(instructions) = &respond_arguments.instructions {
        messages.push(ChatMessage::System {
            content: instructions.clone(),
        });
    }
    messages.push(ChatMessage::User {
        content: respond_arguments.prompt.clone(),
        images,
    });
    let chat_generate_request = DaemonRequest::ChatGenerate {
        model: chat_model_id.to_owned(),
        messages,
        // Zero is the sentinel for "no CLI opinion": the daemon fills the
        // policy default, then the worker-advertised capability limit.
        settings: ChatGenerationSettings {
            max_output_tokens: 0,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: respond_arguments.thinking_budget,
        },
        schema_json,
    };
    tokio::time::timeout(
        respond_dependencies.request_timeout,
        daemon_client.send_request(&chat_generate_request),
    )
    .await
    .map_err(|_elapsed| RespondError::DaemonStoppedResponding)?
    .map_err(|_transport_error| RespondError::DaemonStoppedResponding)?;

    let mut buffered_answer = String::new();
    relay_generation_frames(
        respond_arguments,
        respond_dependencies,
        &mut daemon_client,
        &mut buffered_answer,
    )
    .await
}

/// Relays frames until a terminal frame; writes the answer per stream mode.
async fn relay_generation_frames(
    respond_arguments: &RespondArguments,
    respond_dependencies: &mut RespondDependencies<'_>,
    daemon_client: &mut DaemonIpcClient,
    buffered_answer: &mut String,
) -> Result<(), RespondError> {
    loop {
        // The timeout is an idle guard per frame, not a cap on the whole
        // generation: a healthy stream may run arbitrarily long, but silence
        // beyond the window means the daemon is gone.
        let next_daemon_response = tokio::time::timeout(
            respond_dependencies.request_timeout,
            daemon_client.next_response(),
        )
        .await
        .map_err(|_elapsed| RespondError::DaemonStoppedResponding)?
        .map_err(|_transport_error| RespondError::DaemonStoppedResponding)?;
        let Some(daemon_response) = next_daemon_response else {
            // EOF before a terminal frame: the daemon went away mid-answer.
            return Err(RespondError::DaemonStoppedResponding);
        };
        match daemon_response {
            DaemonResponse::ChatGenerationText { text } => {
                if respond_arguments.no_stream {
                    buffered_answer.push_str(&text);
                } else {
                    write_and_flush(respond_dependencies.stdout, &text)?;
                }
            }
            DaemonResponse::ChatGenerationReasoning { text } => {
                // Reasoning is progress on stderr, never the payload; a
                // closed stderr must not abort the answer stream.
                let _ = write_and_flush_plain(respond_dependencies.stderr, &text);
            }
            DaemonResponse::ChatGenerationToolCall {
                function_name,
                arguments_json,
                ..
            } => {
                // The daemon never leases tools on this surface, so a tool
                // call is progress, never payload; surface it on stderr.
                let _ = writeln!(
                    respond_dependencies.stderr,
                    "[tool call {function_name} {arguments_json}]"
                );
            }
            DaemonResponse::ChatGenerationCompleted { .. } => {
                if respond_arguments.no_stream {
                    write_and_flush(respond_dependencies.stdout, buffered_answer)?;
                }
                return Ok(());
            }
            DaemonResponse::ChatGenerationFailed { reason } => {
                return Err(RespondError::GenerationFailed {
                    reason: chat_generation_failure_reason_text(&reason),
                });
            }
            DaemonResponse::GenerationRejected { reason } => {
                return Err(RespondError::GenerationRejected { reason });
            }
            // Handshake, status, and embeddings frames are protocol
            // violations in the middle of a generation stream.
            _ => return Err(RespondError::DaemonStoppedResponding),
        }
    }
}

fn write_and_flush(output: &mut dyn Write, text: &str) -> Result<(), RespondError> {
    output
        .write_all(text.as_bytes())
        .and_then(|()| output.flush())
        .map_err(|output_error| RespondError::StdoutUnwritable {
            cause: output_error.to_string(),
        })
}

fn write_and_flush_plain(output: &mut dyn Write, text: &str) -> std::io::Result<()> {
    output
        .write_all(text.as_bytes())
        .and_then(|()| output.flush())
}

/// Renders the worker's bounded failure reason for the CLI user. The protocol
/// contract already bounds these reasons and keeps native details in logs, so
/// the text is safe to surface verbatim; only the context-window case needs
/// assembly from its parts.
fn chat_generation_failure_reason_text(reason: &ChatGenerationFailureReason) -> String {
    match reason {
        ChatGenerationFailureReason::InvalidRequest { reason } => reason.clone(),
        ChatGenerationFailureReason::FatalExecution { reason } => reason.clone(),
        ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens,
            maximum_context_tokens,
        } => format!(
            "the prompt and requested output exceed the model context \
             ({actual_total_context_tokens} of {maximum_context_tokens} tokens)"
        ),
        ChatGenerationFailureReason::EngineBusy => {
            "the engine is busy with another generation".to_owned()
        }
        ChatGenerationFailureReason::MalformedModelOutput => {
            "the model output could not be decoded".to_owned()
        }
    }
}
