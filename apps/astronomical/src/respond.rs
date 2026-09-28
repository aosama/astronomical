//! The one-shot `respond` journey: prompt in, streamed answer out, exit.
//!
//! The CLI never touches the REST surface. It speaks the framed daemon IPC
//! protocol over the instance's unix socket: the shared probe establishes
//! which model is resident, then one connection carries the streamed
//! generation.

use std::io::Write;

use astronomical_ipc_protocol::{
    ChatGenerationSettings, ChatMessage, DaemonIpcClient, DaemonRequest, DaemonResponse,
};

use crate::{DaemonProbe, RespondArguments, errors::RespondError};

/// Collaborators the respond journey needs, injected so tests can stub them.
pub struct RespondDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<std::path::PathBuf>,
    /// Where the answer payload goes.
    pub stdout: &'a mut dyn Write,
    /// Where progress, reasoning, and errors go.
    pub stderr: &'a mut dyn Write,
    /// Bound for each protocol stage of the journey.
    pub request_timeout: std::time::Duration,
}

/// Runs the whole respond journey against the resident daemon.
pub async fn run_respond(
    respond_arguments: &RespondArguments,
    respond_dependencies: &mut RespondDependencies<'_>,
) -> Result<(), RespondError> {
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: respond_dependencies.candidate_socket_paths.clone(),
        request_timeout: respond_dependencies.request_timeout,
    };
    let ready_model_id = match daemon_probe.ready_model_id().await? {
        Some(ready_model_id) => ready_model_id,
        None => return Err(RespondError::NoModelLoaded),
    };
    let chat_model_id = match &respond_arguments.model_id {
        Some(requested_model_id) if requested_model_id != &ready_model_id => {
            return Err(RespondError::RequestedModelNotReady {
                requested_model_id: requested_model_id.clone(),
                ready_model_id,
            });
        }
        _ => ready_model_id,
    };
    stream_answer(
        respond_arguments,
        &daemon_probe,
        respond_dependencies,
        &chat_model_id,
    )
    .await
}

/// Streams one chat generation, rendering frames until a terminal frame.
async fn stream_answer(
    respond_arguments: &RespondArguments,
    daemon_probe: &DaemonProbe,
    respond_dependencies: &mut RespondDependencies<'_>,
    chat_model_id: &str,
) -> Result<(), RespondError> {
    let mut daemon_client = daemon_probe.connect().await?;
    let chat_generate_request = DaemonRequest::ChatGenerate {
        model: chat_model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: respond_arguments.prompt.clone(),
            images: vec![],
        }],
        // Zero is the sentinel for "no CLI opinion": the daemon fills the
        // policy default, then the worker-advertised capability limit.
        settings: ChatGenerationSettings {
            max_output_tokens: 0,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    };
    tokio::time::timeout(
        respond_dependencies.request_timeout,
        daemon_client.send_request(&chat_generate_request),
    )
    .await
    .map_err(|_elapsed| RespondError::DaemonStoppedResponding)?
    .map_err(|_transport_error| RespondError::DaemonStoppedResponding)?;

    let mut buffered_answer = String::new();
    let generation_outcome = tokio::time::timeout(
        respond_dependencies.request_timeout,
        relay_generation_frames(
            respond_arguments,
            respond_dependencies,
            &mut daemon_client,
            &mut buffered_answer,
        ),
    )
    .await;
    match generation_outcome {
        Ok(journey_result) => journey_result,
        Err(_elapsed) => Err(RespondError::DaemonStoppedResponding),
    }
}

/// Relays frames until a terminal frame; writes the answer per stream mode.
async fn relay_generation_frames(
    respond_arguments: &RespondArguments,
    respond_dependencies: &mut RespondDependencies<'_>,
    daemon_client: &mut DaemonIpcClient,
    buffered_answer: &mut String,
) -> Result<(), RespondError> {
    while let Some(daemon_response) = daemon_client
        .next_response()
        .await
        .map_err(|_transport_error| RespondError::DaemonStoppedResponding)?
    {
        match daemon_response {
            DaemonResponse::ChatGenerationText { text } => {
                if respond_arguments.no_stream {
                    buffered_answer.push_str(&text);
                } else {
                    write_and_flush(respond_dependencies.stdout, &text)?;
                }
            }
            DaemonResponse::ChatGenerationReasoning { text } => {
                write_and_flush(respond_dependencies.stderr, &text)?;
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
            DaemonResponse::ChatGenerationFailed { .. } => {
                return Err(RespondError::GenerationFailed);
            }
            DaemonResponse::GenerationRejected { reason } => {
                return Err(RespondError::GenerationRejected { reason });
            }
            // Handshake, status, and embeddings frames are protocol
            // violations in the middle of a generation stream.
            _ => return Err(RespondError::DaemonStoppedResponding),
        }
    }
    // EOF before a terminal frame: the daemon went away mid-answer.
    Err(RespondError::DaemonStoppedResponding)
}

fn write_and_flush(output: &mut dyn Write, text: &str) -> Result<(), RespondError> {
    output
        .write_all(text.as_bytes())
        .and_then(|()| output.flush())
        .map_err(|output_error| RespondError::StdoutUnwritable {
            cause: output_error.to_string(),
        })
}
