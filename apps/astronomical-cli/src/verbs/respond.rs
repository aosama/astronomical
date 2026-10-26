//! The one-shot `respond` journey: prompt in, streamed answer out, exit.
//!
//! The CLI never touches the REST surface. It speaks the framed daemon IPC
//! protocol over the instance's unix socket: the shared model lifecycle
//! resolves which model to use (flag, daemon default, or built-in) and
//! downloads it when the Mac lacks it, then one connection carries the
//! streamed generation — the daemon loads or swaps the resident model
//! itself.

use std::{io::Write, time::Duration};

use astronomical_config::AstronomicalRuntimeInstance;

use astronomical_ipc_protocol::{
    ChatGenerationFailureReason, ChatGenerationSettings, ChatImageInput, ChatMessage,
    DaemonIpcClient, DaemonRequest, DaemonResponse,
};

use crate::transport::ipc::DaemonProbeError;
use std::path::PathBuf;
use thiserror::Error;

mod image;
mod schema;

use crate::{DaemonProbe, ModelLifecycle, RequiredCapability};

use self::{image::read_image_inputs, schema::read_schema_input};

/// Collaborators the respond journey needs, injected so tests can stub them.
pub struct RespondDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<std::path::PathBuf>,
    /// Channel these sockets belong to: carried into the not-running
    /// disclosure so the failure names whose daemon is down.
    pub runtime_instance: AstronomicalRuntimeInstance,
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
        runtime_instance: respond_dependencies.runtime_instance,
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
            repetition_penalty_thousandths: None,
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

/// Failures of the one-shot `respond` journey after arguments have parsed.
/// These exit 1, except `ModelUnavailable`, which exits 2 as a usage error.
#[derive(Error)]
pub enum RespondError {
    #[error("{disclosure}")]
    DaemonNotRunning { disclosure: String },
    #[error("The Astronomical worker is not ready yet — retry in a moment.")]
    WorkerNotReady,
    #[error("{reason}")]
    ModelUnavailable { reason: String },
    #[error("the model download failed: {reason}")]
    DownloadFailed { reason: String },
    #[error("The daemon declined the request: {reason}")]
    GenerationRejected { reason: String },
    #[error("The model failed to finish the response: {reason}")]
    GenerationFailed { reason: String },
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("Could not write the answer to standard output: {cause}")]
    StdoutUnwritable { cause: String },
    #[error("could not read image {path}: {cause}")]
    ImageReadFailed { path: PathBuf, cause: String },
    #[error("{path} is not a supported image; supported formats: {supported}")]
    UnsupportedImage { path: PathBuf, supported: String },
    #[error(
        "the combined image size is {actual_bytes} bytes, over the {maximum_bytes}-byte limit; \
         send smaller or fewer --image files"
    )]
    ImageTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("could not read schema {path}: {cause}")]
    SchemaReadFailed { path: PathBuf, cause: String },
    #[error("{path} is not valid UTF-8 text: {cause}")]
    SchemaNotUtf8 { path: PathBuf, cause: String },
    #[error(
        "the schema file is {actual_bytes} bytes, over the {maximum_bytes}-byte limit; \
         send a smaller --schema file"
    )]
    SchemaTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
}

impl From<DaemonProbeError> for RespondError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning { disclosure } => {
                RespondError::DaemonNotRunning { disclosure }
            }
            DaemonProbeError::DaemonStoppedResponding => RespondError::DaemonStoppedResponding,
            DaemonProbeError::DaemonRejected { reason } => RespondError::DownloadFailed { reason },
        }
    }
}

impl From<crate::model_lifecycle::ModelLifecycleError> for RespondError {
    fn from(lifecycle_error: crate::model_lifecycle::ModelLifecycleError) -> Self {
        match lifecycle_error {
            crate::model_lifecycle::ModelLifecycleError::DaemonNotRunning { disclosure } => {
                RespondError::DaemonNotRunning { disclosure }
            }
            crate::model_lifecycle::ModelLifecycleError::WorkerNotReady => {
                RespondError::WorkerNotReady
            }
            crate::model_lifecycle::ModelLifecycleError::DaemonStoppedResponding => {
                RespondError::DaemonStoppedResponding
            }
            crate::model_lifecycle::ModelLifecycleError::ModelUnavailable { reason } => {
                RespondError::ModelUnavailable { reason }
            }
            crate::model_lifecycle::ModelLifecycleError::DownloadFailed { reason } => {
                RespondError::DownloadFailed { reason }
            }
        }
    }
}

// The debug form must carry the same user-facing guidance as the display
/// form: journeys and tests report whichever rendering they captured.
impl std::fmt::Debug for RespondError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

// Arguments for the ephemeral one-shot `astronomical respond` verb.

/// Parsed `astronomical respond` invocation before any IPC work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RespondArguments {
    /// The user's prompt, exactly as supplied on the command line.
    pub prompt: String,
    /// Raster images to attach to the prompt, in the order the user supplied them.
    pub images: Vec<PathBuf>,
    /// Exact model identity to demand from the resident daemon, when given.
    pub model_id: Option<String>,
    /// System-prompt-style guidance applied to the reply, when given.
    pub instructions: Option<String>,
    /// Cap on the tokens a thinking model may spend reasoning, when given.
    /// `None` lets the model think freely up to the output budget.
    pub thinking_budget: Option<u16>,
    /// Path to the JSON schema file the reply must satisfy, when given.
    pub schema_path: Option<PathBuf>,
    /// Buffer the answer and print it once instead of streaming fragments.
    pub no_stream: bool,
}
