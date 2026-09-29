//! The one-shot `astronomical embed` journey: text in, one JSON vector
//! document out, exit.
//!
//! The CLI never touches the REST surface. It speaks the framed daemon IPC
//! protocol over the instance's unix socket: the shared model lifecycle
//! resolves which model to use (flag, daemon default, or built-in) and
//! downloads it when the Mac lacks it, then one connection carries the
//! embeddings batch — the daemon loads or swaps the resident model itself.

use std::{
    io::{Read, Write},
    time::Duration,
};

use astronomical_ipc_protocol::{DaemonRequest, DaemonResponse, EmbeddingsFailureReason};
use serde_json::json;
use tokio::time::timeout;

use crate::{DaemonProbe, EmbedArguments, ModelLifecycle, RequiredCapability, errors::EmbedError};

/// Collaborators the embed journey needs, injected so tests can stub them.
pub struct EmbedDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<std::path::PathBuf>,
    /// Read to EOF when no text or file was supplied.
    pub stdin: &'a mut dyn Read,
    /// Where the JSON vector document goes.
    pub stdout: &'a mut dyn Write,
    /// Where download progress goes.
    pub stderr: &'a mut dyn Write,
    /// Bound for each protocol stage of the journey.
    pub request_timeout: Duration,
    /// Bound for the whole download-wait stage, if the model must download.
    pub download_stage_bound: Duration,
    /// Wait between download status polls.
    pub download_poll_interval: Duration,
}

/// Runs the whole embed journey against the resident daemon: resolve the
/// model (flag, daemon default, or built-in), let the lifecycle download it
/// when the Mac lacks it, then submit the embeddings batch.
pub async fn run_embed(
    embed_arguments: &EmbedArguments,
    embed_dependencies: &mut EmbedDependencies<'_>,
) -> Result<(), EmbedError> {
    let input_text = resolve_input_text(embed_arguments, embed_dependencies.stdin)?;
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: embed_dependencies.candidate_socket_paths.clone(),
        request_timeout: embed_dependencies.request_timeout,
    };
    let model_lifecycle = ModelLifecycle {
        daemon_probe,
        download_stage_bound: embed_dependencies.download_stage_bound,
        download_poll_interval: embed_dependencies.download_poll_interval,
    };
    let mut progress_started = false;
    let embed_model_id = model_lifecycle
        .prepare_model_id(
            embed_arguments.model_id.as_deref(),
            RequiredCapability::Embeddings,
            &mut |progress_line| {
                progress_started = true;
                let _ = write!(embed_dependencies.stderr, "\r{progress_line}");
                let _ = embed_dependencies.stderr.flush();
            },
        )
        .await?;
    if progress_started {
        // Finalize the live progress line before the JSON document owns stdout.
        let _ = writeln!(embed_dependencies.stderr);
    }
    embed_input_text(
        input_text,
        embed_model_id,
        &model_lifecycle.daemon_probe,
        embed_dependencies,
    )
    .await
}

/// Text > file contents > stdin read to EOF. An empty resolved input is a
/// usage failure, not an empty embedding.
fn resolve_input_text(
    embed_arguments: &EmbedArguments,
    stdin: &mut dyn Read,
) -> Result<String, EmbedError> {
    let resolved_input = if let Some(text) = &embed_arguments.text {
        text.clone()
    } else if let Some(file_path) = &embed_arguments.file_path {
        std::fs::read_to_string(file_path).map_err(|read_error| {
            EmbedError::InputFileUnreadable {
                file_path: file_path.clone(),
                cause: read_error.to_string(),
            }
        })?
    } else {
        let mut stdin_text = String::new();
        stdin
            .read_to_string(&mut stdin_text)
            .map_err(|read_error| EmbedError::StdinUnreadable {
                cause: read_error.to_string(),
            })?;
        stdin_text
    };
    if resolved_input.trim().is_empty() {
        return Err(EmbedError::EmbedInputRequired);
    }
    Ok(resolved_input)
}

/// Submits one embeddings batch and writes the single terminal frame as one
/// JSON document on stdout.
async fn embed_input_text(
    input_text: String,
    embed_model_id: String,
    daemon_probe: &DaemonProbe,
    embed_dependencies: &mut EmbedDependencies<'_>,
) -> Result<(), EmbedError> {
    let mut daemon_client = daemon_probe.connect().await?;
    let embed_generate_request = DaemonRequest::EmbedGenerate {
        model: Some(embed_model_id),
        inputs: vec![input_text],
        dimensions: None,
    };
    timeout(
        embed_dependencies.request_timeout,
        daemon_client.send_request(&embed_generate_request),
    )
    .await
    .map_err(|_elapsed| EmbedError::DaemonStoppedResponding)?
    .map_err(|_transport_error| EmbedError::DaemonStoppedResponding)?;

    let embeddings_response = timeout(
        embed_dependencies.request_timeout,
        daemon_client.next_response(),
    )
    .await
    .map_err(|_elapsed| EmbedError::DaemonStoppedResponding)?
    .map_err(|_transport_error| EmbedError::DaemonStoppedResponding)?
    .ok_or(EmbedError::DaemonStoppedResponding)?;
    match embeddings_response {
        DaemonResponse::EmbeddingsCompleted {
            model,
            vectors,
            input_token_counts,
        } => write_vector_document(
            embed_dependencies.stdout,
            &model,
            vectors.first().map(Vec::as_slice).unwrap_or(&[]),
            input_token_counts.first().copied().unwrap_or(0),
        ),
        DaemonResponse::EmbeddingsFailed { reason } => Err(EmbedError::EmbeddingsFailed { reason }),
        DaemonResponse::GenerationRejected { reason } => {
            Err(EmbedError::EmbeddingsRejected { reason })
        }
        // Handshake, status, and chat frames cannot answer an embeddings
        // request; treat the daemon as gone rather than guessing.
        _ => Err(EmbedError::DaemonStoppedResponding),
    }
}

fn write_vector_document(
    stdout: &mut dyn Write,
    model_id: &str,
    embedding: &[f32],
    input_tokens: u32,
) -> Result<(), EmbedError> {
    let vector_document = json!({
        "model": model_id,
        "embedding": embedding,
        "input_tokens": input_tokens,
    });
    let serialized_document =
        serde_json::to_string(&vector_document).map_err(|_serialization_error| {
            EmbedError::StdoutUnwritable {
                cause: "the vector document could not be serialized".to_owned(),
            }
        })?;
    stdout
        .write_all(serialized_document.as_bytes())
        .and_then(|()| {
            stdout.write_all(b"\n")?;
            stdout.flush()
        })
        .map_err(|output_error| EmbedError::StdoutUnwritable {
            cause: output_error.to_string(),
        })
}

/// Human phrasing for the typed embeddings failure, shared by Display and
/// Debug renderings.
pub(crate) fn embeddings_failure_reason_text(reason: &EmbeddingsFailureReason) -> String {
    match reason {
        EmbeddingsFailureReason::InvalidRequest { reason } => {
            format!("the model rejected the input: {reason}")
        }
        EmbeddingsFailureReason::FatalExecution { reason } => {
            format!("the model failed to finish the embedding: {reason}")
        }
        EmbeddingsFailureReason::ContextLengthExceeded {
            actual_total_context_tokens,
            maximum_context_tokens,
        } => format!(
            "the input needs {actual_total_context_tokens} context tokens but the model's \
             context window is {maximum_context_tokens} tokens"
        ),
        EmbeddingsFailureReason::EngineBusy => {
            "the embedding engine is busy with another request".to_owned()
        }
        EmbeddingsFailureReason::MalformedModelOutput => {
            "the model produced vectors that could not be pooled or normalized".to_owned()
        }
    }
}
