//! The one-shot `astronomical decide` journey: a state and typed questions
//! in, one JSON document of answers out, exit.
//!
//! Like `embed`, the CLI never touches the REST surface. It speaks the
//! framed daemon IPC protocol over the instance's unix socket: the shared
//! model lifecycle resolves which decision model to use and downloads it
//! when the Mac lacks it, then one connection carries the request and the
//! daemon loads or swaps the resident model itself. The document printed is
//! the same System One response the REST endpoint returns.

use std::{
    io::{Read, Write},
    path::PathBuf,
    time::Duration,
};

mod input;

pub use input::ASK_QUESTION_NAME;
use input::{resolve_questions, resolve_state};

use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_decision_contract::{DecisionFailure, OrderedJsonValue, system_one_response};
use astronomical_ipc_protocol::{DaemonRequest, DaemonResponse};
use thiserror::Error;
use tokio::time::timeout;

use crate::daemon_frame_wait::{
    DaemonWaitBounds, DaemonWentSilent, DaemonWorkWatch, await_frame_watching_daemon_work,
};
use crate::transport::ipc::DaemonProbeError;
use crate::{DaemonProbe, ModelLifecycle, RequiredCapability};

/// Parsed `astronomical decide` invocation before any IPC work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecideArguments {
    /// The state as a positional argument; `None` reads a file or stdin.
    pub state_text: Option<String>,
    /// A file holding the state.
    pub state_file_path: Option<PathBuf>,
    /// Whether the state is JSON to be read as structured data, not text.
    pub state_is_json: bool,
    /// Where the questions come from. Parsing guarantees exactly one source.
    pub questions_source: DecideQuestionsSource,
    pub model_id: Option<String>,
}

/// The one place a `decide` invocation's questions come from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecideQuestionsSource {
    /// One yes-or-no question given with `--ask`.
    Ask(String),
    /// A System One `questions` object given inline with `--questions`.
    InlineJson(String),
    /// A file holding a System One `questions` object.
    File(PathBuf),
}

/// Collaborators the decide journey needs, injected so tests can stub them.
pub struct DecideDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<PathBuf>,
    /// Channel these sockets belong to, named when the daemon is down.
    pub runtime_instance: AstronomicalRuntimeInstance,
    /// Read to EOF when no state text or file was supplied.
    pub stdin: &'a mut dyn Read,
    /// Where the JSON answers document goes.
    pub stdout: &'a mut dyn Write,
    /// Where download progress and the model load line go.
    pub stderr: &'a mut dyn Write,
    /// How long each protocol stage may go with nothing heard from the daemon.
    pub request_timeout: Duration,
    /// How long a download may go without any change before the wait gives up.
    pub download_stall_bound: Duration,
    /// Wait between download status polls.
    pub download_poll_interval: Duration,
    /// Wait between asking what the daemon is doing while no answer has come.
    pub daemon_work_poll_interval: Duration,
    /// How long one load may stay reported before the wait gives up.
    pub model_load_stall_bound: Duration,
}

/// Runs the whole decide journey against the resident daemon.
///
/// The state and questions are read and checked first, so a malformed
/// invocation fails before any model is downloaded or loaded for it.
pub async fn run_decide(
    decide_arguments: &DecideArguments,
    decide_dependencies: &mut DecideDependencies<'_>,
) -> Result<(), DecideError> {
    let questions = resolve_questions(&decide_arguments.questions_source)?;
    let state = resolve_state(decide_arguments, decide_dependencies.stdin)?;
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: decide_dependencies.candidate_socket_paths.clone(),
        request_timeout: decide_dependencies.request_timeout,
        runtime_instance: decide_dependencies.runtime_instance,
    };
    let model_lifecycle = ModelLifecycle {
        daemon_probe,
        download_stall_bound: decide_dependencies.download_stall_bound,
        download_poll_interval: decide_dependencies.download_poll_interval,
    };
    let mut progress_started = false;
    let decision_model_id = model_lifecycle
        .prepare_model_id(
            decide_arguments.model_id.as_deref(),
            RequiredCapability::Decisions,
            &mut |progress_line: &str| {
                progress_started = true;
                let _ = write!(decide_dependencies.stderr, "\r{progress_line}");
                let _ = decide_dependencies.stderr.flush();
            },
        )
        .await?;
    if progress_started {
        // Finalize the live progress line before the JSON document owns stdout.
        let _ = writeln!(decide_dependencies.stderr);
    }
    submit_decision_request(
        state,
        questions,
        decision_model_id,
        &model_lifecycle.daemon_probe,
        decide_dependencies,
    )
    .await
}

/// Submits one decision request and writes the single terminal frame as one
/// JSON document on stdout.
async fn submit_decision_request(
    state: OrderedJsonValue,
    questions: OrderedJsonValue,
    decision_model_id: String,
    daemon_probe: &DaemonProbe,
    decide_dependencies: &mut DecideDependencies<'_>,
) -> Result<(), DecideError> {
    let mut daemon_client = daemon_probe.connect().await?;
    let decide_generate_request = DaemonRequest::DecideGenerate {
        model: Some(decision_model_id),
        state,
        questions,
    };
    match timeout(
        decide_dependencies.request_timeout,
        daemon_client.send_request(&decide_generate_request),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(_)) | Err(_) => return Err(DecideError::DaemonStoppedResponding),
    }
    // The request can wait behind a generation or a model load with nothing on this
    // connection, so the bound is on hearing nothing at all, not on the answer's time.
    let wait_bounds = DaemonWaitBounds {
        silence_bound: decide_dependencies.request_timeout,
        load_stall_bound: decide_dependencies.model_load_stall_bound,
        work_poll_interval: decide_dependencies.daemon_work_poll_interval,
    };
    let terminal_frame = await_frame_watching_daemon_work(
        daemon_client.next_response(),
        daemon_probe,
        wait_bounds,
        &mut DaemonWorkWatch::default(),
        &mut *decide_dependencies.stderr,
    )
    .await?;
    let decisions_response = match terminal_frame {
        Ok(Some(decisions_response)) => decisions_response,
        Ok(None) | Err(_) => return Err(DecideError::DaemonStoppedResponding),
    };
    write_decisions_outcome(decisions_response, decide_dependencies.stdout)
}

/// Writes the one frame that ends a decision request.
fn write_decisions_outcome(
    decisions_response: DaemonResponse,
    stdout: &mut dyn Write,
) -> Result<(), DecideError> {
    match decisions_response {
        DaemonResponse::DecisionsCompleted {
            model,
            question_outcomes,
            input_token_count,
            ..
        } => {
            let answers_document =
                system_one_response(&model, &question_outcomes, input_token_count);
            write_json_document(stdout, &answers_document)
        }
        DaemonResponse::DecisionsFailed { reason } => Err(DecideError::DecisionsFailed { reason }),
        DaemonResponse::GenerationRejected { reason } => {
            Err(DecideError::DecisionsRejected { reason })
        }
        // No other frame can answer a decision request; treat the daemon as
        // gone rather than guessing.
        _ => Err(DecideError::DaemonStoppedResponding),
    }
}

fn write_json_document(
    stdout: &mut dyn Write,
    answers_document: &OrderedJsonValue,
) -> Result<(), DecideError> {
    let Ok(serialized_document) = serde_json::to_string(answers_document) else {
        return Err(DecideError::StdoutUnwritable {
            cause: "the answers document could not be serialized".to_owned(),
        });
    };
    let write_result = stdout
        .write_all(serialized_document.as_bytes())
        .and_then(|()| stdout.write_all(b"\n"))
        .and_then(|()| stdout.flush());
    match write_result {
        Ok(()) => Ok(()),
        Err(output_error) => Err(DecideError::StdoutUnwritable {
            cause: output_error.to_string(),
        }),
    }
}

/// Human phrasing for the typed decisions failure.
fn decisions_failure_reason_text(reason: &DecisionFailure) -> String {
    match reason {
        DecisionFailure::InvalidRequest { reason } => {
            format!("the model cannot answer this request: {reason}")
        }
        DecisionFailure::FatalExecution { reason } => {
            format!("the model failed to answer: {reason}")
        }
        DecisionFailure::EngineBusy => {
            "the decision engine is busy with another request".to_owned()
        }
    }
}

/// Failures of the one-shot `decide` journey after arguments have parsed.
/// These exit 1, except the input and model errors, which exit 2 as usage errors.
#[derive(Error)]
pub enum DecideError {
    #[error("{disclosure}")]
    DaemonNotRunning { disclosure: String },
    #[error("The Astronomical worker is not ready yet — retry in a moment.")]
    WorkerNotReady,
    #[error("{reason}")]
    ModelUnavailable { reason: String },
    #[error("the model download failed: {reason}")]
    DownloadFailed { reason: String },
    #[error("The daemon declined the request: {reason}")]
    DecisionsRejected { reason: String },
    #[error("{}", decisions_failure_reason_text(reason))]
    DecisionsFailed { reason: DecisionFailure },
    #[error(
        "decide needs a state to decide about. Try: astronomical decide 'We were billed twice.' --ask 'Is a refund requested?'"
    )]
    StateRequired,
    #[error("--ask needs a question. Try: --ask 'Is a refund requested?'")]
    EmptyQuestion,
    #[error("the state is not valid JSON, which --json-state requires: {cause}")]
    StateNotJson { cause: String },
    #[error("the questions are not valid JSON: {cause}")]
    QuestionsNotJson { cause: String },
    #[error("Could not read {file_path}: {cause}")]
    InputFileUnreadable { file_path: PathBuf, cause: String },
    #[error("Could not read standard input: {cause}")]
    StdinUnreadable { cause: String },
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("Could not write the answers document to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

impl DecideError {
    /// Whether the failure is the invocation's fault, so the process exits 2
    /// like a bad flag.
    #[must_use]
    pub const fn is_usage_error(&self) -> bool {
        matches!(
            self,
            Self::ModelUnavailable { .. }
                | Self::StateRequired
                | Self::EmptyQuestion
                | Self::StateNotJson { .. }
                | Self::QuestionsNotJson { .. }
                | Self::InputFileUnreadable { .. }
        )
    }
}

impl From<DaemonWentSilent> for DecideError {
    fn from(_daemon_went_silent: DaemonWentSilent) -> Self {
        DecideError::DaemonStoppedResponding
    }
}

impl From<DaemonProbeError> for DecideError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning { disclosure } => {
                DecideError::DaemonNotRunning { disclosure }
            }
            DaemonProbeError::DaemonStoppedResponding => DecideError::DaemonStoppedResponding,
            DaemonProbeError::DaemonRejected { reason } => DecideError::DownloadFailed { reason },
        }
    }
}

impl From<crate::model_lifecycle::ModelLifecycleError> for DecideError {
    fn from(lifecycle_error: crate::model_lifecycle::ModelLifecycleError) -> Self {
        match lifecycle_error {
            crate::model_lifecycle::ModelLifecycleError::DaemonNotRunning { disclosure } => {
                DecideError::DaemonNotRunning { disclosure }
            }
            crate::model_lifecycle::ModelLifecycleError::WorkerNotReady => {
                DecideError::WorkerNotReady
            }
            crate::model_lifecycle::ModelLifecycleError::DaemonStoppedResponding => {
                DecideError::DaemonStoppedResponding
            }
            crate::model_lifecycle::ModelLifecycleError::ModelUnavailable { reason } => {
                DecideError::ModelUnavailable { reason }
            }
            crate::model_lifecycle::ModelLifecycleError::DownloadFailed { reason } => {
                DecideError::DownloadFailed { reason }
            }
        }
    }
}

// The debug form must carry the same user-facing guidance as the display
// form: journeys and tests report whichever rendering they captured.
impl std::fmt::Debug for DecideError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

#[cfg(test)]
#[path = "decide_tests.rs"]
mod tests;
