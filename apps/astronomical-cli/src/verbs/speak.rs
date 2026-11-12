//! The one-shot `astronomical speak` journey: text in, one audio file on
//! disk, the path on standard output, exit.
//!
//! Like every CLI verb it speaks the framed daemon IPC protocol over the
//! instance's unix socket: the shared model lifecycle resolves which model
//! speaks (flag, else the built-in speech model) and downloads it when the
//! Mac lacks it, then one connection carries the request. The audio comes
//! back in bounded frames and is saved as WAV, or as bare samples for a
//! `.pcm` output.

#[path = "speak/speak_arguments.rs"]
mod speak_arguments;
#[path = "speak/speak_exchange.rs"]
mod speak_exchange;
#[path = "speak/speak_output.rs"]
mod speak_output;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::{DaemonRequest, SpeakGenerateRequest};
use astronomical_speech_contract::SpeechFailure;
use thiserror::Error;

use crate::transport::ipc::DaemonProbeError;
use crate::verbs::download_progress_line::DownloadProgressLine;
use crate::{DaemonProbe, ModelLifecycle, RequiredCapability};

pub use speak_arguments::{SpeakArguments, parse_speak_arguments};
use speak_exchange::{SpeakExchangeBounds, SpokenReply, exchange_speak_request};
use speak_output::{ReservedAudioOutput, reserve_audio_output, write_audio};

/// Collaborators the speak journey needs, injected so tests can stub them.
pub struct SpeakDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<PathBuf>,
    /// Channel these sockets belong to, named when its daemon is down.
    pub runtime_instance: AstronomicalRuntimeInstance,
    /// Where an unnamed output lands: the working directory.
    pub default_output_directory: PathBuf,
    /// Read to EOF when neither TEXT nor `--file` was given.
    pub stdin: &'a mut dyn Read,
    /// Where the saved path goes.
    pub stdout: &'a mut dyn Write,
    /// Where progress and the summary go.
    pub stderr: &'a mut dyn Write,
    /// Bound for each short protocol stage of the journey.
    pub request_timeout: Duration,
    /// Bound for the whole wait on the audio.
    pub generation_bound: Duration,
    /// Between elapsed-time lines while the text is spoken.
    pub progress_interval: Duration,
    /// How long a download may go without any change before the wait gives up.
    pub download_stall_bound: Duration,
    /// Wait between download status polls.
    pub download_poll_interval: Duration,
}

/// Runs the whole journey. The text is read and the output claimed before
/// the daemon is asked, and the claim is released on every failure, so a
/// failed run never leaves an empty file behind.
pub async fn run_speak(
    speak_arguments: &SpeakArguments,
    speak_dependencies: &mut SpeakDependencies<'_>,
) -> Result<(), SpeakError> {
    let spoken_text = resolve_spoken_text(speak_arguments, speak_dependencies.stdin)?;
    let reserved_output = reserve_audio_output(
        speak_arguments.output_path.as_deref(),
        &speak_dependencies.default_output_directory,
    )?;
    match request_speech(speak_arguments, spoken_text, speak_dependencies).await {
        Ok(spoken_reply) => save_spoken_audio(reserved_output, spoken_reply, speak_dependencies),
        Err(speak_error) => {
            reserved_output.reserved_file.discard();
            Err(speak_error)
        }
    }
}

/// The text from the argument, the file, or standard input, in that order.
fn resolve_spoken_text(
    speak_arguments: &SpeakArguments,
    stdin: &mut dyn Read,
) -> Result<String, SpeakError> {
    let spoken_text = match (&speak_arguments.text, &speak_arguments.text_file) {
        (Some(text), _) => text.clone(),
        (None, Some(text_file)) => {
            std::fs::read_to_string(text_file).map_err(|read_error| SpeakError::TextUnreadable {
                path: text_file.clone(),
                cause: read_error.to_string(),
            })?
        }
        (None, None) => {
            let mut stdin_text = String::new();
            stdin
                .read_to_string(&mut stdin_text)
                .map_err(|read_error| SpeakError::TextUnreadable {
                    path: PathBuf::from("standard input"),
                    cause: read_error.to_string(),
                })?;
            stdin_text
        }
    };
    if spoken_text.trim().is_empty() {
        return Err(SpeakError::TextEmpty);
    }
    Ok(spoken_text)
}

/// Resolves the model, sends the request, and collects the audio.
async fn request_speech(
    speak_arguments: &SpeakArguments,
    spoken_text: String,
    speak_dependencies: &mut SpeakDependencies<'_>,
) -> Result<SpokenReply, SpeakError> {
    let model_lifecycle = ModelLifecycle {
        daemon_probe: DaemonProbe {
            candidate_socket_paths: speak_dependencies.candidate_socket_paths.clone(),
            request_timeout: speak_dependencies.request_timeout,
            runtime_instance: speak_dependencies.runtime_instance,
        },
        download_stall_bound: speak_dependencies.download_stall_bound,
        download_poll_interval: speak_dependencies.download_poll_interval,
    };
    let speech_model_id = prepare_speech_model(
        &model_lifecycle,
        speak_arguments.model_id.as_deref(),
        speak_dependencies.stderr,
    )
    .await?;
    let speak_request = DaemonRequest::SpeakGenerate(SpeakGenerateRequest {
        model: Some(speech_model_id),
        text: spoken_text,
        voice_name: speak_arguments.voice_name.clone(),
        speed: speak_arguments.speed,
        seed: speak_arguments.seed,
    });
    let exchange_bounds = SpeakExchangeBounds {
        request_timeout: speak_dependencies.request_timeout,
        generation_bound: speak_dependencies.generation_bound,
        progress_interval: speak_dependencies.progress_interval,
    };
    let mut daemon_client = model_lifecycle.daemon_probe.connect().await?;
    exchange_speak_request(
        &mut daemon_client,
        &speak_request,
        &exchange_bounds,
        speak_dependencies.stderr,
    )
    .await
}

/// Resolves the model (flag or the built-in speech model), letting the
/// lifecycle download it when the Mac lacks it, with live progress.
async fn prepare_speech_model(
    model_lifecycle: &ModelLifecycle,
    requested_model_id: Option<&str>,
    stderr: &mut dyn Write,
) -> Result<String, SpeakError> {
    let mut download_progress = DownloadProgressLine::new(stderr);
    let speech_model_id = model_lifecycle
        .prepare_model_id(
            requested_model_id,
            RequiredCapability::Speech,
            &mut download_progress,
        )
        .await?;
    download_progress.finish();
    Ok(speech_model_id)
}

/// Writes the audio, prints the saved path on standard output (what a
/// script captures), and summarises the run on standard error.
fn save_spoken_audio(
    reserved_output: ReservedAudioOutput,
    spoken_reply: SpokenReply,
    speak_dependencies: &mut SpeakDependencies<'_>,
) -> Result<(), SpeakError> {
    let metadata = spoken_reply.metadata;
    let saved_path = write_audio(
        reserved_output,
        &spoken_reply.pcm16_bytes,
        metadata.sample_rate_hertz,
    )?;
    write_saved_path(speak_dependencies.stdout, &saved_path)?;
    let audio_seconds = metadata.sample_count as f64 / f64::from(metadata.sample_rate_hertz.max(1));
    let _ = writeln!(
        speak_dependencies.stderr,
        "saved {} with {} in voice {}: {audio_seconds:.1}s of audio at {} Hz, seed {} (pass --seed {} to say it again the same way)",
        saved_path.display(),
        spoken_reply.model,
        metadata.voice_name,
        metadata.sample_rate_hertz,
        metadata.seed,
        metadata.seed,
    );
    if metadata.cut_short_segment_count > 0 {
        let _ = writeln!(
            speak_dependencies.stderr,
            "note: the model stopped early in {} part(s) of the text; shorter sentences may help",
            metadata.cut_short_segment_count
        );
    }
    Ok(())
}

fn write_saved_path(stdout: &mut dyn Write, saved_path: &Path) -> Result<(), SpeakError> {
    let write_result = match writeln!(stdout, "{}", saved_path.display()) {
        Ok(()) => stdout.flush(),
        Err(write_error) => Err(write_error),
    };
    write_result.map_err(|output_error| SpeakError::StdoutUnwritable {
        cause: output_error.to_string(),
    })
}

/// Failures of the one-shot `speak` journey after arguments have parsed.
/// These exit 1, except the usage-shaped ones (`is_usage_error`), which exit
/// 2 like a bad flag.
#[derive(Error)]
pub enum SpeakError {
    #[error("{disclosure}")]
    DaemonNotRunning { disclosure: String },
    #[error("The Astronomical worker is not ready yet — retry in a moment.")]
    WorkerNotReady,
    #[error("{reason}")]
    ModelUnavailable { reason: String },
    #[error("the model download failed: {reason}")]
    DownloadFailed { reason: String },
    #[error("The daemon declined the request: {reason}")]
    SpeechRejected { reason: String },
    #[error("{reason}")]
    SpeechFailed { reason: SpeechFailure },
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error(
        "The audio did not finish within {seconds}s; run `astronomical status` to see whether the \
         worker is still busy."
    )]
    SpeechTimedOut { seconds: u64 },
    #[error(
        "speak needs text. Try: astronomical speak 'But soft, what light through yonder window breaks?'"
    )]
    TextEmpty,
    #[error("Could not read the text from {}: {cause}", path.display())]
    TextUnreadable { path: PathBuf, cause: String },
    #[error("{} already exists; choose another --output so nothing is overwritten", path.display())]
    OutputExists { path: PathBuf },
    #[error("the output must be a .wav file, or .pcm for bare samples: {}", path.display())]
    OutputMustBeAudio { path: PathBuf },
    #[error("Could not write the audio to {}: {cause}", path.display())]
    OutputUnwritable { path: PathBuf, cause: String },
    #[error("The daemon returned audio that is not whole 16-bit samples, so nothing was saved.")]
    ReturnedAudioInvalid,
    #[error("Could not write the saved path to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

impl SpeakError {
    /// Failures the user fixes by changing the command, not by retrying.
    pub fn is_usage_error(&self) -> bool {
        matches!(
            self,
            SpeakError::ModelUnavailable { .. }
                | SpeakError::TextEmpty
                | SpeakError::TextUnreadable { .. }
                | SpeakError::OutputExists { .. }
                | SpeakError::OutputMustBeAudio { .. }
                | SpeakError::SpeechFailed {
                    reason: SpeechFailure::InvalidRequest { .. }
                }
        )
    }
}

impl From<DaemonProbeError> for SpeakError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning { disclosure } => {
                SpeakError::DaemonNotRunning { disclosure }
            }
            DaemonProbeError::DaemonStoppedResponding => SpeakError::DaemonStoppedResponding,
            DaemonProbeError::DaemonRejected { reason } => SpeakError::DownloadFailed { reason },
        }
    }
}

impl From<crate::model_lifecycle::ModelLifecycleError> for SpeakError {
    fn from(lifecycle_error: crate::model_lifecycle::ModelLifecycleError) -> Self {
        use crate::model_lifecycle::ModelLifecycleError;
        match lifecycle_error {
            ModelLifecycleError::DaemonNotRunning { disclosure } => {
                SpeakError::DaemonNotRunning { disclosure }
            }
            ModelLifecycleError::WorkerNotReady => SpeakError::WorkerNotReady,
            ModelLifecycleError::DaemonStoppedResponding => SpeakError::DaemonStoppedResponding,
            ModelLifecycleError::ModelUnavailable { reason } => {
                SpeakError::ModelUnavailable { reason }
            }
            ModelLifecycleError::DownloadFailed { reason } => SpeakError::DownloadFailed { reason },
        }
    }
}

// The debug form carries the same guidance as the display form: journeys and
// tests report whichever rendering they captured.
impl std::fmt::Debug for SpeakError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

#[cfg(test)]
#[path = "speak/speak_tests.rs"]
mod speak_tests;
