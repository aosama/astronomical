//! The one-shot `astronomical image` journey: a prompt in, one PNG on disk,
//! the path on standard output, exit.
//!
//! Like every CLI verb it speaks the framed daemon IPC protocol over the
//! instance's unix socket and never touches the REST surface: the shared
//! model lifecycle resolves which model draws (flag, else the built-in image
//! model) and downloads it when the Mac lacks it, then one connection carries
//! the request. The daemon loads or swaps the resident model itself.

#[path = "image/image_arguments.rs"]
mod image_arguments;
#[path = "image/image_exchange.rs"]
mod image_exchange;
#[path = "image/image_output.rs"]
mod image_output;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::{
    DaemonRequest, DaemonResponse, GeneratedImage, ImageGenerateRequest,
    ImageGenerationFailureReason, ImageGenerationResultMetadata,
};
use thiserror::Error;

use crate::model_lifecycle::ProgressSink;
use crate::transport::ipc::DaemonProbeError;
use crate::{DaemonProbe, ModelLifecycle, RequiredCapability};

pub use image_arguments::{DEFAULT_IMAGE_SIDE_PIXELS, ImageArguments, parse_image_arguments};
use image_exchange::{ImageExchangeBounds, exchange_image_request};
use image_output::{ReservedOutput, discard_reserved_output, reserve_output_file, write_png};

/// Collaborators the image journey needs, injected so tests can stub them.
pub struct ImageDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<PathBuf>,
    /// Channel these sockets belong to: carried into the not-running
    /// disclosure so the failure names whose daemon is down.
    pub runtime_instance: AstronomicalRuntimeInstance,
    /// Where an unnamed output lands: the working directory.
    pub default_output_directory: PathBuf,
    /// Where the saved path goes.
    pub stdout: &'a mut dyn Write,
    /// Where progress and the summary go.
    pub stderr: &'a mut dyn Write,
    /// Bound for each short protocol stage of the journey.
    pub request_timeout: Duration,
    /// Bound for the whole wait on the image itself.
    pub generation_bound: Duration,
    /// Between elapsed-time lines while the image is drawn.
    pub progress_interval: Duration,
    /// How long a download may go without any change before the wait gives up.
    pub download_stall_bound: Duration,
    /// Wait between download status polls.
    pub download_poll_interval: Duration,
}

/// What the daemon returned, ready to be written.
struct GeneratedImageReply {
    model: String,
    image: GeneratedImage,
    metadata: ImageGenerationResultMetadata,
}

/// Runs the whole journey. The output path is claimed first and released on
/// every failure, so a failed run never leaves an empty file behind.
pub async fn run_image(
    image_arguments: &ImageArguments,
    image_dependencies: &mut ImageDependencies<'_>,
) -> Result<(), ImageError> {
    let reserved_output = reserve_output_file(
        image_arguments.output_path.as_deref(),
        &image_dependencies.default_output_directory,
    )?;
    match request_generated_image(image_arguments, image_dependencies).await {
        Ok(generated_reply) => {
            save_generated_image(reserved_output, generated_reply, image_dependencies)
        }
        Err(image_error) => {
            discard_reserved_output(reserved_output);
            Err(image_error)
        }
    }
}

/// Resolves the model, sends the request, and interprets the terminal frame.
async fn request_generated_image(
    image_arguments: &ImageArguments,
    image_dependencies: &mut ImageDependencies<'_>,
) -> Result<GeneratedImageReply, ImageError> {
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: image_dependencies.candidate_socket_paths.clone(),
        request_timeout: image_dependencies.request_timeout,
        runtime_instance: image_dependencies.runtime_instance,
    };
    let model_lifecycle = ModelLifecycle {
        daemon_probe,
        download_stall_bound: image_dependencies.download_stall_bound,
        download_poll_interval: image_dependencies.download_poll_interval,
    };
    let image_model_id =
        prepare_image_model(&model_lifecycle, image_arguments, image_dependencies.stderr).await?;
    let image_request = DaemonRequest::ImageGenerate(ImageGenerateRequest {
        model: image_model_id,
        prompt: image_arguments.prompt.clone(),
        width_pixels: image_arguments.width_pixels,
        height_pixels: image_arguments.height_pixels,
        seed: image_arguments.seed,
    });
    let exchange_bounds = ImageExchangeBounds {
        request_timeout: image_dependencies.request_timeout,
        generation_bound: image_dependencies.generation_bound,
        progress_interval: image_dependencies.progress_interval,
    };
    let mut daemon_client = model_lifecycle.daemon_probe.connect().await?;
    let daemon_response = exchange_image_request(
        &mut daemon_client,
        &image_request,
        &exchange_bounds,
        image_dependencies.stderr,
    )
    .await?;
    interpret_image_response(daemon_response)
}

/// Resolves the model (flag or the built-in image model), letting the
/// lifecycle download it when the Mac lacks it, with live progress.
async fn prepare_image_model(
    model_lifecycle: &ModelLifecycle,
    image_arguments: &ImageArguments,
    stderr: &mut dyn Write,
) -> Result<String, ImageError> {
    let mut download_progress = DownloadProgressLine {
        stderr,
        has_written: false,
    };
    let image_model_id = model_lifecycle
        .prepare_model_id(
            image_arguments.model_id.as_deref(),
            RequiredCapability::ImageGeneration,
            &mut download_progress,
        )
        .await?;
    if download_progress.has_written {
        // Finalize the live progress line before the next message owns it.
        let _ = writeln!(download_progress.stderr);
    }
    Ok(image_model_id)
}

/// The live download line: each report rewrites it in place on standard
/// error, the way `respond` and `embed` show a model download.
struct DownloadProgressLine<'a> {
    stderr: &'a mut dyn Write,
    has_written: bool,
}

impl ProgressSink for DownloadProgressLine<'_> {
    fn report(&mut self, progress_line: &str) {
        self.has_written = true;
        let _ = write!(self.stderr, "\r{progress_line}");
        let _ = self.stderr.flush();
    }
}

fn interpret_image_response(
    daemon_response: DaemonResponse,
) -> Result<GeneratedImageReply, ImageError> {
    match daemon_response {
        DaemonResponse::ImageGenerationCompleted {
            model,
            image,
            metadata,
        } => Ok(GeneratedImageReply {
            model,
            image,
            metadata,
        }),
        DaemonResponse::ImageGenerationFailed { reason } => Err(ImageError::ImageFailed { reason }),
        DaemonResponse::GenerationRejected { reason } => Err(ImageError::ImageRejected { reason }),
        // Handshake, status, chat, and embeddings frames cannot answer an
        // image request; treat the daemon as gone rather than guessing.
        _ => Err(ImageError::DaemonStoppedResponding),
    }
}

/// Writes the PNG, prints the saved path on standard output (what a script
/// captures), and summarises the run on standard error.
fn save_generated_image(
    reserved_output: ReservedOutput,
    generated_reply: GeneratedImageReply,
    image_dependencies: &mut ImageDependencies<'_>,
) -> Result<(), ImageError> {
    let saved_path = write_png(reserved_output, &generated_reply.image.encoded_bytes)?;
    write_saved_path(image_dependencies.stdout, &saved_path)?;
    let metadata = generated_reply.metadata;
    let _ = writeln!(
        image_dependencies.stderr,
        "saved {} with {}: {}x{}, seed {} (pass --seed {} to redraw it), {:.1}s",
        saved_path.display(),
        generated_reply.model,
        metadata.width_pixels,
        metadata.height_pixels,
        metadata.seed,
        metadata.seed,
        metadata.elapsed_millis as f64 / 1_000.0,
    );
    Ok(())
}

fn write_saved_path(stdout: &mut dyn Write, saved_path: &Path) -> Result<(), ImageError> {
    let write_result = match writeln!(stdout, "{}", saved_path.display()) {
        Ok(()) => stdout.flush(),
        Err(write_error) => Err(write_error),
    };
    match write_result {
        Ok(()) => Ok(()),
        Err(output_error) => Err(ImageError::StdoutUnwritable {
            cause: output_error.to_string(),
        }),
    }
}

/// Human phrasing for the typed image failure, shared by Display and Debug.
pub(crate) fn image_failure_reason_text(reason: &ImageGenerationFailureReason) -> String {
    match reason {
        ImageGenerationFailureReason::InvalidRequest { reason } => {
            format!("the model rejected the request: {reason}")
        }
        ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration => {
            "the model cannot draw images; pass --model with an image model".to_owned()
        }
        ImageGenerationFailureReason::EngineBusy => {
            "the image engine is busy with another request; try again in a moment".to_owned()
        }
        ImageGenerationFailureReason::EncodingFailed { reason } => {
            format!("the image could not be encoded as a PNG: {reason}")
        }
        ImageGenerationFailureReason::FatalExecution { reason } => {
            format!("the model failed to finish the image: {reason}")
        }
        ImageGenerationFailureReason::Cancelled => "the image was cancelled".to_owned(),
    }
}

/// Failures of the one-shot `image` journey after arguments have parsed.
/// These exit 1, except the usage-shaped ones (`is_usage_error`), which exit
/// 2 like a bad flag.
#[derive(Error)]
pub enum ImageError {
    #[error("{disclosure}")]
    DaemonNotRunning { disclosure: String },
    #[error("The Astronomical worker is not ready yet — retry in a moment.")]
    WorkerNotReady,
    #[error("{reason}")]
    ModelUnavailable { reason: String },
    #[error("the model download failed: {reason}")]
    DownloadFailed { reason: String },
    #[error("The daemon declined the request: {reason}")]
    ImageRejected { reason: String },
    #[error("{}", image_failure_reason_text(reason))]
    ImageFailed {
        reason: ImageGenerationFailureReason,
    },
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error(
        "The image did not finish within {seconds}s. The daemon does not cancel an image when the \
         command stops, so the worker may still be drawing it and a new image waits behind it; \
         run `astronomical status` to see whether the worker is still busy."
    )]
    GenerationTimedOut { seconds: u64 },
    #[error("{} already exists; choose another --output so nothing is overwritten", path.display())]
    OutputExists { path: PathBuf },
    #[error("the output must be a .png file because the image is a PNG: {}", path.display())]
    OutputMustBePng { path: PathBuf },
    #[error("Could not write the image to {}: {cause}", path.display())]
    OutputUnwritable { path: PathBuf, cause: String },
    #[error("The daemon returned something that is not a PNG image, so nothing was saved.")]
    ReturnedImageInvalid,
    #[error("Could not write the saved path to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

impl ImageError {
    /// Failures the user fixes by changing the command, not by retrying: a
    /// model that is absent or cannot draw, and an output path that is taken
    /// or mislabelled.
    pub fn is_usage_error(&self) -> bool {
        matches!(
            self,
            ImageError::ModelUnavailable { .. }
                | ImageError::OutputExists { .. }
                | ImageError::OutputMustBePng { .. }
        )
    }
}

impl From<DaemonProbeError> for ImageError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning { disclosure } => {
                ImageError::DaemonNotRunning { disclosure }
            }
            DaemonProbeError::DaemonStoppedResponding => ImageError::DaemonStoppedResponding,
            DaemonProbeError::DaemonRejected { reason } => ImageError::DownloadFailed { reason },
        }
    }
}

impl From<crate::model_lifecycle::ModelLifecycleError> for ImageError {
    fn from(lifecycle_error: crate::model_lifecycle::ModelLifecycleError) -> Self {
        match lifecycle_error {
            crate::model_lifecycle::ModelLifecycleError::DaemonNotRunning { disclosure } => {
                ImageError::DaemonNotRunning { disclosure }
            }
            crate::model_lifecycle::ModelLifecycleError::WorkerNotReady => {
                ImageError::WorkerNotReady
            }
            crate::model_lifecycle::ModelLifecycleError::DaemonStoppedResponding => {
                ImageError::DaemonStoppedResponding
            }
            crate::model_lifecycle::ModelLifecycleError::ModelUnavailable { reason } => {
                ImageError::ModelUnavailable { reason }
            }
            crate::model_lifecycle::ModelLifecycleError::DownloadFailed { reason } => {
                ImageError::DownloadFailed { reason }
            }
        }
    }
}

// The debug form carries the same guidance as the display form: journeys and
// tests report whichever rendering they captured.
impl std::fmt::Debug for ImageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

#[cfg(test)]
#[path = "image/image_tests.rs"]
mod image_tests;
