//! Non-streaming image execution remains separate from token-streaming chat output.

use std::{future::Future, pin::Pin, time::Duration};

use astronomical_ipc_protocol::{
    GeneratedImage, ImageGenerationCommand, ImageGenerationFailureReason,
    ImageGenerationResultMetadata,
};
use tokio::sync::mpsc;

use crate::{ChatGenerationExecutor, GenerationStartError};

/// One completed worker image and its reproducibility metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageGenerationOutput {
    pub generated_image: GeneratedImage,
    pub result_metadata: ImageGenerationResultMetadata,
}

/// Failure delivered after an image request was admitted to the worker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageGenerationExecutionError {
    WorkerFailure(ImageGenerationFailureReason),
    DeadlineExceeded,
    WorkerUnavailable,
}

/// Supervisor-owned bounds for image execution and lack of forward progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageGenerationTimeouts {
    pub execution_timeout: Duration,
    pub progress_stall_timeout: Duration,
}

impl ImageGenerationTimeouts {
    pub const DEFAULT: Self = Self {
        execution_timeout: Duration::from_secs(15 * 60),
        progress_stall_timeout: Duration::from_secs(3 * 60),
    };

    #[must_use]
    pub const fn new(execution_timeout: Duration, progress_stall_timeout: Duration) -> Self {
        Self {
            execution_timeout,
            progress_stall_timeout,
        }
    }
}

/// Starts one image request through the same bounded worker admission owner as chat.
pub trait ImageGenerationExecutor: ChatGenerationExecutor {
    fn start_image_generation(
        &self,
        _image_generation_command: ImageGenerationCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<
                            Result<ImageGenerationOutput, ImageGenerationExecutionError>,
                        >,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async { Err(GenerationStartError::WorkerUnavailable) })
    }

    /// Signals once policy-bound work owns a place in the immutable FIFO queue.
    fn start_image_generation_with_admission_signal(
        &self,
        image_generation_command: ImageGenerationCommand,
        admission_sender: tokio::sync::oneshot::Sender<()>,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<
                            Result<ImageGenerationOutput, ImageGenerationExecutionError>,
                        >,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let _admission_signal_result = admission_sender.send(());
            self.start_image_generation(image_generation_command).await
        })
    }
}

pub(crate) async fn wait_for_image_disconnect(
    image_result_sender: Option<
        mpsc::Sender<Result<ImageGenerationOutput, ImageGenerationExecutionError>>,
    >,
) {
    let Some(image_result_sender) = image_result_sender else {
        std::future::pending::<()>().await;
        return;
    };
    image_result_sender.closed().await;
}
