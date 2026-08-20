//! Connects the public image-generation executor contract to `WorkerHandle`.
//!
//! Queue ownership remains in `WorkerHandle` so chat and image requests share
//! one bounded FIFO admission path rather than drifting into modality-specific queues.

use std::{future::Future, pin::Pin};

use astronomical_ipc_protocol::ImageGenerationCommand;
use tokio::sync::{mpsc, oneshot};

use crate::{
    GenerationStartError, ImageGenerationExecutionError, ImageGenerationExecutor,
    ImageGenerationOutput, WorkerHandle,
};

impl ImageGenerationExecutor for WorkerHandle {
    fn start_image_generation(
        &self,
        image_generation_command: ImageGenerationCommand,
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
        self.start_image_generation_with_queue_admission(image_generation_command, None)
    }

    fn start_image_generation_with_admission_signal(
        &self,
        image_generation_command: ImageGenerationCommand,
        admission_sender: oneshot::Sender<()>,
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
        self.start_image_generation_with_queue_admission(
            image_generation_command,
            Some(admission_sender),
        )
    }
}
