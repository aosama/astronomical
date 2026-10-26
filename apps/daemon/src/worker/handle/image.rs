//! Connects the public image-generation executor contract to `WorkerHandle`.
//!
//! Queue ownership remains in `WorkerHandle` so chat and image requests share
//! one bounded FIFO admission path rather than drifting into modality-specific queues.

use astronomical_ipc_protocol::{EmbeddingsCommand, ImageGenerationCommand};
use tokio::sync::oneshot;

use crate::serving::GenerationOutcomeStartFuture;
use crate::{
    EmbeddingsExecutionError, EmbeddingsOutput, ImageGenerationExecutionError,
    ImageGenerationExecutor, ImageGenerationOutput, WorkerHandle,
};

impl ImageGenerationExecutor for WorkerHandle {
    fn start_image_generation(
        &self,
        image_generation_command: ImageGenerationCommand,
    ) -> GenerationOutcomeStartFuture<'_, ImageGenerationOutput, ImageGenerationExecutionError>
    {
        self.start_image_generation_with_queue_admission(image_generation_command, None)
    }

    fn start_image_generation_with_admission_signal(
        &self,
        image_generation_command: ImageGenerationCommand,
        admission_sender: oneshot::Sender<()>,
    ) -> GenerationOutcomeStartFuture<'_, ImageGenerationOutput, ImageGenerationExecutionError>
    {
        self.start_image_generation_with_queue_admission(
            image_generation_command,
            Some(admission_sender),
        )
    }

    fn start_embeddings_generation(
        &self,
        embeddings_command: EmbeddingsCommand,
    ) -> GenerationOutcomeStartFuture<'_, EmbeddingsOutput, EmbeddingsExecutionError> {
        self.start_embeddings_generation_with_queue_admission(embeddings_command, None)
    }

    fn start_embeddings_generation_with_admission_signal(
        &self,
        embeddings_command: EmbeddingsCommand,
        admission_sender: oneshot::Sender<()>,
    ) -> GenerationOutcomeStartFuture<'_, EmbeddingsOutput, EmbeddingsExecutionError> {
        self.start_embeddings_generation_with_queue_admission(
            embeddings_command,
            Some(admission_sender),
        )
    }
}
