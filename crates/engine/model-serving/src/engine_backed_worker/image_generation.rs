//! Serializes one bounded image request and publishes cleanup after its private outcome.

use super::output::finalized_memory_snapshot;
use super::worker_engines::WorkerEngines;
use ipc_protocol::{
    ImageGenerationCommand, ImageGenerationFailureReason, MlxMemorySnapshotSource, ProtocolWriter,
    RequestId, WorkerEvent, WorkerMlxMemorySnapshot,
};
use tokio::io::AsyncWrite;

use super::support::{ActiveImageGeneration, ModelFactory, WorkerRuntimeError};
use super::{EngineBackedWorker, LoadedRuntime};
use crate::{ImageGenerationEngine, ImageGenerationEngineStep};

impl<Engines: WorkerEngines> EngineBackedWorker<Engines> {
    pub(crate) async fn start_image_generation<WriteTransport>(
        &mut self,
        generation_command: ImageGenerationCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveImageGeneration>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = generation_command.request_id;
        let total_steps = generation_command.settings.steps;
        if let Err(validation_error) = generation_command.validate() {
            self.emit_image_failure_and_finalization(
                request_id,
                ImageGenerationFailureReason::invalid_request(validation_error.to_string()),
                0,
                event_writer,
            )
            .await?;
            return Ok(None);
        }
        let start_result = match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Image(image_engine)) => {
                image_engine.start_generation(generation_command)
            }
            Some(LoadedRuntime::Autoregressive(_))
            | Some(LoadedRuntime::Embeddings(_))
            | Some(LoadedRuntime::Decisions(_))
            | Some(LoadedRuntime::Speech(_))
            | None => Err(ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration),
        };
        match start_result {
            Ok(()) => Ok(Some(ActiveImageGeneration::new(
                request_id,
                total_steps,
                self.model_factory
                    .as_ref()
                    .is_some_and(ModelFactory::performance_attribution_enabled),
            ))),
            Err(failure_reason) => {
                self.release_failed_image_request(request_id)?;
                self.emit_image_failure_and_finalization(
                    request_id,
                    failure_reason,
                    0,
                    event_writer,
                )
                .await?;
                Ok(None)
            }
        }
    }

    pub(crate) async fn advance_image_generation<WriteTransport>(
        &mut self,
        active_generation: ActiveImageGeneration,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ActiveImageGeneration>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = active_generation.request_id;
        let engine_step = self.advance_loaded_image_engine(request_id);
        match engine_step {
            Ok(ImageGenerationEngineStep::Progress {
                phase,
                completed_steps,
                total_steps,
                elapsed_millis,
            }) if progress_is_within_requested_steps(
                completed_steps,
                total_steps,
                active_generation.total_steps,
            ) =>
            {
                let mlx_memory_snapshot = self.collect_image_step_memory_snapshot();
                event_writer
                    .send_event(&WorkerEvent::ImageGenerationProgress {
                        request_id,
                        phase,
                        completed_steps,
                        total_steps,
                        elapsed_millis,
                        mlx_memory_snapshot,
                    })
                    .await?;
                Ok(Some(active_generation))
            }
            Ok(ImageGenerationEngineStep::Progress { .. }) => {
                self.release_failed_image_request(request_id)?;
                self.emit_image_failure_and_finalization(
                    request_id,
                    ImageGenerationFailureReason::FatalExecution {
                        reason: "image engine reported progress outside the requested step bounds"
                            .to_owned(),
                    },
                    active_generation.elapsed_millis(),
                    event_writer,
                )
                .await?;
                Ok(None)
            }
            Ok(ImageGenerationEngineStep::Completed {
                generated_image,
                result_metadata,
            }) => {
                event_writer
                    .send_event(&WorkerEvent::ImageGenerationCompleted {
                        request_id,
                        generated_image,
                        result_metadata,
                    })
                    .await?;
                self.emit_image_finalization(&active_generation, event_writer)
                    .await?;
                Ok(None)
            }
            Err(failure_reason) => {
                self.release_failed_image_request(request_id)?;
                self.emit_image_failure_and_finalization(
                    request_id,
                    failure_reason,
                    active_generation.elapsed_millis(),
                    event_writer,
                )
                .await?;
                Ok(None)
            }
        }
    }

    /// Takes one denoising step on the loaded image engine; finding no image
    /// runtime loaded any more is a fatal failure of that request.
    fn advance_loaded_image_engine(
        &mut self,
        request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Image(image_engine)) => image_engine.advance_generation(request_id),
            Some(LoadedRuntime::Autoregressive(_))
            | Some(LoadedRuntime::Embeddings(_))
            | Some(LoadedRuntime::Decisions(_))
            | Some(LoadedRuntime::Speech(_))
            | None => Err(ImageGenerationFailureReason::FatalExecution {
                reason: "the loaded image runtime was removed during generation".to_owned(),
            }),
        }
    }

    /// MLX memory as the image engine reports it after a denoising step, or
    /// `None` when no image engine is loaded or it has nothing to report.
    fn collect_image_step_memory_snapshot(&self) -> Option<WorkerMlxMemorySnapshot> {
        let memory_telemetry = match self.loaded_runtime.as_ref() {
            Some(LoadedRuntime::Image(image_engine)) => image_engine.collect_mlx_memory_telemetry(),
            Some(LoadedRuntime::Autoregressive(_))
            | Some(LoadedRuntime::Embeddings(_))
            | Some(LoadedRuntime::Decisions(_))
            | Some(LoadedRuntime::Speech(_))
            | None => None,
        }?;
        Some(super::output::worker_memory_snapshot(
            MlxMemorySnapshotSource::ImageGenerationStep,
            memory_telemetry,
        ))
    }

    fn release_failed_image_request(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), WorkerRuntimeError> {
        if let Some(LoadedRuntime::Image(image_engine)) = self.loaded_runtime.as_mut() {
            image_engine
                .cancel_generation(request_id)
                .map_err(|cleanup_failure| image_cleanup_error(request_id, cleanup_failure))?;
        }
        Ok(())
    }

    pub(crate) async fn cancel_image_generation<WriteTransport>(
        &mut self,
        active_generation: ActiveImageGeneration,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = active_generation.request_id;
        let cancellation = match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Image(image_engine)) => image_engine.cancel_generation(request_id),
            Some(LoadedRuntime::Autoregressive(_))
            | Some(LoadedRuntime::Embeddings(_))
            | Some(LoadedRuntime::Decisions(_))
            | Some(LoadedRuntime::Speech(_))
            | None => Ok(()),
        };
        cancellation.map_err(|cleanup_failure| image_cleanup_error(request_id, cleanup_failure))?;
        self.emit_image_failure_and_finalization(
            request_id,
            ImageGenerationFailureReason::Cancelled,
            active_generation.elapsed_millis(),
            event_writer,
        )
        .await
    }

    pub(crate) async fn emit_image_failure_and_finalization<WriteTransport>(
        &mut self,
        request_id: RequestId,
        failure_reason: ImageGenerationFailureReason,
        elapsed_millis: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::ImageGenerationFailed {
                request_id,
                reason: bounded_image_failure_reason(failure_reason),
            })
            .await?;
        self.emit_image_finalization_event(request_id, elapsed_millis, event_writer)
            .await
    }

    async fn emit_image_finalization<WriteTransport>(
        &mut self,
        active_generation: &ActiveImageGeneration,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        self.emit_image_finalization_event(
            active_generation.request_id,
            active_generation.elapsed_millis(),
            event_writer,
        )
        .await
    }

    async fn emit_image_finalization_event<WriteTransport>(
        &mut self,
        request_id: RequestId,
        elapsed_millis: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let post_cleanup_memory_telemetry = match self.loaded_runtime.as_mut() {
            Some(LoadedRuntime::Image(image_engine)) => {
                image_engine.take_post_cleanup_memory_telemetry()
            }
            Some(LoadedRuntime::Autoregressive(_))
            | Some(LoadedRuntime::Embeddings(_))
            | Some(LoadedRuntime::Decisions(_))
            | Some(LoadedRuntime::Speech(_))
            | None => None,
        };
        event_writer
            .send_event(&WorkerEvent::ImageGenerationFinalized {
                request_id,
                elapsed_millis,
                mlx_memory_snapshot: finalized_memory_snapshot(post_cleanup_memory_telemetry),
            })
            .await?;
        Ok(())
    }
}

/// Whether an engine's progress report agrees with the request: the engine
/// must run the step count that was asked for, and may not report more steps
/// done than that count.
fn progress_is_within_requested_steps(
    completed_steps: u16,
    reported_total_steps: u16,
    requested_total_steps: u16,
) -> bool {
    reported_total_steps == requested_total_steps && completed_steps <= requested_total_steps
}

fn image_cleanup_error(
    request_id: RequestId,
    cleanup_failure: ImageGenerationFailureReason,
) -> WorkerRuntimeError {
    WorkerRuntimeError::InferenceEngineGenerationFailed {
        reason: format!(
            "image generation cleanup failed for request {}: {cleanup_failure:?}",
            request_id.value(),
        ),
    }
}

fn bounded_image_failure_reason(
    failure_reason: ImageGenerationFailureReason,
) -> ImageGenerationFailureReason {
    match failure_reason {
        ImageGenerationFailureReason::InvalidRequest { reason } => {
            ImageGenerationFailureReason::InvalidRequest {
                reason: reason.chars().take(256).collect(),
            }
        }
        ImageGenerationFailureReason::EncodingFailed { reason } => {
            ImageGenerationFailureReason::EncodingFailed {
                reason: reason.chars().take(256).collect(),
            }
        }
        ImageGenerationFailureReason::FatalExecution { reason } => {
            ImageGenerationFailureReason::FatalExecution {
                reason: reason.chars().take(256).collect(),
            }
        }
        other_failure_reason => other_failure_reason,
    }
}

#[cfg(test)]
#[path = "image_generation_tests.rs"]
mod image_generation_tests;
