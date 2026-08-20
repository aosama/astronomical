//! Deterministic image terminal ordering for supervisor lifecycle tests.

use std::{io::Cursor, time::Duration};

use astronomical_ipc_protocol::{
    GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationFailureReason, ImageGenerationPhase, ImageGenerationResultMetadata,
    MlxMemorySnapshotSource, ProtocolError, ProtocolWriter, RequestId, WorkerEvent,
    WorkerMlxMemorySnapshot,
};
use image::{
    ExtendedColorType, ImageEncoder,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};

pub(super) fn image_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        minimum_width_pixels: 64,
        maximum_width_pixels: 1_024,
        minimum_height_pixels: 64,
        maximum_height_pixels: 1_024,
        dimension_multiple_pixels: 16,
        maximum_steps: 4,
        maximum_guidance_thousandths: 1_000,
        output_mime_types: vec!["image/png".to_owned()],
    }
}

pub(super) enum ScriptedImageCommandOutcome {
    Finished,
    CancellationPending { should_acknowledge: bool },
}

pub(super) async fn handle_image_command<WriteTransport>(
    generation_command: ImageGenerationCommand,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<ScriptedImageCommandOutcome, ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    match generation_command.prompt.as_str() {
        "delayed-image-generation-fixture" => {
            Ok(ScriptedImageCommandOutcome::CancellationPending {
                should_acknowledge: true,
            })
        }
        "unacknowledged-image-cancellation-fixture" => {
            Ok(ScriptedImageCommandOutcome::CancellationPending {
                should_acknowledge: false,
            })
        }
        "progress-stall-image-fixture" => {
            event_writer
                .send_event(&WorkerEvent::ImageGenerationProgress {
                    request_id: generation_command.request_id,
                    phase: ImageGenerationPhase::Preparing,
                    completed_steps: 0,
                    total_steps: generation_command.settings.steps,
                    elapsed_millis: 1,
                })
                .await?;
            Ok(ScriptedImageCommandOutcome::CancellationPending {
                should_acknowledge: true,
            })
        }
        "duplicate-progress-stall-image-fixture" => {
            for _duplicate_index in 0..3 {
                event_writer
                    .send_event(&WorkerEvent::ImageGenerationProgress {
                        request_id: generation_command.request_id,
                        phase: ImageGenerationPhase::Preparing,
                        completed_steps: 0,
                        total_steps: generation_command.settings.steps,
                        elapsed_millis: 1,
                    })
                    .await?;
                tokio::time::sleep(Duration::from_millis(70)).await;
            }
            Ok(ScriptedImageCommandOutcome::CancellationPending {
                should_acknowledge: true,
            })
        }
        "elapsed-progress-refresh-image-fixture" => {
            for elapsed_millis in [1, 2, 3] {
                event_writer
                    .send_event(&WorkerEvent::ImageGenerationProgress {
                        request_id: generation_command.request_id,
                        phase: ImageGenerationPhase::Preparing,
                        completed_steps: 0,
                        total_steps: generation_command.settings.steps,
                        elapsed_millis,
                    })
                    .await?;
                tokio::time::sleep(Duration::from_millis(70)).await;
            }
            send_completed_image(generation_command, Duration::ZERO, event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        "failed-image-generation-fixture" => {
            send_failed_image(generation_command.request_id, event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        "completion-before-finalization-fixture" => {
            send_completed_image(generation_command, Duration::from_millis(150), event_writer)
                .await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        "phase-regression-image-fixture" => {
            send_malformed_progress(generation_command, "phase", event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        "step-regression-image-fixture" => {
            send_malformed_progress(generation_command, "steps", event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        "elapsed-regression-image-fixture" => {
            send_malformed_progress(generation_command, "elapsed", event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        "guidance-mismatch-image-fixture" => {
            send_guidance_mismatch(generation_command, event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        "malformed-png-image-fixture" => {
            send_invalid_png_completion(generation_command, false, event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        "png-dimension-mismatch-image-fixture" => {
            send_invalid_png_completion(generation_command, true, event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
        _ => {
            send_completed_image(generation_command, Duration::ZERO, event_writer).await?;
            Ok(ScriptedImageCommandOutcome::Finished)
        }
    }
}

async fn send_guidance_mismatch<WriteTransport>(
    generation_command: ImageGenerationCommand,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::ImageGenerationCompleted {
            request_id: generation_command.request_id,
            generated_image: GeneratedImage {
                mime_type: "image/png".to_owned(),
                encoded_bytes: valid_png_bytes(
                    generation_command.settings.width_pixels,
                    generation_command.settings.height_pixels,
                ),
            },
            result_metadata: ImageGenerationResultMetadata {
                width_pixels: generation_command.settings.width_pixels,
                height_pixels: generation_command.settings.height_pixels,
                steps: generation_command.settings.steps,
                guidance_thousandths: generation_command.settings.guidance_thousandths + 1,
                seed: generation_command.settings.seed,
                elapsed_millis: 25,
            },
        })
        .await?;
    send_finalized_image(generation_command.request_id, event_writer).await
}

async fn send_invalid_png_completion<WriteTransport>(
    generation_command: ImageGenerationCommand,
    should_mismatch_dimensions: bool,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    let encoded_bytes = if should_mismatch_dimensions {
        valid_png_bytes(
            generation_command.settings.width_pixels / 2,
            generation_command.settings.height_pixels,
        )
    } else {
        vec![137, 80, 78, 71]
    };
    event_writer
        .send_event(&WorkerEvent::ImageGenerationCompleted {
            request_id: generation_command.request_id,
            generated_image: GeneratedImage {
                mime_type: "image/png".to_owned(),
                encoded_bytes,
            },
            result_metadata: ImageGenerationResultMetadata {
                width_pixels: generation_command.settings.width_pixels,
                height_pixels: generation_command.settings.height_pixels,
                steps: generation_command.settings.steps,
                guidance_thousandths: generation_command.settings.guidance_thousandths,
                seed: generation_command.settings.seed,
                elapsed_millis: 25,
            },
        })
        .await
}

pub(super) async fn send_completed_image<WriteTransport>(
    generation_command: ImageGenerationCommand,
    finalization_delay: Duration,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::ImageGenerationProgress {
            request_id: generation_command.request_id,
            phase: ImageGenerationPhase::Denoising,
            completed_steps: generation_command.settings.steps,
            total_steps: generation_command.settings.steps,
            elapsed_millis: 20,
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::ImageGenerationCompleted {
            request_id: generation_command.request_id,
            generated_image: GeneratedImage {
                mime_type: "image/png".to_owned(),
                encoded_bytes: valid_png_bytes(
                    generation_command.settings.width_pixels,
                    generation_command.settings.height_pixels,
                ),
            },
            result_metadata: ImageGenerationResultMetadata {
                width_pixels: generation_command.settings.width_pixels,
                height_pixels: generation_command.settings.height_pixels,
                steps: generation_command.settings.steps,
                guidance_thousandths: generation_command.settings.guidance_thousandths,
                seed: generation_command.settings.seed,
                elapsed_millis: 25,
            },
        })
        .await?;
    tokio::time::sleep(finalization_delay).await;
    send_finalized_image(generation_command.request_id, event_writer).await
}

// The idle-worker binary shares this module for swap coverage but does not script failures.
#[allow(dead_code)]
pub(super) async fn send_failed_image<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::ImageGenerationFailed {
            request_id,
            reason: ImageGenerationFailureReason::Cancelled,
        })
        .await?;
    send_finalized_image(request_id, event_writer).await
}

async fn send_malformed_progress<WriteTransport>(
    generation_command: ImageGenerationCommand,
    malformed_kind: &str,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    let (first_phase, second_phase, first_steps, second_steps, first_elapsed, second_elapsed) =
        match malformed_kind {
            "phase" => (
                ImageGenerationPhase::Decoding,
                ImageGenerationPhase::Denoising,
                2,
                2,
                20,
                21,
            ),
            "steps" => (
                ImageGenerationPhase::Denoising,
                ImageGenerationPhase::Denoising,
                2,
                1,
                20,
                21,
            ),
            _ => (
                ImageGenerationPhase::Denoising,
                ImageGenerationPhase::Denoising,
                1,
                2,
                20,
                19,
            ),
        };
    for (phase, completed_steps, elapsed_millis) in [
        (first_phase, first_steps, first_elapsed),
        (second_phase, second_steps, second_elapsed),
    ] {
        event_writer
            .send_event(&WorkerEvent::ImageGenerationProgress {
                request_id: generation_command.request_id,
                phase,
                completed_steps,
                total_steps: generation_command.settings.steps,
                elapsed_millis,
            })
            .await?;
    }
    Ok(())
}

async fn send_finalized_image<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::ImageGenerationFinalized {
            request_id,
            elapsed_millis: 30,
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source: MlxMemorySnapshotSource::Finalized,
                active_memory_bytes: 96_000_000,
                allocator_cache_memory_bytes: 0,
                peak_memory_bytes: 512_000_000,
                expert_payload_bytes: 0,
                model_core_payload_bytes: 96_000_000,
                context_state_payload_bytes: 0,
                speculative_prefill_draft_memory_bytes: 0,
            }),
        })
        .await
}

fn valid_png_bytes(width_pixels: u32, height_pixels: u32) -> Vec<u8> {
    let rgb_bytes = vec![0; width_pixels as usize * height_pixels as usize * 3];
    let mut png_bytes = Vec::new();
    PngEncoder::new_with_quality(
        Cursor::new(&mut png_bytes),
        CompressionType::Best,
        FilterType::Adaptive,
    )
    .write_image(
        &rgb_bytes,
        width_pixels,
        height_pixels,
        ExtendedColorType::Rgb8,
    )
    .expect("the scripted RGB image should encode as PNG");
    png_bytes
}
