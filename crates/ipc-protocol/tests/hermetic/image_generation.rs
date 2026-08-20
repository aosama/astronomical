use std::{io::Cursor, ops::Range};

use astronomical_ipc_protocol::{
    ChatModelCapabilities, GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationCompletionValidationError, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, ImageGenerationSettings, ImageGenerationValidationError,
    MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, MtpDepthStatus, MtpRuntimeState, ProtocolError,
    ProtocolReader, ProtocolWriter, RequestId, SpeculativePrefillRuntimeState, WorkerCommand,
    WorkerEvent, WorkerMlxMemorySnapshot, WorkerModelCapabilities,
    WorkerModelCapabilitiesValidationError, decode_command, decode_event, encode_event,
};
use image::{
    ExtendedColorType, ImageEncoder,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use tokio::io::duplex;

mod png_validation;
mod protocol_contracts;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_round_trip_an_image_generation_journey() {
    let request_id = RequestId::new(501);
    let image_generation_command = ImageGenerationCommand {
        request_id,
        model: "fictional/image-model".to_owned(),
        prompt: "A brass astrolabe under a clear night sky".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: 1_024,
            height_pixels: 768,
            steps: 28,
            guidance_thousandths: 3_500,
            seed: 42,
        },
    };
    let worker_command = WorkerCommand::GenerateImage(image_generation_command.clone());
    let worker_events = [
        WorkerEvent::ImageGenerationProgress {
            request_id,
            phase: ImageGenerationPhase::Denoising,
            completed_steps: 14,
            total_steps: 28,
            elapsed_millis: 900,
        },
        WorkerEvent::ImageGenerationCompleted {
            request_id,
            generated_image: GeneratedImage {
                mime_type: "image/png".to_owned(),
                encoded_bytes: valid_png_bytes(1_024, 768),
            },
            result_metadata: ImageGenerationResultMetadata {
                width_pixels: 1_024,
                height_pixels: 768,
                steps: 28,
                guidance_thousandths: 3_500,
                seed: 42,
                elapsed_millis: 1_800,
            },
        },
        WorkerEvent::ImageGenerationFinalized {
            request_id,
            elapsed_millis: 1_825,
            mlx_memory_snapshot: Some(finalized_image_memory_snapshot()),
        },
    ];
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let (supervisor_reader_transport, supervisor_writer_transport) =
        tokio::io::split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = tokio::io::split(worker_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut worker_reader = ProtocolReader::new(worker_reader_transport);
    let mut worker_writer = ProtocolWriter::new(worker_writer_transport);

    supervisor_writer
        .send_command(&worker_command)
        .await
        .expect("the image command should be written");
    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("the image command should decode"),
        Some(worker_command)
    );

    for worker_event in worker_events {
        worker_writer
            .send_event(&worker_event)
            .await
            .expect("the image generation event should be written");
        assert_eq!(
            supervisor_reader
                .next_event()
                .await
                .expect("the image generation event should decode"),
            Some(worker_event)
        );
    }

    assert_eq!(image_generation_command.validate(), Ok(()));
}

#[tokio::test]
async fn should_round_trip_an_image_generation_failure_event() {
    let worker_event = WorkerEvent::ImageGenerationFailed {
        request_id: RequestId::new(503),
        reason: ImageGenerationFailureReason::invalid_request(
            "requested dimensions exceed the loaded model capability",
        ),
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("the image failure event should be written");
    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the image failure event should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_round_trip_an_image_only_model_swap_capability() {
    let image_capabilities = image_generation_capabilities();
    let worker_event = WorkerEvent::ModelSwapped {
        model_id: "fictional/image-model".to_owned(),
        capabilities: WorkerModelCapabilities::image_generation(image_capabilities),
        expert_memory_mode: None,
        minimum_mlx_memory_ceiling_bytes: 1_000_000_000,
        mtp_runtime_state: MtpRuntimeState::Disabled,
        mtp_unavailable_reason: None,
        mtp_depth_status: MtpDepthStatus::EMPTY,
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
        speculative_prefill_unavailable_reason: None,
        speculative_prefill_draft_model_id: None,
        speculative_prefill_draft_model_revision: None,
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("the image-only model swap should be written");
    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the image-only model swap should decode"),
        Some(worker_event)
    );
}

#[test]
fn should_reject_invalid_image_completion_metadata_at_the_wire_boundary() {
    let worker_event = image_completion_event(
        valid_png_bytes(1_024, 768),
        ImageGenerationResultMetadata {
            guidance_thousandths: 100_001,
            ..valid_image_result_metadata()
        },
    );

    assert!(matches!(
        decode_event(&encode_event(&worker_event).expect("event should serialize")),
        Err(ProtocolError::InvalidImageGenerationCompletion(
            ImageGenerationCompletionValidationError::InvalidMetadata(
                ImageGenerationValidationError::GuidanceOutOfRange { .. }
            )
        ))
    ));
}

#[test]
fn should_reject_crc_corrupt_png_at_the_wire_boundary() {
    let mut crc_corrupt_png = valid_png_bytes(1_024, 768);
    let idat_range = png_chunk_range(&crc_corrupt_png, b"IDAT");
    crc_corrupt_png[idat_range.end - 1] ^= 0xff;

    assert_invalid_png_encoding(crc_corrupt_png);
}

#[test]
fn should_reject_png_without_idat_at_the_wire_boundary() {
    let mut png_without_idat = valid_png_bytes(1_024, 768);
    let idat_range = png_chunk_range(&png_without_idat, b"IDAT");
    png_without_idat.drain(idat_range);

    assert_invalid_png_encoding(png_without_idat);
}

#[test]
fn should_reject_non_rgb8_png_at_the_wire_boundary() {
    let rgba_png = encode_solid_png(1_024, 768, ExtendedColorType::Rgba8, 4);
    let worker_event = image_completion_event(rgba_png, valid_image_result_metadata());

    assert!(matches!(
        decode_event(&encode_event(&worker_event).expect("event should serialize")),
        Err(ProtocolError::InvalidImageGenerationCompletion(
            ImageGenerationCompletionValidationError::NonRgb8Png
        ))
    ));
}

#[test]
fn should_reject_invalid_mime_type_without_echoing_worker_content() {
    let worker_event = WorkerEvent::ImageGenerationCompleted {
        request_id: RequestId::new(501),
        generated_image: GeneratedImage {
            mime_type: "file:///private/image.png".to_owned(),
            encoded_bytes: valid_png_bytes(1_024, 768),
        },
        result_metadata: valid_image_result_metadata(),
    };

    let protocol_error =
        decode_event(&encode_event(&worker_event).expect("event should serialize"))
            .expect_err("the MIME type should be rejected");
    assert!(matches!(
        &protocol_error,
        ProtocolError::InvalidImageGenerationCompletion(
            ImageGenerationCompletionValidationError::InvalidMimeType
        )
    ));
    assert!(!protocol_error.to_string().contains("private"));
}

#[test]
fn should_reject_oversized_frames_before_png_decoding() {
    let oversized_frame = vec![0; MAX_IPC_FRAME_BYTES + 1];

    assert!(matches!(
        decode_event(&oversized_frame),
        Err(ProtocolError::IncomingMessageTooLarge { .. })
    ));
}

#[test]
fn should_reject_png_dimensions_that_do_not_match_completion_metadata() {
    let worker_event =
        image_completion_event(valid_png_bytes(512, 768), valid_image_result_metadata());

    assert!(matches!(
        decode_event(&encode_event(&worker_event).expect("event should serialize")),
        Err(ProtocolError::InvalidImageGenerationCompletion(
            ImageGenerationCompletionValidationError::PngDimensionsMismatch {
                encoded_width_pixels: 512,
                encoded_height_pixels: 768,
                metadata_width_pixels: 1_024,
                metadata_height_pixels: 768,
            }
        ))
    ));
}

#[test]
fn should_represent_chat_and_image_capabilities_independently() {
    let chat_capabilities = ChatModelCapabilities {
        supports_reasoning: true,
        supports_tool_calls: true,
        has_vision: false,
        max_input_tokens: 8_000,
        max_output_tokens: 1_000,
        context_window: 9_000,
    };
    let image_capabilities = image_generation_capabilities();

    assert_eq!(
        WorkerModelCapabilities::from(chat_capabilities.clone()),
        WorkerModelCapabilities {
            chat: Some(chat_capabilities.clone()),
            image_generation: None,
        }
    );
    assert_eq!(
        WorkerModelCapabilities::image_generation(image_capabilities.clone()),
        WorkerModelCapabilities {
            chat: None,
            image_generation: Some(image_capabilities.clone()),
        }
    );
    assert_eq!(
        WorkerModelCapabilities::chat_and_image(
            chat_capabilities.clone(),
            image_capabilities.clone()
        ),
        WorkerModelCapabilities {
            chat: Some(chat_capabilities),
            image_generation: Some(image_capabilities),
        }
    );
}

#[test]
fn should_round_trip_every_image_failure_reason() {
    let failure_reasons = [
        ImageGenerationFailureReason::invalid_request("width exceeds model capability"),
        ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration,
        ImageGenerationFailureReason::EngineBusy,
        ImageGenerationFailureReason::EncodingFailed {
            reason: "PNG encoder rejected the generated pixels".to_owned(),
        },
        ImageGenerationFailureReason::FatalExecution {
            reason: "image execution stopped".to_owned(),
        },
        ImageGenerationFailureReason::Cancelled,
    ];

    for failure_reason in failure_reasons {
        let serialized_failure =
            serde_json::to_string(&failure_reason).expect("image failure should serialize");
        assert_eq!(
            serde_json::from_str::<ImageGenerationFailureReason>(&serialized_failure)
                .expect("image failure should deserialize"),
            failure_reason
        );
    }
}

#[test]
fn should_summarize_image_events_without_image_or_prompt_payloads() {
    let request_id = RequestId::new(777);
    let events_and_summaries = [
        (
            WorkerEvent::ImageGenerationProgress {
                request_id,
                phase: ImageGenerationPhase::Decoding,
                completed_steps: 28,
                total_steps: 28,
                elapsed_millis: 1_700,
            },
            "image_generation_progress request_id=777",
        ),
        (
            WorkerEvent::ImageGenerationCompleted {
                request_id,
                generated_image: GeneratedImage {
                    mime_type: "image/png".to_owned(),
                    encoded_bytes: valid_png_bytes(64, 64),
                },
                result_metadata: ImageGenerationResultMetadata {
                    width_pixels: 64,
                    height_pixels: 64,
                    steps: 1,
                    guidance_thousandths: 1_000,
                    seed: 9,
                    elapsed_millis: 5,
                },
            },
            "image_generation_completed request_id=777",
        ),
        (
            WorkerEvent::ImageGenerationFailed {
                request_id,
                reason: ImageGenerationFailureReason::Cancelled,
            },
            "image_generation_failed request_id=777",
        ),
        (
            WorkerEvent::ImageGenerationFinalized {
                request_id,
                elapsed_millis: 1_725,
                mlx_memory_snapshot: Some(finalized_image_memory_snapshot()),
            },
            "image_generation_finalized request_id=777",
        ),
    ];

    for (worker_event, expected_summary) in events_and_summaries {
        assert_eq!(worker_event.diagnostic_summary(), expected_summary);
    }
}

fn finalized_image_memory_snapshot() -> WorkerMlxMemorySnapshot {
    WorkerMlxMemorySnapshot {
        source: MlxMemorySnapshotSource::Finalized,
        active_memory_bytes: 96_000_000,
        allocator_cache_memory_bytes: 0,
        peak_memory_bytes: 512_000_000,
        expert_payload_bytes: 0,
        model_core_payload_bytes: 0,
        context_state_payload_bytes: 0,
        speculative_prefill_draft_memory_bytes: 0,
    }
}

fn image_generation_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        minimum_width_pixels: 64,
        maximum_width_pixels: 2_048,
        minimum_height_pixels: 64,
        maximum_height_pixels: 2_048,
        dimension_multiple_pixels: 8,
        maximum_steps: 50,
        maximum_guidance_thousandths: 20_000,
        output_mime_types: vec!["image/png".to_owned()],
    }
}

fn valid_image_generation_command() -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(502),
        model: "fictional/image-model".to_owned(),
        prompt: "A public-domain astronomical engraving".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: 1_024,
            height_pixels: 768,
            steps: 28,
            guidance_thousandths: 3_500,
            seed: 42,
        },
    }
}

fn valid_image_result_metadata() -> ImageGenerationResultMetadata {
    ImageGenerationResultMetadata {
        width_pixels: 1_024,
        height_pixels: 768,
        steps: 4,
        guidance_thousandths: 1_000,
        seed: 42,
        elapsed_millis: 1_800,
    }
}

fn image_completion_event(
    encoded_bytes: Vec<u8>,
    result_metadata: ImageGenerationResultMetadata,
) -> WorkerEvent {
    WorkerEvent::ImageGenerationCompleted {
        request_id: RequestId::new(501),
        generated_image: png_image(encoded_bytes),
        result_metadata,
    }
}

fn png_image(encoded_bytes: Vec<u8>) -> GeneratedImage {
    GeneratedImage {
        mime_type: "image/png".to_owned(),
        encoded_bytes,
    }
}

fn valid_png_bytes(width_pixels: u32, height_pixels: u32) -> Vec<u8> {
    encode_solid_png(width_pixels, height_pixels, ExtendedColorType::Rgb8, 3)
}

fn encode_solid_png(
    width_pixels: u32,
    height_pixels: u32,
    color_type: ExtendedColorType,
    channel_count: usize,
) -> Vec<u8> {
    let pixel_bytes = vec![0; width_pixels as usize * height_pixels as usize * channel_count];
    let mut png_bytes = Vec::new();
    PngEncoder::new_with_quality(
        Cursor::new(&mut png_bytes),
        CompressionType::Best,
        FilterType::Adaptive,
    )
    .write_image(&pixel_bytes, width_pixels, height_pixels, color_type)
    .expect("the test image should encode as PNG");
    png_bytes
}

fn png_chunk_range(png_bytes: &[u8], expected_chunk_type: &[u8; 4]) -> Range<usize> {
    let mut chunk_start = 8;
    while chunk_start < png_bytes.len() {
        let length_end = chunk_start + 4;
        let chunk_data_bytes = u32::from_be_bytes(
            png_bytes[chunk_start..length_end]
                .try_into()
                .expect("the encoded PNG should contain a chunk length"),
        ) as usize;
        let chunk_end = chunk_start + 12 + chunk_data_bytes;
        if &png_bytes[length_end..length_end + 4] == expected_chunk_type {
            return chunk_start..chunk_end;
        }
        chunk_start = chunk_end;
    }
    panic!("the encoded PNG should contain the requested chunk")
}

fn assert_invalid_png_encoding(encoded_bytes: Vec<u8>) {
    let worker_event = image_completion_event(encoded_bytes, valid_image_result_metadata());
    assert!(matches!(
        decode_event(&encode_event(&worker_event).expect("event should serialize")),
        Err(ProtocolError::InvalidImageGenerationCompletion(
            ImageGenerationCompletionValidationError::InvalidPngEncoding
        ))
    ));
}

fn model_swapped_event_json(capabilities: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "kind": "model_swapped",
        "model_id": "fictional/image-model",
        "capabilities": capabilities,
        "expert_memory_mode": null,
        "minimum_mlx_memory_ceiling_bytes": 1,
        "mtp_runtime_state": "disabled",
        "mtp_unavailable_reason": null,
        "mtp_depth_status": {
            "configured_draft_depth": null,
            "artifact_maximum_draft_depth": null,
            "artifact_default_draft_depth": null,
            "resolved_requested_draft_depth": null,
            "effective_execution_draft_depth": null
        },
        "speculative_prefill_runtime_state": "disabled",
        "speculative_prefill_unavailable_reason": null,
        "speculative_prefill_draft_model_id": null,
        "speculative_prefill_draft_model_revision": null
    })
}
