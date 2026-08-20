//! Request, serialization, and capability contracts for image IPC.

use astronomical_ipc_protocol::{
    GeneratedImage, ImageGenerationCommand, ImageGenerationSettings,
    ImageGenerationValidationError, ProtocolError, WorkerCommand,
    WorkerModelCapabilitiesValidationError, decode_command, decode_event,
};

use super::{model_swapped_event_json, valid_image_generation_command, valid_png_bytes};

#[test]
fn should_serialize_generated_image_bytes_as_base64_json() {
    let generated_image = GeneratedImage {
        mime_type: "image/png".to_owned(),
        encoded_bytes: valid_png_bytes(1, 1),
    };

    let serialized_image =
        serde_json::to_string(&generated_image).expect("generated image should serialize");
    let serialized_image_value: serde_json::Value =
        serde_json::from_str(&serialized_image).expect("generated image JSON should parse");
    assert!(serialized_image_value["encoded_bytes"].is_string());
    assert_eq!(
        serde_json::from_str::<GeneratedImage>(&serialized_image)
            .expect("generated image should deserialize"),
        generated_image
    );
}

#[test]
fn should_reject_invalid_image_generation_commands_and_settings() {
    let valid_command = valid_image_generation_command();
    let invalid_commands = [
        (
            ImageGenerationCommand {
                model: " \t".to_owned(),
                ..valid_command.clone()
            },
            ImageGenerationValidationError::EmptyModelId,
        ),
        (
            ImageGenerationCommand {
                prompt: " \n".to_owned(),
                ..valid_command.clone()
            },
            ImageGenerationValidationError::EmptyPrompt,
        ),
        (
            ImageGenerationCommand {
                settings: ImageGenerationSettings {
                    width_pixels: 63,
                    ..valid_command.settings
                },
                ..valid_command.clone()
            },
            ImageGenerationValidationError::WidthOutOfRange {
                actual_width_pixels: 63,
                minimum_width_pixels: 64,
                maximum_width_pixels: 16_384,
            },
        ),
        (
            ImageGenerationCommand {
                settings: ImageGenerationSettings {
                    width_pixels: 65,
                    ..valid_command.settings
                },
                ..valid_command.clone()
            },
            ImageGenerationValidationError::WidthNotAligned {
                actual_width_pixels: 65,
                required_multiple_pixels: 8,
            },
        ),
        (
            ImageGenerationCommand {
                settings: ImageGenerationSettings {
                    height_pixels: 16_385,
                    ..valid_command.settings
                },
                ..valid_command.clone()
            },
            ImageGenerationValidationError::HeightOutOfRange {
                actual_height_pixels: 16_385,
                minimum_height_pixels: 64,
                maximum_height_pixels: 16_384,
            },
        ),
        (
            ImageGenerationCommand {
                settings: ImageGenerationSettings {
                    height_pixels: 65,
                    ..valid_command.settings
                },
                ..valid_command.clone()
            },
            ImageGenerationValidationError::HeightNotAligned {
                actual_height_pixels: 65,
                required_multiple_pixels: 8,
            },
        ),
        (
            ImageGenerationCommand {
                settings: ImageGenerationSettings {
                    steps: 0,
                    ..valid_command.settings
                },
                ..valid_command.clone()
            },
            ImageGenerationValidationError::StepsOutOfRange {
                actual_steps: 0,
                minimum_steps: 1,
                maximum_steps: 1_000,
            },
        ),
        (
            ImageGenerationCommand {
                settings: ImageGenerationSettings {
                    guidance_thousandths: 100_001,
                    ..valid_command.settings
                },
                ..valid_command
            },
            ImageGenerationValidationError::GuidanceOutOfRange {
                actual_guidance_thousandths: 100_001,
                maximum_guidance_thousandths: 100_000,
            },
        ),
    ];

    for (invalid_command, expected_error) in invalid_commands {
        assert_eq!(invalid_command.validate(), Err(expected_error));
    }

    let maximum_seed_command = ImageGenerationCommand {
        settings: ImageGenerationSettings {
            seed: u64::MAX,
            ..valid_image_generation_command().settings
        },
        ..valid_image_generation_command()
    };
    assert_eq!(maximum_seed_command.validate(), Ok(()));
}

#[test]
fn should_reject_whitespace_only_image_inputs_before_worker_execution() {
    for (field_name, field_value) in [("model", "  "), ("prompt", "\t\n")] {
        let mut serialized_command = serde_json::to_value(WorkerCommand::GenerateImage(
            valid_image_generation_command(),
        ))
        .expect("image command should serialize");
        serialized_command[field_name] = serde_json::json!(field_value);

        let decoded_command = decode_command(
            &serde_json::to_vec(&serialized_command).expect("wire command should serialize"),
        )
        .expect("the transport should preserve a structurally valid command");
        let WorkerCommand::GenerateImage(image_command) = decoded_command else {
            panic!("the image command variant should survive transport");
        };
        assert!(image_command.validate().is_err());
    }
}

#[test]
fn should_reject_malformed_image_capabilities_at_the_wire_boundary() {
    let malformed_capabilities = [
        (
            serde_json::json!({
                "minimum_width_pixels": 64, "maximum_width_pixels": 1024,
                "minimum_height_pixels": 64, "maximum_height_pixels": 1024,
                "dimension_multiple_pixels": 0, "maximum_steps": 4,
                "maximum_guidance_thousandths": 1000,
                "output_mime_types": ["image/png"]
            }),
            WorkerModelCapabilitiesValidationError::ZeroImageDimensionAlignment,
        ),
        (
            serde_json::json!({
                "minimum_width_pixels": 1024, "maximum_width_pixels": 64,
                "minimum_height_pixels": 64, "maximum_height_pixels": 1024,
                "dimension_multiple_pixels": 16, "maximum_steps": 4,
                "maximum_guidance_thousandths": 1000,
                "output_mime_types": ["image/png"]
            }),
            WorkerModelCapabilitiesValidationError::InvertedImageWidthBounds {
                minimum_width_pixels: 1024,
                maximum_width_pixels: 64,
            },
        ),
        (
            serde_json::json!({
                "minimum_width_pixels": 64, "maximum_width_pixels": 1024,
                "minimum_height_pixels": 1024, "maximum_height_pixels": 64,
                "dimension_multiple_pixels": 16, "maximum_steps": 4,
                "maximum_guidance_thousandths": 1000,
                "output_mime_types": ["image/png"]
            }),
            WorkerModelCapabilitiesValidationError::InvertedImageHeightBounds {
                minimum_height_pixels: 1024,
                maximum_height_pixels: 64,
            },
        ),
        (
            serde_json::json!({
                "minimum_width_pixels": 64, "maximum_width_pixels": 1024,
                "minimum_height_pixels": 64, "maximum_height_pixels": 1024,
                "dimension_multiple_pixels": 16, "maximum_steps": 4,
                "maximum_guidance_thousandths": 1000,
                "output_mime_types": []
            }),
            WorkerModelCapabilitiesValidationError::EmptyImageOutputMimeType,
        ),
        (
            serde_json::json!({
                "minimum_width_pixels": 64, "maximum_width_pixels": 1024,
                "minimum_height_pixels": 64, "maximum_height_pixels": 1024,
                "dimension_multiple_pixels": 16, "maximum_steps": 4,
                "maximum_guidance_thousandths": 1000,
                "output_mime_types": ["  "]
            }),
            WorkerModelCapabilitiesValidationError::EmptyImageOutputMimeType,
        ),
    ];

    for (image_capabilities, expected_error) in malformed_capabilities {
        let serialized_event = model_swapped_event_json(serde_json::json!({
            "chat": null,
            "image_generation": image_capabilities
        }));
        assert!(matches!(
            decode_event(&serde_json::to_vec(&serialized_event).expect("event should serialize")),
            Err(ProtocolError::InvalidWorkerModelCapabilities(actual_error))
                if actual_error == expected_error
        ));
    }

    let serialized_event = model_swapped_event_json(serde_json::json!({
        "chat": null,
        "image_generation": null
    }));
    assert!(matches!(
        decode_event(&serde_json::to_vec(&serialized_event).expect("event should serialize")),
        Err(ProtocolError::InvalidWorkerModelCapabilities(
            WorkerModelCapabilitiesValidationError::NoCapabilities
        ))
    ));
}
