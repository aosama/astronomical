use astronomical_rest_contract::{
    OpenAiGeneratedImageParts, OpenAiImageGenerationRequest, OpenAiImageGenerationResponse,
    OpenAiImageGenerationResponseFormat, OpenAiImageGenerationValidationError,
};

#[test]
fn should_validate_a_complete_image_generation_request_into_effective_parts() {
    let request = deserialize_request(
        r#"{
            "model":"black-forest-labs/FLUX.2-klein-4B",
            "prompt":"A moonlit balcony scene from Romeo and Juliet.",
            "seed":18446744073709551615,
            "width":1024,
            "height":768,
            "steps":4,
            "guidance":1.0,
            "response_format":"b64_json",
            "n":1
        }"#,
    );

    let request_parts = request
        .into_parts()
        .expect("the supported image request should validate");

    assert_eq!(request_parts.model, "black-forest-labs/FLUX.2-klein-4B");
    assert_eq!(
        request_parts.prompt,
        "A moonlit balcony scene from Romeo and Juliet."
    );
    assert_eq!(request_parts.seed, Some(u64::MAX));
    assert_eq!(request_parts.width, 1_024);
    assert_eq!(request_parts.height, 768);
    assert_eq!(request_parts.steps, 4);
    assert_eq!(request_parts.guidance, 1.0);
    assert_eq!(
        request_parts.response_format,
        OpenAiImageGenerationResponseFormat::Base64Json
    );
    assert_eq!(request_parts.image_count, 1);
}

#[test]
fn should_default_the_image_count_and_optional_seed() {
    let request = deserialize_request(&valid_request_fields("64", "64", "4", "1.0"));

    let request_parts = request
        .into_parts()
        .expect("the initial image count should default to one");

    assert_eq!(request_parts.image_count, 1);
    assert_eq!(request_parts.seed, None);
}

#[test]
fn should_reject_blank_or_non_string_prompts() {
    let blank_prompt_request = deserialize_request(
        r#"{"model":"flux","prompt":" \n\t","width":64,"height":64,"steps":4,"guidance":1.0,"response_format":"b64_json"}"#,
    );
    assert_eq!(
        blank_prompt_request.into_parts(),
        Err(OpenAiImageGenerationValidationError::BlankPrompt)
    );

    assert_malformed_request(
        r#"{"model":"flux","prompt":7,"width":64,"height":64,"steps":4,"guidance":1.0,"response_format":"b64_json"}"#,
    );
}

#[test]
fn should_reject_dimensions_outside_the_supported_geometry() {
    let invalid_dimensions = [
        ("0", "64", "width", 0),
        ("48", "64", "width", 48),
        ("65", "64", "width", 65),
        ("64", "1040", "height", 1_040),
    ];

    for (width, height, parameter_name, actual_pixels) in invalid_dimensions {
        let request = deserialize_request(&valid_request_fields(width, height, "4", "1.0"));
        assert_eq!(
            request.into_parts(),
            Err(OpenAiImageGenerationValidationError::UnsupportedDimension {
                parameter_name,
                actual_pixels,
                minimum_pixels: 64,
                maximum_pixels: 1_024,
            })
        );
    }
}

#[test]
fn should_reject_malformed_dimension_and_step_transport_values() {
    for malformed_fields in [
        valid_request_fields("64.5", "64", "4", "1.0"),
        valid_request_fields("-64", "64", "4", "1.0"),
        valid_request_fields("4294967296", "64", "4", "1.0"),
        valid_request_fields("64", "64", "4.5", "1.0"),
        valid_request_fields("64", "64", "4294967296", "1.0"),
    ] {
        assert_malformed_request(&malformed_fields);
    }
}

#[test]
fn should_reject_unsupported_steps_and_guidance() {
    let unsupported_steps = deserialize_request(&valid_request_fields("64", "64", "5", "1.0"));
    assert_eq!(
        unsupported_steps.into_parts(),
        Err(OpenAiImageGenerationValidationError::UnsupportedStepCount { actual_steps: 5 })
    );

    let unsupported_guidance = deserialize_request(&valid_request_fields("64", "64", "4", "0.0"));
    assert_eq!(
        unsupported_guidance.into_parts(),
        Err(OpenAiImageGenerationValidationError::UnsupportedGuidance {
            actual_guidance: 0.0,
        })
    );
}

#[test]
fn should_reject_malformed_guidance_and_seed_transport_values() {
    assert_malformed_request(&valid_request_fields("64", "64", "4", "\"1.0\""));

    for malformed_seed in ["-1", "1.5", "18446744073709551616", "\"7\""] {
        let request_json = format!(
            r#"{{"model":"flux","prompt":"A rose.","seed":{malformed_seed},"width":64,"height":64,"steps":4,"guidance":1.0,"response_format":"b64_json"}}"#
        );
        assert_malformed_request(&request_json);
    }
}

#[test]
fn should_reject_unsupported_format_count_and_unknown_fields() {
    let invalid_requests = [
        (
            r#"{"model":"flux","prompt":"A rose.","width":64,"height":64,"steps":4,"guidance":1.0,"response_format":"url"}"#,
            OpenAiImageGenerationValidationError::UnsupportedResponseFormat {
                response_format: "url".to_owned(),
            },
        ),
        (
            r#"{"model":"flux","prompt":"A rose.","width":64,"height":64,"steps":4,"guidance":1.0,"response_format":"b64_json","n":2}"#,
            OpenAiImageGenerationValidationError::UnsupportedImageCount { actual_images: 2 },
        ),
        (
            r#"{"model":"flux","prompt":"A rose.","width":64,"height":64,"steps":4,"guidance":1.0,"response_format":"b64_json","quality":"hd"}"#,
            OpenAiImageGenerationValidationError::UnknownField {
                field_name: "quality".to_owned(),
            },
        ),
    ];

    for (request_json, expected_error) in invalid_requests {
        assert_eq!(
            deserialize_request(request_json).into_parts(),
            Err(expected_error)
        );
    }
}

#[test]
fn should_serialize_one_generated_image_with_reproducibility_metadata() {
    let response = OpenAiImageGenerationResponse::new(
        1_787_010_400,
        OpenAiGeneratedImageParts {
            b64_json: "iVBORw0KGgoAAAANSUhEUg==".to_owned(),
            mime_type: "image/png".to_owned(),
            model_revision: "0123456789abcdef".to_owned(),
            effective_seed: u64::MAX,
            width: 1_024,
            height: 768,
        },
    );

    assert_eq!(
        serde_json::to_string(&response).expect("the generated image response should serialize"),
        r#"{"created":1787010400,"data":[{"b64_json":"iVBORw0KGgoAAAANSUhEUg==","mime_type":"image/png","model_revision":"0123456789abcdef","seed":18446744073709551615,"width":1024,"height":768}]}"#
    );
}

fn deserialize_request(request_json: &str) -> OpenAiImageGenerationRequest {
    serde_json::from_str(request_json)
        .expect("the structurally valid image request should deserialize")
}

fn assert_malformed_request(request_json: &str) {
    serde_json::from_str::<OpenAiImageGenerationRequest>(request_json)
        .expect_err("the malformed transport value must be rejected during deserialization");
}

fn valid_request_fields(width: &str, height: &str, steps: &str, guidance: &str) -> String {
    format!(
        r#"{{"model":"flux","prompt":"A rose.","width":{width},"height":{height},"steps":{steps},"guidance":{guidance},"response_format":"b64_json"}}"#
    )
}
