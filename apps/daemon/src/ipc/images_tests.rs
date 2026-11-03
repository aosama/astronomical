//! Unit tests for the pure steps of the IPC image handler: model matching,
//! command building, and outcome mapping. The journeys that cross the real
//! socket live in `tests/integration/daemon_ipc_images.rs`.

use astronomical_config::{
    DiscoveredModel, ImageGenerationCapabilities, ModelCapabilities, ModelFamily,
};
use astronomical_ipc_protocol::{
    DaemonResponse, GeneratedImage, ImageGenerateRequest, ImageGenerationFailureReason,
    ImageGenerationResultMetadata,
};
use std::path::PathBuf;

use super::{
    ImageTargetModel, build_image_command, find_discovered_model, image_outcome_to_response,
};
use crate::{ImageGenerationExecutionError, ImageGenerationOutput};

fn image_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        supports_text_to_image: true,
        supports_image_editing: false,
        supports_multiple_reference_images: false,
        default_steps: 4,
        minimum_dimension_pixels: 64,
        maximum_dimension_pixels: 1_024,
        dimension_multiple_pixels: 16,
    }
}

fn discovered_image_model(model_id: &str) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: None,
        model_family: ModelFamily::Flux2Klein,
        revision: "revision".to_owned(),
        model_directory: PathBuf::from("/tmp/models"),
        capabilities: ModelCapabilities::ImageGeneration(image_capabilities()),
        license: None,
        model_size_bytes: 1,
    }
}

fn target_model() -> ImageTargetModel {
    ImageTargetModel {
        resolved_model_id: "FLUX.2-klein-4B".to_owned(),
        image_capabilities: image_capabilities(),
    }
}

fn image_request(seed: Option<u64>) -> ImageGenerateRequest {
    ImageGenerateRequest {
        model: "FLUX.2-klein-4B".to_owned(),
        prompt: "a lighthouse at dusk".to_owned(),
        width_pixels: 1_024,
        height_pixels: 576,
        seed,
    }
}

fn completed_output() -> ImageGenerationOutput {
    ImageGenerationOutput {
        generated_image: GeneratedImage {
            mime_type: "image/png".to_owned(),
            encoded_bytes: vec![1, 2, 3],
        },
        result_metadata: ImageGenerationResultMetadata {
            width_pixels: 1_024,
            height_pixels: 576,
            steps: 4,
            guidance_thousandths: 1_000,
            seed: 7,
            elapsed_millis: 900,
        },
    }
}

#[test]
fn should_find_a_model_by_its_exact_id() {
    let discovered_models = [discovered_image_model("FLUX.2-klein-4B")];

    let found_model = find_discovered_model(&discovered_models, "FLUX.2-klein-4B")
        .expect("an exact id should resolve");

    assert_eq!(found_model.model_id, "FLUX.2-klein-4B");
}

#[test]
fn should_find_a_model_by_the_leaf_of_its_full_id() {
    let discovered_models = [discovered_image_model("FLUX.2-klein-4B")];

    let found_model =
        find_discovered_model(&discovered_models, "black-forest-labs/FLUX.2-klein-4B")
            .expect("the org-qualified id should resolve to the leaf");

    assert_eq!(found_model.model_id, "FLUX.2-klein-4B");
}

#[test]
fn should_name_the_near_matches_for_an_unknown_model() {
    let discovered_models = [discovered_image_model("FLUX.2-klein-4B")];

    let rejection_reason = find_discovered_model(&discovered_models, "FLUX.2-klein-4")
        .expect_err("a model that is not installed must not resolve");

    assert!(
        rejection_reason.contains("is unknown"),
        "{rejection_reason}"
    );
    // The shared suggestion helper lowercases what it suggests.
    assert!(
        rejection_reason
            .to_ascii_lowercase()
            .contains("did you mean: flux.2-klein-4b"),
        "{rejection_reason}"
    );
}

#[test]
fn should_reject_every_model_when_none_is_installed() {
    let rejection_reason =
        find_discovered_model(&[], "FLUX.2-klein-4B").expect_err("nothing is installed");

    assert!(
        rejection_reason.contains("astronomical models list"),
        "{rejection_reason}"
    );
}

#[test]
fn should_build_the_command_from_the_family_steps_and_the_requested_seed() {
    let image_command = build_image_command(9, &target_model(), image_request(Some(42)));

    assert_eq!(image_command.settings.steps, 4);
    assert_eq!(image_command.settings.guidance_thousandths, 1_000);
    assert_eq!(image_command.settings.seed, 42);
    assert_eq!(image_command.settings.width_pixels, 1_024);
    assert_eq!(image_command.settings.height_pixels, 576);
    assert_eq!(image_command.model, "FLUX.2-klein-4B");
}

#[test]
fn should_draw_a_seed_when_the_request_carries_none() {
    let first_command = build_image_command(1, &target_model(), image_request(None));
    let second_command = build_image_command(2, &target_model(), image_request(None));

    assert_ne!(first_command.settings.seed, second_command.settings.seed);
}

#[test]
fn should_keep_a_requested_seed_of_zero() {
    let image_command = build_image_command(5, &target_model(), image_request(Some(0)));

    assert_eq!(image_command.settings.seed, 0);
}

#[test]
fn should_complete_with_the_png_and_the_resolved_model() {
    let response = image_outcome_to_response(Ok(completed_output()), target_model());

    let DaemonResponse::ImageGenerationCompleted {
        model,
        image,
        metadata,
    } = response
    else {
        panic!("a worker image must become a completion frame");
    };
    assert_eq!(model, "FLUX.2-klein-4B");
    assert_eq!(image.encoded_bytes, vec![1, 2, 3]);
    assert_eq!(metadata.seed, 7);
}

#[test]
fn should_pass_a_worker_failure_through_unchanged() {
    let response = image_outcome_to_response(
        Err(ImageGenerationExecutionError::WorkerFailure(
            ImageGenerationFailureReason::EngineBusy,
        )),
        target_model(),
    );

    assert_eq!(
        response,
        DaemonResponse::ImageGenerationFailed {
            reason: ImageGenerationFailureReason::EngineBusy
        }
    );
}

#[test]
fn should_report_a_missed_deadline_as_a_failed_image() {
    let response = image_outcome_to_response(
        Err(ImageGenerationExecutionError::DeadlineExceeded),
        target_model(),
    );

    let DaemonResponse::ImageGenerationFailed { reason } = response else {
        panic!("a missed deadline is a failure after admission");
    };
    assert!(matches!(
        reason,
        ImageGenerationFailureReason::FatalExecution { .. }
    ));
}

#[test]
fn should_reject_when_the_worker_vanished_before_answering() {
    let response = image_outcome_to_response(
        Err(ImageGenerationExecutionError::WorkerUnavailable),
        target_model(),
    );

    assert_eq!(
        response,
        DaemonResponse::GenerationRejected {
            reason: "the worker is unavailable".to_owned()
        }
    );
}
