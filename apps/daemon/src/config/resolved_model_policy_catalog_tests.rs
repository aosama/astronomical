//! Unit contracts for how a discovered model's family is named to the worker.

use std::path::PathBuf;

use config::{
    DecisionModelCapabilities, DiscoveredModel, ImageGenerationCapabilities, ModelCapabilities,
    ModelFamily,
};
use ipc_protocol::WorkerModelConfiguration;

use super::ResolvedModelPolicyCatalog;

fn discovered_model(model_family: ModelFamily, capabilities: ModelCapabilities) -> DiscoveredModel {
    DiscoveredModel {
        model_id: "fictional-model".to_owned(),
        provider_model_id: None,
        model_family,
        revision: "fictional-revision".to_owned(),
        model_directory: PathBuf::from("fictional-model"),
        capabilities,
        license: None,
        model_size_bytes: 1,
    }
}

fn decision_capabilities() -> DecisionModelCapabilities {
    DecisionModelCapabilities {
        maximum_tokens_per_question: 512,
    }
}

fn image_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        supports_text_to_image: true,
        supports_image_editing: false,
        supports_multiple_reference_images: false,
        default_steps: 4,
        minimum_dimension_pixels: 256,
        maximum_dimension_pixels: 1_024,
        dimension_multiple_pixels: 16,
    }
}

fn decision_family_sent_to_worker(model_family: ModelFamily) -> String {
    let decision_model = discovered_model(
        model_family,
        ModelCapabilities::Decisions(decision_capabilities()),
    );
    match ResolvedModelPolicyCatalog::decisions_policy(&decision_model, &decision_capabilities())
        .worker_model_configuration
    {
        WorkerModelConfiguration::Decisions(decision_configuration) => {
            decision_configuration.model_family
        }
        other_configuration => panic!("a decision policy was {other_configuration:?}"),
    }
}

#[test]
fn should_name_each_decision_family_by_its_identifier_to_the_worker() {
    for (model_family, expected_identifier) in [
        (config::KEV_MODEL_FAMILY, "kev"),
        (config::CLEF_MODEL_FAMILY, "clef"),
        (config::LAYA_MODEL_FAMILY, "laya"),
    ] {
        assert_eq!(
            decision_family_sent_to_worker(model_family),
            expected_identifier
        );
    }
}

#[test]
fn should_name_a_family_without_a_decision_engine_as_itself_for_the_worker_to_refuse() {
    assert_eq!(
        decision_family_sent_to_worker(config::QWEN3_5_MODEL_FAMILY),
        "qwen3_5",
        "only Kev, Clef and Laya folders report a decision capability; the worker refuses any other family at load"
    );
}

#[test]
fn should_send_each_image_family_as_one_image_configuration_naming_the_family() {
    for (model_family, expected_identifier) in [
        (config::FLUX2_KLEIN_MODEL_FAMILY, "flux2_klein"),
        (config::QWEN_IMAGE_21_MODEL_FAMILY, "qwen_image_21"),
    ] {
        let image_model = discovered_model(
            model_family,
            ModelCapabilities::ImageGeneration(image_capabilities()),
        );

        match ResolvedModelPolicyCatalog::image_policy(&image_model).worker_model_configuration {
            WorkerModelConfiguration::ImageGeneration(image_configuration) => {
                assert_eq!(image_configuration.model_family, expected_identifier);
                assert_eq!(image_configuration.artifact_revision, "fictional-revision");
            }
            other_configuration => panic!("an image policy was {other_configuration:?}"),
        }
    }
}
