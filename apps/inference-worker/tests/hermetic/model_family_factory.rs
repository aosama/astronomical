//! Hermetic contracts for selecting a runtime without loading model artifacts.

use astronomical_config::{PromptCacheConfig, discover_models};
use astronomical_inference_worker::model_family_factory::ModelFamilyFactory;
use astronomical_ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerFlux2KleinModelConfiguration, WorkerImageGenerationModelFamily, WorkerModelConfiguration,
};
use astronomical_model_serving::{ModelFactory, ModelFactoryRuntime};
use serde_json::json;

use super::flux2_klein_fixture::{
    CANONICAL_MODEL_ID, REVIEWED_REVISION, replace_json_field, write_executable_artifact,
    write_revision,
};

#[tokio::test]
async fn should_create_a_lazy_flux_image_runtime_for_the_pinned_configuration() {
    let model_directory = flux_model_directory();
    let factory = model_family_factory(model_directory.path());

    let factory_runtime = factory
        .create(
            model_directory
                .path()
                .to_str()
                .expect("the temporary model path should be UTF-8"),
            flux_configuration(CANONICAL_MODEL_ID, REVIEWED_REVISION),
        )
        .await
        .expect("the pinned FLUX configuration should select an image runtime");

    let ModelFactoryRuntime::Image(image_engine) = factory_runtime else {
        panic!("FLUX must not receive a fabricated autoregressive processor");
    };
    assert_eq!(image_engine.loaded_revision(), None);
}

#[tokio::test]
async fn should_reject_an_autoregressive_configuration_for_a_flux_family() {
    let model_directory = flux_model_directory();
    let factory = model_family_factory(model_directory.path());

    let factory_outcome = factory
        .create(
            model_directory
                .path()
                .to_str()
                .expect("the temporary model path should be UTF-8"),
            autoregressive_configuration(),
        )
        .await;
    let Err(load_failure_reason) = factory_outcome else {
        panic!("a family and configuration modality mismatch must fail before loading");
    };

    assert_eq!(
        load_failure_reason,
        "selected model configuration does not match its classified model family"
    );
}

#[tokio::test]
async fn should_reject_unpinned_flux_identity_before_model_loading() {
    let model_directory = flux_model_directory();
    let factory = model_family_factory(model_directory.path());

    for model_configuration in [
        flux_configuration("different-model", REVIEWED_REVISION),
        flux_configuration(CANONICAL_MODEL_ID, "different-revision"),
    ] {
        let factory_outcome = factory
            .create(
                model_directory
                    .path()
                    .to_str()
                    .expect("the temporary model path should be UTF-8"),
                model_configuration,
            )
            .await;
        let Err(load_failure_reason) = factory_outcome else {
            panic!("unreviewed FLUX provenance must fail before loading");
        };
        assert_eq!(
            load_failure_reason,
            "selected FLUX.2 Klein model identity or revision is unsupported"
        );
    }
}

#[tokio::test]
async fn should_reject_changed_evidence_between_supervisor_discovery_and_worker_load() {
    for mutate_discovered_artifact in [
        mutate_revision as fn(&std::path::Path),
        mutate_license,
        mutate_profile,
        remove_component,
    ] {
        let model_directory = flux_model_directory();
        let discovered_model = discover_models(&[model_directory.path().to_path_buf()])
            .expect("supervisor discovery should complete")
            .remove(0)
            .discovered_models
            .remove(0);
        mutate_discovered_artifact(model_directory.path());

        let factory_outcome = model_family_factory(model_directory.path())
            .create(
                model_directory
                    .path()
                    .to_str()
                    .expect("the temporary model path should be UTF-8"),
                flux_configuration(&discovered_model.model_id, &discovered_model.revision),
            )
            .await;
        let Err(load_failure_reason) = factory_outcome else {
            panic!("changed selected-directory evidence must fail before engine construction");
        };
        assert_eq!(
            load_failure_reason,
            "selected FLUX.2 Klein artifact failed exact-directory verification"
        );
        assert!(
            !load_failure_reason.contains(
                model_directory
                    .path()
                    .to_str()
                    .expect("temporary path should be UTF-8")
            )
        );
    }
}

fn model_family_factory(fixture_directory: &std::path::Path) -> ModelFamilyFactory {
    ModelFamilyFactory::new(
        2_000_000_000,
        200_000_000,
        PromptCacheConfig::new(fixture_directory.join("prompt-cache"), 1_000_000_000),
        true,
        fixture_directory.join("performance-attribution.jsonl"),
        true,
    )
}

fn flux_model_directory() -> tempfile::TempDir {
    let model_directory =
        tempfile::tempdir().expect("the FLUX classification fixture should exist");
    write_executable_artifact(model_directory.path());
    model_directory
}

fn flux_configuration(model_id: &str, artifact_revision: &str) -> WorkerModelConfiguration {
    WorkerModelConfiguration::Flux2Klein(WorkerFlux2KleinModelConfiguration {
        model_id: model_id.to_owned(),
        model_family: WorkerImageGenerationModelFamily::Flux2Klein,
        artifact_revision: artifact_revision.to_owned(),
    })
}

fn autoregressive_configuration() -> WorkerModelConfiguration {
    WorkerModelConfiguration::Autoregressive(WorkerAutoregressiveModelConfiguration {
        model_id: CANONICAL_MODEL_ID.to_owned(),
        maximum_context_tokens: 4_096,
        maximum_output_tokens: 1_024,
        chunking: WorkerChunkingConfiguration {
            fixed_prompt_processing_chunk_size_tokens: 1_024,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 2_048,
            full_attention_key_value_growth_tokens: 256,
            speculative_prefill_draft_forward_tokens: 1_024,
            prefill_graph_submission_layer_interval: 0,
            experimental_ssd_paging_prefill_graph_submission_layer_interval: 1,
            experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
            prompt_cache_block_tokens: None,
            prompt_cache_common_prefix_stride_blocks: 4,
        },
        mtp_enabled: true,
        mtp_draft_depth: None,
        speculative_prefill: None,
    })
}

fn mutate_revision(model_directory: &std::path::Path) {
    write_revision(model_directory, "changed-revision");
}

fn mutate_license(model_directory: &std::path::Path) {
    std::fs::write(
        model_directory.join("LICENSE.md"),
        "Fictional model license",
    )
    .expect("changed license should be written");
}

fn mutate_profile(model_directory: &std::path::Path) {
    replace_json_field(
        &model_directory.join("text_encoder/config.json"),
        "dtype",
        json!("float16"),
    );
}

fn remove_component(model_directory: &std::path::Path) {
    std::fs::remove_file(model_directory.join("vae/diffusion_pytorch_model.safetensors"))
        .expect("required component should be removed");
}
