//! Hermetic contracts for selecting the Qwen-Image-2.1 runtime without loading model artifacts.

use astronomical_config::{PromptCacheConfig, discover_models};
use astronomical_inference_worker::model_family_factory::ModelFamilyFactory;
use astronomical_ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerImageGenerationModelFamily, WorkerModelConfiguration,
    WorkerQwenImage21ModelConfiguration,
};
use astronomical_model_serving::{ModelFactory, ModelFactoryRuntime, ModelFamilyImageEngine};

use super::flux2_klein_fixture::CANONICAL_MODEL_ID as FLUX_CANONICAL_MODEL_ID;
use super::qwen_image_21_fixture::{
    CANONICAL_MODEL_ID, REVIEWED_REVISION, mutate_profile, remove_component,
    write_executable_artifact, write_revision,
};

#[tokio::test]
async fn should_create_a_lazy_qwen_image_runtime_for_the_pinned_configuration() {
    let model_directory = qwen_model_directory();
    let factory = model_family_factory(model_directory.path());

    let factory_runtime = factory
        .create(
            model_directory
                .path()
                .to_str()
                .expect("the temporary model path should be UTF-8"),
            qwen_configuration(CANONICAL_MODEL_ID, REVIEWED_REVISION),
        )
        .await
        .expect("the pinned Qwen-Image-2.1 configuration should select an image runtime");

    let ModelFactoryRuntime::Image(image_engine) = factory_runtime else {
        panic!("Qwen-Image-2.1 must not receive a fabricated autoregressive processor");
    };
    let ModelFamilyImageEngine::QwenImage21(qwen_engine) = image_engine else {
        panic!("the Qwen-Image-2.1 configuration must select the Qwen image engine");
    };
    assert_eq!(
        qwen_engine.loaded_revision(),
        None,
        "runtime selection must stay lazy: no artifact load happens before the worker binds"
    );
}

#[tokio::test]
async fn should_reject_an_autoregressive_configuration_for_a_qwen_family() {
    let model_directory = qwen_model_directory();
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
async fn should_reject_unpinned_qwen_identity_before_model_loading() {
    let model_directory = qwen_model_directory();
    let factory = model_family_factory(model_directory.path());

    for model_configuration in [
        qwen_configuration("different-model", REVIEWED_REVISION),
        qwen_configuration(CANONICAL_MODEL_ID, "different-revision"),
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
            panic!("unreviewed Qwen-Image-2.1 provenance must fail before loading");
        };
        assert_eq!(
            load_failure_reason,
            "selected Qwen-Image-2.1 model identity or revision is unsupported"
        );
    }
}

#[tokio::test]
async fn should_reject_changed_evidence_between_supervisor_discovery_and_worker_load() {
    // A mutated revision breaks the worker's config-vs-directory identity check before
    // artifact verification runs; the other mutations leave provenance intact and are caught
    // by exact-directory verification.
    for (mutate_discovered_artifact, expected_failure_reason) in [
        (
            mutate_revision as fn(&std::path::Path),
            "selected Qwen-Image-2.1 model identity or revision is unsupported",
        ),
        (
            mutate_profile as fn(&std::path::Path),
            "selected Qwen-Image-2.1 artifact failed exact-directory verification",
        ),
        (
            remove_component as fn(&std::path::Path),
            "selected Qwen-Image-2.1 artifact failed exact-directory verification",
        ),
    ] {
        let model_directory = qwen_model_directory();
        let discovered_model = discover_models(&[model_directory.path().to_path_buf()])
            .expect("supervisor discovery should complete")
            .remove(0)
            .discovered_models
            .remove(0);
        assert_eq!(discovered_model.model_id, CANONICAL_MODEL_ID);
        mutate_discovered_artifact(model_directory.path());

        let factory_outcome = model_family_factory(model_directory.path())
            .create(
                model_directory
                    .path()
                    .to_str()
                    .expect("the temporary model path should be UTF-8"),
                qwen_configuration(&discovered_model.model_id, &discovered_model.revision),
            )
            .await;
        let Err(load_failure_reason) = factory_outcome else {
            panic!("changed selected-directory evidence must fail before engine construction");
        };
        assert_eq!(load_failure_reason, expected_failure_reason);
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

fn mutate_revision(model_directory: &std::path::Path) {
    write_revision(model_directory, "ffffffffffffffffffffffffffffffffffffffff");
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

fn qwen_model_directory() -> tempfile::TempDir {
    let model_directory =
        tempfile::tempdir().expect("the Qwen-Image-2.1 classification fixture should exist");
    write_executable_artifact(model_directory.path());
    model_directory
}

fn qwen_configuration(model_id: &str, artifact_revision: &str) -> WorkerModelConfiguration {
    WorkerModelConfiguration::QwenImage21(WorkerQwenImage21ModelConfiguration {
        model_id: model_id.to_owned(),
        model_family: WorkerImageGenerationModelFamily::QwenImage21,
        artifact_revision: artifact_revision.to_owned(),
    })
}

fn autoregressive_configuration() -> WorkerModelConfiguration {
    WorkerModelConfiguration::Autoregressive(WorkerAutoregressiveModelConfiguration {
        model_id: FLUX_CANONICAL_MODEL_ID.to_owned(),
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
            experimental_decode_stage_attribution_enabled: false,
            experimental_quantized_kv_cache_enabled: false,
            experimental_fused_moe_decode_enabled: false,
        },
        mtp_enabled: false,
        mtp_draft_depth: None,
        speculative_prefill: None,
    })
}
