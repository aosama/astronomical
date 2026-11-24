//! Integration contracts for selecting the Qwen-Image-2.1 runtime without loading model artifacts.

use ipc_protocol::WorkerModelConfiguration;
use model_serving::{ModelFactory, ModelFactoryRuntime, QwenImage21ImageEngine};

use super::image_factory_support::{
    autoregressive_configuration, image_model_configuration, model_family_factory,
};
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
    let image_engine_as_any: &dyn std::any::Any = image_engine.as_ref();
    let Some(qwen_engine) = image_engine_as_any.downcast_ref::<QwenImage21ImageEngine>() else {
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
async fn should_reject_changed_evidence_between_daemon_discovery_and_worker_load() {
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
        let discovered_model = config::discover_models(&[model_directory.path().to_path_buf()])
            .expect("daemon discovery should complete")
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

fn qwen_model_directory() -> tempfile::TempDir {
    let model_directory =
        tempfile::tempdir().expect("the Qwen-Image-2.1 classification fixture should exist");
    write_executable_artifact(model_directory.path());
    model_directory
}

fn qwen_configuration(model_id: &str, artifact_revision: &str) -> WorkerModelConfiguration {
    image_model_configuration("qwen_image_21", model_id, artifact_revision)
}
