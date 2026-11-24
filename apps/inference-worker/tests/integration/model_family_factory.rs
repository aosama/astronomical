//! Integration contracts for selecting a runtime without loading model artifacts.

use ipc_protocol::WorkerModelConfiguration;
use model_serving::{Flux2KleinImageEngine, ModelFactory, ModelFactoryRuntime};
use serde_json::json;

use super::flux2_klein_fixture::{
    CANONICAL_MODEL_ID, REVIEWED_REVISION, replace_json_field, write_executable_artifact,
    write_revision,
};
use super::image_factory_support::{
    autoregressive_configuration, image_model_configuration, model_family_factory,
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
    let image_engine_as_any: &dyn std::any::Any = image_engine.as_ref();
    let Some(flux_engine) = image_engine_as_any.downcast_ref::<Flux2KleinImageEngine>() else {
        panic!("the FLUX configuration must select the FLUX image engine");
    };
    assert_eq!(flux_engine.loaded_revision(), None);
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
async fn should_reject_changed_evidence_between_daemon_discovery_and_worker_load() {
    // A mutated revision breaks the worker's config-vs-directory identity
    // check before artifact verification runs; the other mutations leave
    // provenance intact and are caught by exact-directory verification.
    for (mutate_discovered_artifact, expected_failure_reason) in [
        (
            mutate_revision as fn(&std::path::Path),
            "selected FLUX.2 Klein model identity or revision is unsupported",
        ),
        (
            mutate_license as fn(&std::path::Path),
            "selected FLUX.2 Klein artifact failed exact-directory verification",
        ),
        (
            mutate_profile as fn(&std::path::Path),
            "selected FLUX.2 Klein artifact failed exact-directory verification",
        ),
        (
            remove_component as fn(&std::path::Path),
            "selected FLUX.2 Klein artifact failed exact-directory verification",
        ),
    ] {
        let model_directory = flux_model_directory();
        let discovered_model = config::discover_models(&[model_directory.path().to_path_buf()])
            .expect("daemon discovery should complete")
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

fn flux_model_directory() -> tempfile::TempDir {
    let model_directory =
        tempfile::tempdir().expect("the FLUX classification fixture should exist");
    write_executable_artifact(model_directory.path());
    model_directory
}

fn flux_configuration(model_id: &str, artifact_revision: &str) -> WorkerModelConfiguration {
    image_model_configuration("flux2_klein", model_id, artifact_revision)
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
