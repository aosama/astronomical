//! Beside-source coverage for the file-opening and config-parsing stages of artifact validation.
//! Each test writes small placeholder documents into a temporary directory; the weight-component
//! checks need full-size safetensors headers, so the integration suite of the serving crate
//! covers them with its synthetic artifact.

use std::collections::BTreeMap;
use std::path::Path;

use artifact_validation::ValidatedRequiredFile;
use test_harness::fixtures::image_pipeline_fixture::qwen_image_21_pipeline_index;

use super::{
    open_document_files, open_processor_sidecars, open_required_files, parse_component_configs,
};
use crate::artifact::{
    CONFIG_FILE_NAMES, INDEX_FILE_NAMES, MODEL_INDEX_FILE_NAME, PROCESSOR_SIDECAR_FILE_NAMES,
    QwenImage21ArtifactError,
};
use crate::configuration::QwenImage21ConfigError;

fn write_placeholder_files(root: &Path, relative_names: &[&str]) {
    for relative_name in relative_names {
        let file_path = root.join(relative_name);
        std::fs::create_dir_all(
            file_path
                .parent()
                .expect("every placeholder sits in a folder"),
        )
        .expect("the placeholder folder should be created");
        std::fs::write(file_path, b"{}").expect("the placeholder file should be written");
    }
}

fn all_document_names() -> Vec<&'static str> {
    [MODEL_INDEX_FILE_NAME]
        .into_iter()
        .chain(CONFIG_FILE_NAMES)
        .chain(INDEX_FILE_NAMES)
        .collect()
}

fn missing_file_name(rejection: QwenImage21ArtifactError) -> String {
    match rejection {
        QwenImage21ArtifactError::ArtifactFile { file_name } => file_name,
        other => panic!("expected an unavailable-file rejection, got {other}"),
    }
}

fn opened_documents_with_model_index(
    root: &Path,
    model_index_json: &str,
) -> BTreeMap<String, ValidatedRequiredFile> {
    write_placeholder_files(root, &all_document_names());
    std::fs::write(root.join(MODEL_INDEX_FILE_NAME), model_index_json)
        .expect("the model index should be written");
    open_document_files(root).expect("every placeholder document should open")
}

#[test]
fn should_key_every_opened_file_by_its_relative_name() {
    let root = tempfile::tempdir().expect("a temporary directory");
    write_placeholder_files(
        root.path(),
        &["text_encoder/config.json", "model_index.json"],
    );

    let opened_files = open_required_files(
        root.path(),
        ["text_encoder/config.json", "model_index.json"],
    )
    .expect("both files exist");

    let opened_names: Vec<&str> = opened_files.keys().map(String::as_str).collect();
    assert_eq!(
        opened_names,
        ["model_index.json", "text_encoder/config.json"]
    );
}

#[test]
fn should_open_nothing_when_given_no_file_names() {
    let root = tempfile::tempdir().expect("a temporary directory");

    let opened_files = open_required_files(root.path(), []).expect("nothing to open");

    assert!(opened_files.is_empty());
}

#[test]
fn should_reject_with_the_first_missing_relative_name_and_no_local_path() {
    let root = tempfile::tempdir().expect("a temporary directory");
    write_placeholder_files(root.path(), &["vae/config.json"]);

    let rejection = open_required_files(
        root.path(),
        [
            "vae/config.json",
            "vae/missing_first.json",
            "vae/missing_second.json",
        ],
    )
    .expect_err("the second name is absent");

    let message = rejection.to_string();
    assert!(message.contains("vae/missing_first.json"));
    assert!(!message.contains(&root.path().display().to_string()));
    assert_eq!(missing_file_name(rejection), "vae/missing_first.json");
}

#[test]
fn should_open_the_model_index_four_configs_and_three_shard_indexes() {
    let root = tempfile::tempdir().expect("a temporary directory");
    write_placeholder_files(root.path(), &all_document_names());

    let document_files = open_document_files(root.path()).expect("all documents exist");

    assert_eq!(document_files.len(), 8);
    assert!(document_files.contains_key("model_index.json"));
    assert!(document_files.contains_key("vae/model.safetensors.index.json"));
}

#[test]
fn should_name_the_one_missing_shard_index_when_the_other_documents_exist() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let present_names: Vec<&str> = all_document_names()
        .into_iter()
        .filter(|name| *name != "vae/model.safetensors.index.json")
        .collect();
    write_placeholder_files(root.path(), &present_names);

    let rejection = open_document_files(root.path()).expect_err("one document is absent");

    assert_eq!(
        missing_file_name(rejection),
        "vae/model.safetensors.index.json"
    );
}

#[test]
fn should_open_all_nine_processor_sidecars_and_name_a_missing_one() {
    let root = tempfile::tempdir().expect("a temporary directory");
    write_placeholder_files(root.path(), &PROCESSOR_SIDECAR_FILE_NAMES);

    let sidecars = open_processor_sidecars(root.path()).expect("all sidecars exist");
    assert_eq!(sidecars.len(), 9);

    std::fs::remove_file(root.path().join("processor/vocab.json")).expect("remove one sidecar");
    let rejection = open_processor_sidecars(root.path()).expect_err("one sidecar is absent");
    assert_eq!(missing_file_name(rejection), "processor/vocab.json");
}

#[test]
fn should_reject_an_empty_document_set_by_naming_the_model_index() {
    let rejection = parse_component_configs(&BTreeMap::new())
        .map(|_| ())
        .expect_err("no document was opened");

    assert_eq!(missing_file_name(rejection), "model_index.json");
}

#[test]
fn should_parse_the_model_index_before_any_other_document() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let document_files = opened_documents_with_model_index(root.path(), "{}");

    let rejection = parse_component_configs(&document_files)
        .map(|_| ())
        .expect_err("an empty model index is malformed");

    assert!(matches!(
        rejection,
        QwenImage21ArtifactError::Configuration(QwenImage21ConfigError::Malformed {
            document: "model_index.json",
            ..
        })
    ));
}

#[test]
fn should_parse_the_scheduler_config_right_after_a_reviewed_model_index() {
    let root = tempfile::tempdir().expect("a temporary directory");
    let reviewed_model_index = serde_json::to_string(&qwen_image_21_pipeline_index())
        .expect("the reviewed model index serializes");
    let document_files = opened_documents_with_model_index(root.path(), &reviewed_model_index);

    let rejection = parse_component_configs(&document_files)
        .map(|_| ())
        .expect_err("the placeholder scheduler config is malformed");

    assert!(matches!(
        rejection,
        QwenImage21ArtifactError::Configuration(QwenImage21ConfigError::Malformed {
            document: "scheduler/scheduler_config.json",
            ..
        })
    ));
}
