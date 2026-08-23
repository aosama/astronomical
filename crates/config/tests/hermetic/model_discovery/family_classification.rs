use std::fs;

use astronomical_config::{
    ModelFamily, ModelFamilyClassificationError, classify_model_directory,
    discover_classified_model_artifacts, requestable_model_id,
};

use super::{discover_configured_models, write_minimal_model_config, write_required_model_files};

#[test]
fn should_reject_malformed_duplicate_or_oversized_pipeline_family_markers() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("Invalid-Pipeline-Fixture");
    fs::create_dir_all(&model_directory).expect("pipeline root should be created");

    for invalid_pipeline_index in [
        br#"{"_class_name":"Flux2KleinPipeline"# as &[u8],
        br#"{"_class_name":"Flux2KleinPipeline","_class_name":"Flux2KleinPipeline"}"#,
    ] {
        fs::write(
            model_directory.join("model_index.json"),
            invalid_pipeline_index,
        )
        .expect("invalid pipeline index should be written");
        assert!(matches!(
            classify_model_directory(&model_directory),
            Err(ModelFamilyClassificationError::ParsePipelineIndex { .. })
        ));
    }

    fs::write(
        model_directory.join("model_index.json"),
        vec![b' '; 1024 * 1024 + 1],
    )
    .expect("oversized pipeline index should be written");
    assert!(matches!(
        classify_model_directory(&model_directory),
        Err(ModelFamilyClassificationError::PipelineIndexTooLarge { .. })
    ));
}

#[test]
fn should_classify_laguna_without_discovering_it_as_executable() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let laguna_model_directory = temporary_directory.path().join("Laguna-XS-Fixture");
    fs::create_dir_all(&laguna_model_directory).expect("Laguna model directory should be created");
    write_minimal_model_config(&laguna_model_directory, "laguna", 262_144);
    write_required_model_files(&laguna_model_directory);

    assert_eq!(
        classify_model_directory(&laguna_model_directory)
            .expect("Laguna family classification should complete"),
        Some(ModelFamily::Laguna)
    );
    assert!(
        discover_configured_models(&temporary_directory)[0]
            .discovered_models
            .is_empty()
    );
    let classified_artifacts =
        discover_classified_model_artifacts(&[temporary_directory.path().to_path_buf()])
            .expect("classified Laguna discovery should complete");
    assert_eq!(classified_artifacts.len(), 1);
    assert_eq!(classified_artifacts[0].model_family, ModelFamily::Laguna);
    assert_eq!(
        requestable_model_id(&laguna_model_directory).as_deref(),
        Some("Laguna-XS-Fixture")
    );
}

#[test]
fn should_reject_duplicate_or_oversized_family_configuration_before_dispatch() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let duplicate_model_directory = temporary_directory.path().join("duplicate-family");
    fs::create_dir_all(&duplicate_model_directory)
        .expect("duplicate family directory should be created");
    fs::write(
        duplicate_model_directory.join("config.json"),
        br#"{"model_type":"laguna","model_type":"qwen3_5"}"#,
    )
    .expect("duplicate family config should be written");

    assert!(matches!(
        classify_model_directory(&duplicate_model_directory),
        Err(ModelFamilyClassificationError::ParseConfig { .. })
    ));

    let oversized_model_directory = temporary_directory.path().join("oversized-family");
    fs::create_dir_all(&oversized_model_directory)
        .expect("oversized family directory should be created");
    fs::write(
        oversized_model_directory.join("config.json"),
        vec![b' '; 4 * 1024 * 1024 + 1],
    )
    .expect("oversized family config should be written");

    assert!(matches!(
        classify_model_directory(&oversized_model_directory),
        Err(ModelFamilyClassificationError::ConfigTooLarge { .. })
    ));
}
