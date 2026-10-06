use std::fs;

use astronomical_config::{ModelFamilyClassificationError, classify_model_directory};

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
fn should_reject_duplicate_or_oversized_family_configuration_before_dispatch() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let duplicate_model_directory = temporary_directory.path().join("duplicate-family");
    fs::create_dir_all(&duplicate_model_directory)
        .expect("duplicate family directory should be created");
    fs::write(
        duplicate_model_directory.join("config.json"),
        br#"{"model_type":"k2_horizon_mova","model_type":"qwen3_5"}"#,
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
