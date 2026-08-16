use std::fs;

use astronomical_config::{
    ModelFamily, ModelFamilyClassificationError, classify_model_directory,
    discover_classified_model_artifacts, requestable_model_id,
};

use super::{discover_configured_models, write_minimal_model_config, write_required_model_files};

#[test]
fn should_classify_supported_model_family_markers() {
    for qwen_model_type in ["qwen3_5", "qwen3_5_moe", "qwen3_5_moe_vision"] {
        assert_eq!(
            ModelFamily::from_model_type(Some(qwen_model_type)),
            Some(ModelFamily::Qwen3_5)
        );
    }
    assert_eq!(
        ModelFamily::from_model_type(Some("laguna")),
        Some(ModelFamily::Laguna)
    );
    assert_eq!(
        ModelFamily::from_model_type(Some("deepseek_v4")),
        Some(ModelFamily::DeepSeekV4)
    );
    assert_eq!(ModelFamily::from_model_type(Some("unknown")), None);
    assert_eq!(ModelFamily::from_model_type(None), None);
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
fn should_classify_deepseek_v4_without_discovering_it_as_executable() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let deepseek_model_directory = temporary_directory.path().join("DeepSeek-V4-Fixture");
    fs::create_dir_all(&deepseek_model_directory)
        .expect("DeepSeek model directory should be created");
    write_minimal_model_config(&deepseek_model_directory, "deepseek_v4", 262_144);
    write_required_model_files(&deepseek_model_directory);

    assert_eq!(
        classify_model_directory(&deepseek_model_directory)
            .expect("DeepSeek family classification should complete"),
        Some(ModelFamily::DeepSeekV4)
    );
    assert!(
        discover_configured_models(&temporary_directory)[0]
            .discovered_models
            .is_empty()
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
