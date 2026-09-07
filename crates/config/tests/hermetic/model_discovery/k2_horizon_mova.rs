use std::fs;

use astronomical_config::{
    ModelFamily, classify_model_directory, discover_classified_model_artifacts,
    requestable_model_id,
};

use super::{discover_configured_models, write_minimal_model_config, write_required_model_files};

#[test]
fn should_classify_k2_horizon_mova_without_advertising_it() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let family_model_directory = temporary_directory
        .path()
        .join("K2-Horizon-MoVA-Family-Fixture");
    fs::create_dir_all(&family_model_directory)
        .expect("K2 Horizon MoVA model directory should be created");
    write_minimal_model_config(&family_model_directory, "k2_horizon_mova", 524_288);
    write_required_model_files(&family_model_directory);

    assert_eq!(
        classify_model_directory(&family_model_directory)
            .expect("K2 Horizon MoVA family classification should complete"),
        Some(ModelFamily::K2HorizonMoVA)
    );
    assert!(
        discover_configured_models(&temporary_directory)[0]
            .discovered_models
            .is_empty(),
        "classified K2 Horizon MoVA artifacts must stay unpublished until serving exists"
    );
    let classified_artifacts =
        discover_classified_model_artifacts(&[temporary_directory.path().to_path_buf()])
            .expect("classified K2 Horizon MoVA discovery should complete");
    assert_eq!(classified_artifacts.len(), 1);
    assert_eq!(
        classified_artifacts[0].model_family,
        ModelFamily::K2HorizonMoVA
    );
    assert_eq!(
        requestable_model_id(&family_model_directory).as_deref(),
        Some("K2-Horizon-MoVA-Family-Fixture")
    );
}

#[test]
fn should_skip_unrelated_model_types_instead_of_classifying_them_as_k2_horizon_mova() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let unrelated_model_directory = temporary_directory.path().join("Unrelated-Family-Fixture");
    fs::create_dir_all(&unrelated_model_directory)
        .expect("unrelated model directory should be created");
    write_minimal_model_config(&unrelated_model_directory, "llama", 4_096);
    write_required_model_files(&unrelated_model_directory);

    assert_eq!(
        classify_model_directory(&unrelated_model_directory)
            .expect("unrelated family classification should complete"),
        None
    );
    assert!(
        discover_configured_models(&temporary_directory)[0]
            .discovered_models
            .is_empty()
    );
}
