use astronomical_model_serving::{
    LagunaArtifactValidator, PerformanceAttribution, PerformanceOperation,
};

use super::artifact_support::SyntheticLagunaArtifact;

#[test]
fn should_fingerprint_wrapper_equivalent_artifacts_as_one_storage_contract() {
    let bare_directory = tempfile::tempdir().expect("the test should create a bare directory");
    let wrapped_directory =
        tempfile::tempdir().expect("the test should create a wrapped directory");
    SyntheticLagunaArtifact::dense("").write(bare_directory.path());
    SyntheticLagunaArtifact::dense("language_model.").write(wrapped_directory.path());

    let bare_fingerprint = *LagunaArtifactValidator::new()
        .validate(bare_directory.path())
        .expect("the bare artifact should validate")
        .storage_fingerprint();
    let wrapped_fingerprint = *LagunaArtifactValidator::new()
        .validate(wrapped_directory.path())
        .expect("the wrapped artifact should validate")
        .storage_fingerprint();

    // Packaging aliases must not create separate persistent-cache compatibility identities.
    assert_eq!(bare_fingerprint, wrapped_fingerprint);
}

#[test]
fn should_attribute_laguna_artifact_mapping_and_binding_when_enabled() {
    let model_directory = tempfile::tempdir().expect("the test should create a directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());
    let mut performance_attribution = PerformanceAttribution::enabled();

    LagunaArtifactValidator::new()
        .validate_with_performance_attribution(model_directory.path(), &mut performance_attribution)
        .expect("the attributed artifact should validate");

    for operation in [
        PerformanceOperation::ArtifactValidation,
        PerformanceOperation::ModelSafetensorsMapping,
        PerformanceOperation::ModelTensorBinding,
    ] {
        assert_eq!(
            performance_attribution
                .operation_measurement(operation)
                .map(|measurement| measurement.occurrence_count()),
            Some(1)
        );
    }
}
