use astronomical_model_serving::{
    LagunaArtifactValidationError, LagunaArtifactValidator, LagunaIndexTotalSizeSemantics,
};

use super::artifact_support::SyntheticLagunaArtifact;

#[test]
fn should_reconcile_both_evidenced_index_total_size_semantics() {
    let serialized_directory =
        tempfile::tempdir().expect("the test should create a model directory");
    let serialized_fixture = SyntheticLagunaArtifact::dense("");
    let expected_tensor_payload_bytes = serialized_fixture.tensor_payload_bytes();
    let expected_shard_file_bytes = serialized_fixture.serialized_shard_file_bytes();
    serialized_fixture.write(serialized_directory.path());

    let serialized_artifact = LagunaArtifactValidator::new()
        .validate(serialized_directory.path())
        .expect("serialized shard-byte total_size should validate");
    assert_eq!(
        serialized_artifact.index_total_size_semantics(),
        LagunaIndexTotalSizeSemantics::SerializedShardFiles
    );
    assert_eq!(
        serialized_artifact.total_shard_file_bytes(),
        expected_shard_file_bytes
    );
    assert_eq!(
        serialized_artifact.total_tensor_payload_bytes(),
        expected_tensor_payload_bytes
    );

    let payload_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut payload_fixture = SyntheticLagunaArtifact::dense("");
    payload_fixture.declared_shard_file_size_override = Some(expected_tensor_payload_bytes);
    payload_fixture.write(payload_directory.path());
    let payload_artifact = LagunaArtifactValidator::new()
        .validate(payload_directory.path())
        .expect("tensor-payload total_size should validate");
    assert_eq!(
        payload_artifact.index_total_size_semantics(),
        LagunaIndexTotalSizeSemantics::TensorPayload
    );
}

#[test]
fn should_reject_an_index_total_size_matching_neither_evidenced_semantics() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::dense("");
    let unsupported_total_size = fixture
        .serialized_shard_file_bytes()
        .checked_add(1)
        .expect("the synthetic total should not overflow");
    fixture.declared_shard_file_size_override = Some(unsupported_total_size);
    fixture.write(model_directory.path());

    assert!(matches!(
        LagunaArtifactValidator::new()
            .validate(model_directory.path())
            .expect_err("an unknown index total_size convention should fail"),
        LagunaArtifactValidationError::IndexTotalSizeMismatch {
            declared_total_size_bytes,
            ..
        } if declared_total_size_bytes == unsupported_total_size
    ));
}
