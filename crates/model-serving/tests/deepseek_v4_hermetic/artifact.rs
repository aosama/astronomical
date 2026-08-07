use astronomical_model_serving::{DeepSeekV4ArtifactValidator, DeepSeekV4DsparkArtifactCapability};

use super::support::{
    add_unknown_index_tensor, add_unsupported_dspark_stage_tensor, remove_index_tensor,
    secondary_shard_path, write_artifact,
};

#[test]
fn should_validate_target_only_and_dspark_artifacts_without_loading_weights() {
    let target_only_temporary_directory =
        tempfile::tempdir().expect("the test should create a target-only directory");
    let target_only_artifact_directory =
        write_artifact(&target_only_temporary_directory, false, None, None);
    let target_only_artifact = DeepSeekV4ArtifactValidator::new()
        .validate(&target_only_artifact_directory)
        .expect("the target-only artifact should validate");
    assert_eq!(
        target_only_artifact.dspark_artifact_capability(),
        &DeepSeekV4DsparkArtifactCapability::TargetOnly
    );

    let dspark_temporary_directory =
        tempfile::tempdir().expect("the test should create a DSpark directory");
    let dspark_artifact_directory = write_artifact(&dspark_temporary_directory, true, None, None);
    let dspark_artifact = DeepSeekV4ArtifactValidator::new()
        .validate(&dspark_artifact_directory)
        .expect("the DSpark artifact should validate");
    assert!(dspark_artifact.dspark_artifact_capability().is_declared());
}

#[test]
fn should_reject_missing_shards_unknown_namespaces_incomplete_dspark_and_unindexed_headers() {
    let missing_shard_temporary_directory =
        tempfile::tempdir().expect("the test should create a missing-shard directory");
    let missing_shard_artifact_directory =
        write_artifact(&missing_shard_temporary_directory, false, None, None);
    std::fs::remove_file(secondary_shard_path(&missing_shard_artifact_directory))
        .expect("the test should remove an indexed shard");
    assert!(
        DeepSeekV4ArtifactValidator::new()
            .validate(&missing_shard_artifact_directory)
            .is_err()
    );

    let unknown_namespace_temporary_directory =
        tempfile::tempdir().expect("the test should create an unknown-namespace directory");
    let unknown_namespace_artifact_directory =
        write_artifact(&unknown_namespace_temporary_directory, false, None, None);
    add_unknown_index_tensor(&unknown_namespace_artifact_directory);
    assert!(
        DeepSeekV4ArtifactValidator::new()
            .validate(&unknown_namespace_artifact_directory)
            .is_err()
    );

    let incomplete_dspark_temporary_directory =
        tempfile::tempdir().expect("the test should create an incomplete-DSpark directory");
    let incomplete_dspark_artifact_directory =
        write_artifact(&incomplete_dspark_temporary_directory, true, None, None);
    remove_index_tensor(
        &incomplete_dspark_artifact_directory,
        "mtp.2.markov_head.markov_w2.weight",
    );
    assert!(
        DeepSeekV4ArtifactValidator::new()
            .validate(&incomplete_dspark_artifact_directory)
            .is_err()
    );

    let unindexed_header_temporary_directory =
        tempfile::tempdir().expect("the test should create an unindexed-header directory");
    let unindexed_header_artifact_directory = write_artifact(
        &unindexed_header_temporary_directory,
        false,
        Some("model.layers.1.unindexed.weight"),
        None,
    );
    assert!(
        DeepSeekV4ArtifactValidator::new()
            .validate(&unindexed_header_artifact_directory)
            .is_err()
    );

    let missing_header_tensor_temporary_directory =
        tempfile::tempdir().expect("the test should create a missing-header-tensor directory");
    let missing_header_tensor_artifact_directory = write_artifact(
        &missing_header_tensor_temporary_directory,
        false,
        None,
        Some("model.layers.1.attn.wq_a.weight"),
    );
    assert!(
        DeepSeekV4ArtifactValidator::new()
            .validate(&missing_header_tensor_artifact_directory)
            .is_err()
    );

    let unsupported_dspark_stage_temporary_directory =
        tempfile::tempdir().expect("the test should create an unsupported-stage directory");
    let unsupported_dspark_stage_artifact_directory = write_artifact(
        &unsupported_dspark_stage_temporary_directory,
        true,
        None,
        None,
    );
    add_unsupported_dspark_stage_tensor(&unsupported_dspark_stage_artifact_directory);
    assert!(
        DeepSeekV4ArtifactValidator::new()
            .validate(&unsupported_dspark_stage_artifact_directory)
            .is_err()
    );
}
