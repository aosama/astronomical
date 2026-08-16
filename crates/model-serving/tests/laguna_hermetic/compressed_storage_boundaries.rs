use std::fs;

use astronomical_model_serving::{
    LagunaArtifactValidationError, LagunaArtifactValidator, LagunaAttentionProjection,
    LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId, LagunaTensorSourceRole,
};

use super::artifact_support::FIRST_SHARD_FILE_NAME;
use super::compressed_artifact_support::{CompressedFixtureFormat, dense_fixture};

#[test]
fn should_reject_nonintegral_block_coverage_and_preserve_source_bytes() {
    let malformed_directory = tempfile::tempdir().expect("the test should create a directory");
    let mut malformed_fixture = dense_fixture(
        "",
        CompressedFixtureFormat::BlockFp8 {
            block_row_extent: 128,
            block_column_extent: 128,
        },
    );
    malformed_fixture
        .tensor_mut("model.layers.0.self_attn.q_proj.weight_scale")
        .shape = vec![3, 2];
    malformed_fixture.write(malformed_directory.path());
    assert!(matches!(
        validation_error(malformed_directory.path()),
        LagunaArtifactValidationError::InvalidBlockFp8Coverage { .. }
    ));

    let exact_directory = tempfile::tempdir().expect("the test should create a directory");
    dense_fixture("", CompressedFixtureFormat::TwoLevelNvfp4).write(exact_directory.path());
    let shard_path = exact_directory.path().join(FIRST_SHARD_FILE_NAME);
    let bytes_before_validation =
        fs::read(&shard_path).expect("the source shard should be readable");
    let artifact = validate(exact_directory.path());
    let bytes_after_validation =
        fs::read(&shard_path).expect("the source shard should remain readable");
    assert_eq!(bytes_after_validation, bytes_before_validation);
    assert!(
        artifact
            .tensor_contract()
            .descriptors()
            .values()
            .all(|descriptor| {
                descriptor.sources().iter().all(|source| {
                    source.data_end_offset_bytes() - source.data_start_offset_bytes()
                        == source.payload_bytes()
                })
            })
    );
    let metadata_only_fingerprint = *artifact.storage_fingerprint();
    drop(artifact);
    let mut payload_changed_bytes = bytes_after_validation;
    *payload_changed_bytes
        .last_mut()
        .expect("the synthetic shard should contain payload bytes") = 1;
    fs::write(&shard_path, payload_changed_bytes).expect("the payload-only mutation should write");
    assert_eq!(
        *validate(exact_directory.path()).storage_fingerprint(),
        metadata_only_fingerprint
    );
}

#[test]
fn should_require_global_scale_but_allow_evidenced_optional_nvfp4_metadata() {
    let optional_directory = tempfile::tempdir().expect("the test should create a directory");
    let mut optional_fixture = dense_fixture("", CompressedFixtureFormat::TwoLevelNvfp4);
    optional_fixture.remove_tensor_completely("model.layers.0.self_attn.q_proj.input_global_scale");
    optional_fixture.remove_tensor_completely("model.layers.0.self_attn.q_proj.weight_shape");
    optional_fixture.write(optional_directory.path());
    let optional_artifact = validate(optional_directory.path());
    let optional_weight = descriptor(&optional_artifact, LagunaTensorComponent::Weight);
    assert!(!optional_weight.sources().iter().any(|source| matches!(
        source.role(),
        LagunaTensorSourceRole::InputGlobalScale | LagunaTensorSourceRole::LogicalShape
    )));

    let missing_directory = tempfile::tempdir().expect("the test should create a directory");
    let mut missing_fixture = dense_fixture("", CompressedFixtureFormat::TwoLevelNvfp4);
    missing_fixture.remove_tensor_completely("model.layers.0.self_attn.q_proj.weight_global_scale");
    missing_fixture.write(missing_directory.path());
    assert!(matches!(
        validation_error(missing_directory.path()),
        LagunaArtifactValidationError::ExpectedTensorMissing { .. }
    ));
}

fn descriptor(
    artifact: &astronomical_model_serving::ValidatedLagunaArtifact,
    component: LagunaTensorComponent,
) -> &astronomical_model_serving::LagunaCanonicalTensorDescriptor {
    artifact
        .tensor_contract()
        .descriptor(&LagunaTensorId::Layer {
            layer_index: 0,
            role: LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
            component,
        })
        .expect("the canonical query component should exist")
}

fn validate(
    model_directory: &std::path::Path,
) -> astronomical_model_serving::ValidatedLagunaArtifact {
    LagunaArtifactValidator::new()
        .validate(model_directory)
        .expect("the exact compressed artifact should validate")
}

fn validation_error(model_directory: &std::path::Path) -> LagunaArtifactValidationError {
    LagunaArtifactValidator::new()
        .validate(model_directory)
        .expect_err("the malformed exact compressed artifact should fail before construction")
}
