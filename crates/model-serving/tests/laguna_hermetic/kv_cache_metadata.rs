use astronomical_model_serving::{LagunaArtifactValidationError, LagunaArtifactValidator};

use super::compressed_artifact_support::{CompressedFixtureFormat, dense_fixture};

#[test]
fn should_require_complete_scalar_fp8_key_value_cache_metadata() {
    let missing_directory = tempfile::tempdir().expect("the test should create a directory");
    let mut missing_fixture = dense_fixture(
        "",
        CompressedFixtureFormat::BlockFp8 {
            block_row_extent: 128,
            block_column_extent: 128,
        },
    );
    missing_fixture.remove_tensor_completely("model.layers.0.self_attn.v_scale");
    missing_fixture.write(missing_directory.path());
    assert!(matches!(
        validation_error(missing_directory.path()),
        LagunaArtifactValidationError::ExpectedTensorMissing { .. }
    ));

    for (metadata_name, malformed_dtype, malformed_shape) in [
        ("model.layers.0.self_attn.k_scale", "BF16", vec![1]),
        ("model.layers.0.self_attn.v_scale", "F32", vec![2]),
    ] {
        let malformed_directory = tempfile::tempdir().expect("the test should create a directory");
        let mut malformed_fixture = dense_fixture(
            "",
            CompressedFixtureFormat::BlockFp8 {
                block_row_extent: 128,
                block_column_extent: 128,
            },
        );
        let metadata_tensor = malformed_fixture.tensor_mut(metadata_name);
        metadata_tensor.dtype = malformed_dtype;
        metadata_tensor.shape = malformed_shape;
        malformed_fixture.write(malformed_directory.path());
        assert!(matches!(
            validation_error(malformed_directory.path()),
            LagunaArtifactValidationError::TensorDtypeMismatch { .. }
                | LagunaArtifactValidationError::TensorShapeMismatch { .. }
        ));
    }
}

fn validation_error(model_directory: &std::path::Path) -> LagunaArtifactValidationError {
    LagunaArtifactValidator::new()
        .validate(model_directory)
        .expect_err("malformed FP8 key/value-cache metadata should fail")
}
