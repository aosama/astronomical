use astronomical_model_serving::{
    LagunaArtifactValidator, LagunaAttentionProjection, LagunaLayerTensorRole,
    LagunaTensorComponent, LagunaTensorId, LagunaTensorSourceRole,
};
use safetensors::Dtype;

use super::compressed_artifact_support::{CompressedFixtureFormat, dense_fixture};

#[test]
fn should_preserve_e4m3_nvfp4_scales_and_json_scalar_global_scales() {
    let model_directory = tempfile::tempdir().expect("the test should create a directory");
    let mut fixture = dense_fixture("", CompressedFixtureFormat::TwoLevelNvfp4);
    for shard_tensors in fixture.tensors_by_shard.values_mut() {
        for source_tensor in shard_tensors {
            if source_tensor.name.ends_with(".weight_scale") {
                source_tensor.dtype = "F8_E4M3";
            }
            if source_tensor.name.ends_with("_global_scale") {
                source_tensor.shape = Vec::new();
            }
        }
    }
    fixture.write(model_directory.path());

    let artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the evidenced NVFP4 source variants should validate");
    let query_weight = artifact
        .tensor_contract()
        .descriptor(&LagunaTensorId::Layer {
            layer_index: 0,
            role: LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
            component: LagunaTensorComponent::Weight,
        })
        .expect("the canonical query weight should exist");

    assert!(query_weight.sources().iter().any(|source| {
        source.role() == LagunaTensorSourceRole::GroupScales && source.raw_dtype() == Dtype::F8_E4M3
    }));
    assert!(
        query_weight
            .sources()
            .iter()
            .filter(|source| {
                matches!(
                    source.role(),
                    LagunaTensorSourceRole::WeightGlobalScale
                        | LagunaTensorSourceRole::InputGlobalScale
                )
            })
            .all(|source| source.raw_shape().is_empty())
    );
}

#[test]
fn should_reject_block_fp8_scale_geometry_that_contradicts_config() {
    let model_directory = tempfile::tempdir().expect("the test should create a directory");
    let mut fixture = dense_fixture(
        "",
        CompressedFixtureFormat::BlockFp8 {
            block_row_extent: 128,
            block_column_extent: 128,
        },
    );
    fixture
        .tensor_mut("model.layers.0.self_attn.q_proj.weight_scale")
        .shape = vec![4, 2];
    fixture.write(model_directory.path());

    assert!(matches!(
        LagunaArtifactValidator::new()
            .validate(model_directory.path())
            .expect_err("physical FP8 block geometry must match the config"),
        astronomical_model_serving::LagunaArtifactValidationError::BlockFp8GeometryMismatch {
            declared_block_row_extent: 128,
            declared_block_column_extent: 128,
            actual_block_row_extent: 32,
            actual_block_column_extent: 64,
            ..
        }
    ));
}
