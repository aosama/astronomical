use astronomical_model_serving::{
    LagunaArtifactValidationError, LagunaArtifactValidator, LagunaAttentionProjection,
    LagunaCanonicalTensorAssemblyKind, LagunaExpertProjection, LagunaGlobalTensorRole,
    LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
    LagunaTensorNameNormalizationError, LagunaTensorStorageEncoding,
};
use safetensors::Dtype;

use super::artifact_support::{FIRST_SHARD_FILE_NAME, SyntheticLagunaArtifact, SyntheticTensor};

#[test]
fn should_validate_every_direct_mlx_affine_width_and_divisible_group_size() {
    for bit_width in [2, 3, 4, 5, 6, 8] {
        for group_size in [32, 64, 128] {
            let model_directory =
                tempfile::tempdir().expect("the test should create a model directory");
            SyntheticLagunaArtifact::direct_affine_dense("", bit_width, group_size, &[])
                .write(model_directory.path());

            let validated_artifact = LagunaArtifactValidator::new()
                .validate(model_directory.path())
                .expect("the complete direct-affine artifact should validate");
            let query_weight = descriptor(
                &validated_artifact,
                layer_component_id(
                    LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
                    LagunaTensorComponent::Weight,
                ),
            );
            let query_scales = descriptor(
                &validated_artifact,
                layer_component_id(
                    LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
                    LagunaTensorComponent::Scales,
                ),
            );

            assert_eq!(query_weight.logical_shape(), &[128, 128]);
            assert_eq!(query_weight.execution_dtype(), Dtype::F32);
            assert_eq!(query_weight.storage_dtype(), Dtype::U32);
            assert_eq!(query_weight.sources()[0].raw_dtype(), Dtype::U32);
            assert_eq!(
                query_weight.sources()[0].raw_shape(),
                &[128, 128 * bit_width as usize / 32]
            );
            assert_eq!(query_scales.logical_shape(), &[128, 128]);
            assert_eq!(query_scales.execution_dtype(), Dtype::F32);
            assert_eq!(query_scales.storage_dtype(), Dtype::F32);
            assert_eq!(
                query_scales.sources()[0].raw_shape(),
                &[128, 128 / group_size as usize]
            );
            assert_affine_profile(query_weight.storage_encoding(), bit_width, group_size);
            assert_affine_profile(query_scales.storage_encoding(), bit_width, group_size);
        }
    }
}

#[test]
fn should_apply_default_and_wrapped_overrides_to_canonical_executable_modules() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::direct_affine_dense(
        "language_model.",
        2,
        128,
        &[("language_model.model.layers.0.self_attn.q_proj", 5, 32)],
    )
    .write(model_directory.path());

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("a wrapped explicit override should resolve to its canonical module");
    let query_weight = descriptor(
        &validated_artifact,
        layer_component_id(
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
            LagunaTensorComponent::Weight,
        ),
    );
    let key_weight = descriptor(
        &validated_artifact,
        layer_component_id(
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Key),
            LagunaTensorComponent::Weight,
        ),
    );

    assert_eq!(
        query_weight.canonical_module_name(),
        Some("model.layers.0.self_attn.q_proj")
    );
    assert_affine_profile(query_weight.storage_encoding(), 5, 32);
    assert_eq!(query_weight.sources()[0].raw_shape(), &[128, 20]);
    assert_affine_profile(key_weight.storage_encoding(), 2, 128);
    assert_eq!(key_weight.sources()[0].raw_shape(), &[64, 8]);
}

#[test]
fn should_expose_complete_components_without_quantizing_norms() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::direct_affine_dense("", 4, 32, &[]).write(model_directory.path());

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the complete affine components should validate");
    for component in [
        LagunaTensorComponent::Weight,
        LagunaTensorComponent::Scales,
        LagunaTensorComponent::Biases,
    ] {
        assert!(
            validated_artifact
                .tensor_contract()
                .descriptor(&LagunaTensorId::Global {
                    role: LagunaGlobalTensorRole::TokenEmbedding,
                    component,
                })
                .is_some()
        );
    }
    let final_norm = descriptor(
        &validated_artifact,
        LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::FinalNormalization,
            component: LagunaTensorComponent::Weight,
        },
    );
    assert_eq!(final_norm.canonical_module_name(), None);
    assert_eq!(
        final_norm.storage_encoding(),
        &LagunaTensorStorageEncoding::Unquantized
    );
    assert!(
        validated_artifact
            .tensor_contract()
            .descriptor(&LagunaTensorId::Global {
                role: LagunaGlobalTensorRole::FinalNormalization,
                component: LagunaTensorComponent::Scales,
            })
            .is_none()
    );
}

#[test]
fn should_store_affine_sidecars_in_their_supported_physical_float_dtype() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::direct_affine_dense("", 4, 32, &[]);
    fixture.config["torch_dtype"] = serde_json::json!("bfloat16");
    for tensor in fixture.tensors_by_shard.values_mut().flatten() {
        if tensor.dtype == "F32" {
            tensor.dtype = "BF16";
        }
    }
    fixture
        .tensor_mut("model.layers.0.self_attn.q_proj.scales")
        .dtype = "F16";
    fixture.write(model_directory.path());

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("affine sidecars should preserve their source float dtype");
    let query_scales = descriptor(
        &validated_artifact,
        layer_component_id(
            LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
            LagunaTensorComponent::Scales,
        ),
    );
    assert_eq!(query_scales.execution_dtype(), Dtype::BF16);
    assert_eq!(query_scales.storage_dtype(), Dtype::F16);
    assert_eq!(query_scales.sources()[0].raw_dtype(), Dtype::F16);
}

#[test]
fn should_reject_wrong_affine_component_shapes_and_dtypes() {
    for (tensor_name, malformed_shape, malformed_dtype) in [
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![128, 15],
            "F32",
        ),
        (
            "model.layers.0.self_attn.q_proj.scales",
            vec![128, 3],
            "U32",
        ),
        (
            "model.layers.0.self_attn.q_proj.biases",
            vec![128, 3],
            "U32",
        ),
    ] {
        let shape_directory =
            tempfile::tempdir().expect("the test should create a model directory");
        let mut shape_fixture = SyntheticLagunaArtifact::direct_affine_dense("", 4, 32, &[]);
        shape_fixture.tensor_mut(tensor_name).shape = malformed_shape;
        shape_fixture.write(shape_directory.path());
        assert!(matches!(
            validation_error(shape_directory.path()),
            LagunaArtifactValidationError::TensorShapeMismatch { .. }
        ));

        let dtype_directory =
            tempfile::tempdir().expect("the test should create a model directory");
        let mut dtype_fixture = SyntheticLagunaArtifact::direct_affine_dense("", 4, 32, &[]);
        dtype_fixture.tensor_mut(tensor_name).dtype = malformed_dtype;
        dtype_fixture.write(dtype_directory.path());
        assert!(matches!(
            validation_error(dtype_directory.path()),
            LagunaArtifactValidationError::TensorDtypeMismatch { .. }
        ));
    }
}

#[test]
fn should_reject_non_integral_packed_affine_geometry() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::direct_affine_dense("", 3, 32, &[]);
    fixture.config["hidden_size"] = serde_json::json!(127);
    fixture.write(model_directory.path());

    assert!(matches!(
        validation_error(model_directory.path()),
        LagunaArtifactValidationError::InvalidAffineDimension { .. }
    ));
}

#[test]
fn should_reject_missing_and_extra_sidecars_before_construction() {
    let missing_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut missing_fixture = SyntheticLagunaArtifact::direct_affine_dense("", 4, 32, &[]);
    missing_fixture.remove_tensor_completely("model.layers.0.self_attn.q_proj.biases");
    missing_fixture.write(missing_directory.path());
    assert!(matches!(
        validation_error(missing_directory.path()),
        LagunaArtifactValidationError::ExpectedTensorMissing { .. }
    ));

    for unquantized_sidecar_name in ["model.norm.scales", "model.layers.0.input_layernorm.biases"] {
        let norm_directory = tempfile::tempdir().expect("the test should create a model directory");
        let mut norm_fixture = SyntheticLagunaArtifact::direct_affine_dense("", 4, 32, &[]);
        norm_fixture.add_tensor(
            FIRST_SHARD_FILE_NAME,
            SyntheticTensor {
                name: unquantized_sidecar_name.to_owned(),
                dtype: "F32",
                shape: vec![128, 4],
            },
        );
        norm_fixture.write(norm_directory.path());
        assert!(matches!(
            validation_error(norm_directory.path()),
            LagunaArtifactValidationError::TensorNames(
                LagunaTensorNameNormalizationError::UnknownTensorName { .. }
            )
        ));
    }

    let router_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut router_fixture = SyntheticLagunaArtifact::direct_affine_sparse_stacked(4, 32);
    router_fixture.add_tensor(
        FIRST_SHARD_FILE_NAME,
        SyntheticTensor {
            name: "model.layers.0.mlp.gate.scales".to_owned(),
            dtype: "F32",
            shape: vec![2, 4],
        },
    );
    router_fixture.write(router_directory.path());
    assert!(matches!(
        validation_error(router_directory.path()),
        LagunaArtifactValidationError::UnexpectedCanonicalTensor { .. }
    ));
}

#[test]
fn should_reject_unapplied_or_inapplicable_explicit_overrides() {
    for override_name in [
        "model.layers.0.self_attn.q_porj",
        "model.layers.0.input_layernorm",
    ] {
        let model_directory =
            tempfile::tempdir().expect("the test should create a model directory");
        SyntheticLagunaArtifact::direct_affine_dense("", 4, 32, &[(override_name, 8, 32)])
            .write(model_directory.path());

        assert!(matches!(
            validation_error(model_directory.path()),
            LagunaArtifactValidationError::AffineOverrideResolution {
                resolved_module_count: 0,
                ..
            }
        ));
    }
}

#[test]
fn should_validate_stacked_per_expert_and_fused_affine_components() {
    for (fixture, expected_assembly, expected_source_count, expected_source_shape) in [
        (
            SyntheticLagunaArtifact::direct_affine_sparse_stacked(4, 32),
            LagunaCanonicalTensorAssemblyKind::StackedSource,
            1,
            vec![2, 128, 16],
        ),
        (
            SyntheticLagunaArtifact::direct_affine_sparse_per_expert(4, 32),
            LagunaCanonicalTensorAssemblyKind::PerExpertStack,
            2,
            vec![128, 16],
        ),
        (
            SyntheticLagunaArtifact::direct_affine_sparse_fused_stacked(4, 32),
            LagunaCanonicalTensorAssemblyKind::FusedGateUpSource {
                projection: LagunaExpertProjection::Gate,
            },
            1,
            vec![2, 256, 16],
        ),
        (
            SyntheticLagunaArtifact::direct_affine_sparse_fused_per_expert(4, 32),
            LagunaCanonicalTensorAssemblyKind::FusedPerExpertGateUp {
                projection: LagunaExpertProjection::Gate,
            },
            2,
            vec![256, 16],
        ),
    ] {
        let model_directory =
            tempfile::tempdir().expect("the test should create a model directory");
        fixture.write(model_directory.path());
        let validated_artifact = LagunaArtifactValidator::new()
            .validate(model_directory.path())
            .expect("all evidenced expert packaging should validate component-wise");
        let routed_gate_weight = descriptor(
            &validated_artifact,
            layer_component_id(
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
                LagunaTensorComponent::Weight,
            ),
        );
        let routed_gate_scales = descriptor(
            &validated_artifact,
            layer_component_id(
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
                LagunaTensorComponent::Scales,
            ),
        );

        assert_eq!(routed_gate_weight.logical_shape(), &[2, 128, 128]);
        assert_eq!(routed_gate_weight.assembly_kind(), expected_assembly);
        assert_eq!(routed_gate_weight.sources().len(), expected_source_count);
        assert_eq!(
            routed_gate_weight.sources()[0].raw_shape(),
            expected_source_shape
        );
        let mut expected_scale_shape = routed_gate_weight.sources()[0].raw_shape().to_vec();
        *expected_scale_shape
            .last_mut()
            .expect("the affine source should have an input axis") = 4;
        assert_eq!(
            routed_gate_scales.sources()[0].raw_shape(),
            expected_scale_shape
        );
        assert_eq!(
            routed_gate_weight.canonical_module_name(),
            Some("model.layers.0.mlp.switch_mlp.gate_proj")
        );
        assert!(
            validated_artifact
                .tensor_contract()
                .descriptor(&layer_component_id(
                    LagunaLayerTensorRole::SharedExpertGate,
                    LagunaTensorComponent::Weight,
                ))
                .is_none()
        );
    }
}

#[test]
fn should_reject_mixed_expert_component_packaging() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::direct_affine_sparse_stacked(4, 32);
    fixture.remove_tensor_completely("model.layers.0.mlp.switch_mlp.gate_proj.scales");
    for expert_index in 0..2 {
        fixture.add_tensor(
            FIRST_SHARD_FILE_NAME,
            SyntheticTensor {
                name: format!("model.layers.0.mlp.experts.{expert_index}.gate_proj.scales"),
                dtype: "F32",
                shape: vec![128, 4],
            },
        );
    }
    fixture.write(model_directory.path());

    assert!(matches!(
        validation_error(model_directory.path()),
        LagunaArtifactValidationError::TensorNames(
            LagunaTensorNameNormalizationError::MixedExpertPackaging { .. }
        )
    ));
}

fn descriptor(
    artifact: &astronomical_model_serving::ValidatedLagunaArtifact,
    tensor_id: LagunaTensorId,
) -> &astronomical_model_serving::LagunaCanonicalTensorDescriptor {
    artifact
        .tensor_contract()
        .descriptor(&tensor_id)
        .expect("the canonical affine descriptor should exist")
}

fn assert_affine_profile(
    storage_encoding: &LagunaTensorStorageEncoding,
    expected_bit_width: u32,
    expected_group_size: u32,
) {
    let LagunaTensorStorageEncoding::DirectAffine { profile } = storage_encoding else {
        panic!("the component should carry its direct-affine profile");
    };
    assert_eq!(profile.bits(), expected_bit_width);
    assert_eq!(profile.group_size(), expected_group_size);
}

fn layer_component_id(
    role: LagunaLayerTensorRole,
    component: LagunaTensorComponent,
) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index: 0,
        role,
        component,
    }
}

fn validation_error(model_directory: &std::path::Path) -> LagunaArtifactValidationError {
    LagunaArtifactValidator::new()
        .validate(model_directory)
        .expect_err("the malformed direct-affine artifact should fail before construction")
}
