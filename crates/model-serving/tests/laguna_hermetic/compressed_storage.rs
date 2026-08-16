use astronomical_model_serving::{
    LagunaArtifactValidationError, LagunaArtifactValidator, LagunaAttentionProjection,
    LagunaCanonicalSourceLayout, LagunaCanonicalTensorAssemblyKind, LagunaExactStorageSupport,
    LagunaExpertProjection, LagunaLayerTensorRole, LagunaNormalizationError,
    LagunaStorageDescriptor, LagunaTargetNormalizer, LagunaTensorComponent, LagunaTensorId,
    LagunaTensorSourceRole, LagunaTensorStorageEncoding,
};
use safetensors::Dtype;
use serde_json::json;

use super::artifact_support::{FIRST_SHARD_FILE_NAME, SyntheticTensor};
use super::compressed_artifact_support::{CompressedFixtureFormat, dense_fixture, sparse_fixture};
use super::support::{config_bytes, config_value};

#[test]
fn should_normalize_each_evidenced_exact_storage_profile_and_format_location() {
    let documents = [
        (
            json!({
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {"group_0": {"weights": {
                    "num_bits": 4, "group_size": 32, "type": "int"
                }}}
            }),
            "symmetric",
        ),
        (
            json!({"group_size": 16, "bits": 4, "mode": "nvfp4"}),
            "native_nvfp4",
        ),
        (
            json!({
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "config_groups": {"group_0": {"weights": {
                    "num_bits": 4, "group_size": 16
                }}}
            }),
            "compressed_nvfp4",
        ),
        (
            json!({
                "quant_method": "compressed-tensors",
                "config_groups": {"group_0": {
                    "format": "float-quantized",
                    "weights": {"num_bits": 8, "type": "float"}
                }}
            }),
            "block_fp8",
        ),
    ];

    for (quantization_document, expected_variant) in documents {
        let mut config = config_value(1);
        config["quantization_config"] = quantization_document;
        let storage = LagunaTargetNormalizer::normalize(&config_bytes(&config))
            .expect("the evidenced exact storage profile should normalize")
            .storage()
            .clone();
        match (expected_variant, storage) {
            ("symmetric", LagunaStorageDescriptor::Compressed(compressed)) => {
                let astronomical_model_serving::LagunaCompressedWeightEncoding::SymmetricPackedAffine(profile) =
                    compressed.weight_encoding()
                else {
                    panic!("pack-quantized should retain symmetric packed storage");
                };
                assert_eq!((profile.bits(), profile.group_size()), (4, 32));
                assert_eq!(profile.support(), LagunaExactStorageSupport::RuntimeReady);
            }
            ("native_nvfp4", LagunaStorageDescriptor::NativeNvfp4(profile)) => {
                assert_eq!((profile.bits(), profile.group_size()), (4, 16));
                assert_eq!(profile.support(), LagunaExactStorageSupport::RuntimeReady);
            }
            ("compressed_nvfp4", LagunaStorageDescriptor::Compressed(compressed)) => {
                let astronomical_model_serving::LagunaCompressedWeightEncoding::TwoLevelNvfp4(
                    profile,
                ) = compressed.weight_encoding()
                else {
                    panic!("compressed NVFP4 should retain its exact storage");
                };
                assert_eq!(
                    profile.support(),
                    LagunaExactStorageSupport::FutureExactKernel
                );
            }
            ("block_fp8", LagunaStorageDescriptor::Compressed(compressed)) => {
                let astronomical_model_serving::LagunaCompressedWeightEncoding::BlockFp8(profile) =
                    compressed.weight_encoding()
                else {
                    panic!("float-quantized should retain block FP8 storage");
                };
                assert_eq!(
                    profile.support(),
                    LagunaExactStorageSupport::FutureExactKernel
                );
            }
            _ => panic!("the normalized exact storage variant should match its format"),
        }
    }
}

#[test]
fn should_reject_format_conflicts_unknown_formats_and_asymmetric_packed_storage() {
    let malformed_documents = [
        json!({
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": {
                "format": "float-quantized",
                "weights": {"num_bits": 4, "group_size": 32, "type": "int"}
            }}
        }),
        json!({"quant_method": "compressed-tensors", "format": "future-format"}),
        json!({
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {"group_0": {"targets": ["Linear"]}}
        }),
        json!({
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {
                "num_bits": 4, "group_size": 32, "type": "int", "symmetric": false
            }}}
        }),
        json!({
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {"group_0": {"weights": {
                "num_bits": 8, "group_size": 16
            }}}
        }),
        json!({
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {"group_0": {"weights": {
                "num_bits": 4, "group_size": 32
            }}}
        }),
        json!({
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {"group_0": {"weights": {
                "num_bits": 4, "type": "int"
            }}}
        }),
    ];
    for malformed_document in malformed_documents {
        let mut config = config_value(1);
        config["quantization_config"] = malformed_document;
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&config)),
            Err(LagunaNormalizationError::UnsupportedStorageEncoding { .. })
                | Err(LagunaNormalizationError::UnsupportedQuantizationValue { .. })
                | Err(LagunaNormalizationError::ConflictingQuantizationDocuments)
        ));
    }
}

#[test]
fn should_compare_top_level_and_group_formats_across_quantization_copies() {
    let mut equivalent_config = config_value(1);
    equivalent_config["quantization"] = json!({
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {"group_0": {"weights": {
            "num_bits": 4, "group_size": 16
        }}}
    });
    equivalent_config["quantization_config"] = json!({
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {
            "format": "nvfp4-pack-quantized",
            "weights": {"num_bits": 4, "group_size": 16}
        }}
    });
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&equivalent_config))
            .expect("equivalent format locations should normalize")
            .storage(),
        LagunaStorageDescriptor::Compressed(compressed)
            if matches!(
                compressed.weight_encoding(),
                astronomical_model_serving::LagunaCompressedWeightEncoding::TwoLevelNvfp4(_)
            )
    ));

    equivalent_config["quantization_config"]["config_groups"]["group_0"] = json!({
        "format": "float-quantized",
        "weights": {"num_bits": 8, "type": "float"}
    });
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&equivalent_config)),
        Err(LagunaNormalizationError::ConflictingQuantizationDocuments)
    ));
}

#[test]
fn should_preserve_symmetric_packed_codes_scales_and_derived_bias_recipe() {
    for (namespace_prefix, format, expected_packed_dtype, expected_packed_width) in [
        (
            "",
            CompressedFixtureFormat::SymmetricPackedI32,
            Dtype::I32,
            16,
        ),
        (
            "language_model.",
            CompressedFixtureFormat::SymmetricPackedU8,
            Dtype::U8,
            64,
        ),
    ] {
        let model_directory = tempfile::tempdir().expect("the test should create a directory");
        dense_fixture(namespace_prefix, format).write(model_directory.path());
        let artifact = validate(model_directory.path());
        let weight = descriptor(&artifact, LagunaTensorComponent::Weight);
        let scales = descriptor(&artifact, LagunaTensorComponent::Scales);
        let biases = descriptor(&artifact, LagunaTensorComponent::Biases);

        assert_eq!(weight.storage_dtype(), Dtype::U32);
        assert_eq!(weight.sources()[0].raw_dtype(), expected_packed_dtype);
        assert_eq!(
            weight.sources()[0].raw_shape(),
            &[128, expected_packed_width]
        );
        assert_eq!(
            weight.sources()[0].role(),
            LagunaTensorSourceRole::PackedWeightCodes
        );
        assert_eq!(
            weight.assembly_kind(),
            LagunaCanonicalTensorAssemblyKind::ReinterpretPackedBits {
                source_layout: astronomical_model_serving::LagunaCanonicalSourceLayout::Direct,
            }
        );
        assert_eq!(
            scales.sources()[0].role(),
            LagunaTensorSourceRole::GroupScales
        );
        assert_eq!(
            biases.sources()[0].raw_tensor_name(),
            scales.sources()[0].raw_tensor_name()
        );
        assert_eq!(
            biases.assembly_kind(),
            LagunaCanonicalTensorAssemblyKind::DeriveSymmetricBias {
                source_layout: astronomical_model_serving::LagunaCanonicalSourceLayout::Direct,
                negative_code_offset: 8,
            }
        );
        assert!(matches!(
            biases.storage_encoding(),
            LagunaTensorStorageEncoding::SymmetricPackedAffine { .. }
        ));
    }
}

#[test]
fn should_preserve_native_and_future_exact_storage_without_conversion() {
    for (format, expected_encoding, expected_assembly, source_roles) in [
        (
            CompressedFixtureFormat::NativeNvfp4,
            "native",
            LagunaCanonicalTensorAssemblyKind::NativeNvfp4 {
                source_layout: astronomical_model_serving::LagunaCanonicalSourceLayout::Direct,
            },
            vec![LagunaTensorSourceRole::WeightCodes],
        ),
        (
            CompressedFixtureFormat::TwoLevelNvfp4,
            "two_level",
            LagunaCanonicalTensorAssemblyKind::TwoLevelCompressedNvfp4 {
                source_layout: astronomical_model_serving::LagunaCanonicalSourceLayout::Direct,
            },
            vec![
                LagunaTensorSourceRole::PackedWeightCodes,
                LagunaTensorSourceRole::GroupScales,
                LagunaTensorSourceRole::WeightGlobalScale,
                LagunaTensorSourceRole::InputGlobalScale,
                LagunaTensorSourceRole::LogicalShape,
            ],
        ),
        (
            CompressedFixtureFormat::BlockFp8 {
                block_row_extent: 128,
                block_column_extent: 128,
            },
            "block_fp8",
            LagunaCanonicalTensorAssemblyKind::BlockFp8 {
                source_layout: astronomical_model_serving::LagunaCanonicalSourceLayout::Direct,
            },
            vec![
                LagunaTensorSourceRole::WeightCodes,
                LagunaTensorSourceRole::BlockScales,
            ],
        ),
    ] {
        for fixture in [
            dense_fixture("", format),
            sparse_fixture(false, format),
            sparse_fixture(true, format),
        ] {
            let model_directory = tempfile::tempdir().expect("the test should create a directory");
            fixture.write(model_directory.path());
            let artifact = validate(model_directory.path());
            let weight = descriptor(&artifact, LagunaTensorComponent::Weight);
            if fixture.config["mlp_layer_types"] == json!(["sparse"]) {
                assert!(matches!(
                    weight.assembly_kind(),
                    LagunaCanonicalTensorAssemblyKind::NativeNvfp4 { .. }
                        | LagunaCanonicalTensorAssemblyKind::TwoLevelCompressedNvfp4 { .. }
                        | LagunaCanonicalTensorAssemblyKind::BlockFp8 { .. }
                ));
            } else {
                assert_eq!(weight.assembly_kind(), expected_assembly);
            }
            assert_eq!(
                weight
                    .sources()
                    .iter()
                    .map(|source| source.role())
                    .collect::<Vec<_>>(),
                source_roles
            );
            match (expected_encoding, weight.storage_encoding()) {
                ("native", LagunaTensorStorageEncoding::NativeNvfp4 { .. })
                | ("two_level", LagunaTensorStorageEncoding::TwoLevelCompressedNvfp4 { .. })
                | ("block_fp8", LagunaTensorStorageEncoding::BlockFp8 { .. }) => {}
                _ => panic!("the exact storage encoding should remain declarative"),
            }
            if expected_encoding == "native" {
                assert!(
                    artifact
                        .tensor_contract()
                        .descriptor(&LagunaTensorId::Layer {
                            layer_index: 0,
                            role: LagunaLayerTensorRole::Attention(
                                LagunaAttentionProjection::Query
                            ),
                            component: LagunaTensorComponent::Biases,
                        })
                        .is_none()
                );
            }
        }
    }
}

#[test]
fn should_retain_stacked_and_per_expert_source_layouts_for_each_exact_recipe() {
    for (format, expected_recipe) in [
        (CompressedFixtureFormat::SymmetricPackedI32, "symmetric"),
        (CompressedFixtureFormat::NativeNvfp4, "native"),
        (CompressedFixtureFormat::TwoLevelNvfp4, "two_level"),
        (
            CompressedFixtureFormat::BlockFp8 {
                block_row_extent: 128,
                block_column_extent: 128,
            },
            "block_fp8",
        ),
    ] {
        for (is_per_expert, expected_layout, expected_source_count) in [
            (false, LagunaCanonicalSourceLayout::Stacked, 1),
            (true, LagunaCanonicalSourceLayout::PerExpert, 2),
        ] {
            let model_directory = tempfile::tempdir().expect("the test should create a directory");
            sparse_fixture(is_per_expert, format).write(model_directory.path());
            let artifact = validate(model_directory.path());
            let routed_weight = artifact
                .tensor_contract()
                .descriptor(&LagunaTensorId::Layer {
                    layer_index: 0,
                    role: LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
                    component: LagunaTensorComponent::Weight,
                })
                .expect("the routed exact weight should exist");
            let actual_layout =
                match routed_weight.assembly_kind() {
                    LagunaCanonicalTensorAssemblyKind::ReinterpretPackedBits { source_layout }
                        if expected_recipe == "symmetric" =>
                    {
                        source_layout
                    }
                    LagunaCanonicalTensorAssemblyKind::NativeNvfp4 { source_layout }
                        if expected_recipe == "native" =>
                    {
                        source_layout
                    }
                    LagunaCanonicalTensorAssemblyKind::TwoLevelCompressedNvfp4 {
                        source_layout,
                    } if expected_recipe == "two_level" => source_layout,
                    LagunaCanonicalTensorAssemblyKind::BlockFp8 { source_layout }
                        if expected_recipe == "block_fp8" =>
                    {
                        source_layout
                    }
                    _ => panic!("the routed tensor should retain its exact recipe"),
                };
            assert_eq!(actual_layout, expected_layout);
            assert_eq!(
                routed_weight
                    .sources()
                    .iter()
                    .filter(|source| matches!(
                        source.role(),
                        LagunaTensorSourceRole::WeightCodes
                            | LagunaTensorSourceRole::PackedWeightCodes
                    ))
                    .count(),
                expected_source_count
            );
        }
    }
}

#[test]
fn should_filter_only_evidenced_attention_scale_metadata_from_model_weights() {
    let model_directory = tempfile::tempdir().expect("the test should create a directory");
    let fixture = dense_fixture(
        "",
        CompressedFixtureFormat::BlockFp8 {
            block_row_extent: 128,
            block_column_extent: 128,
        },
    );
    fixture.write(model_directory.path());

    let artifact = validate(model_directory.path());
    assert_eq!(
        artifact.tensor_contract().non_executable_metadata().len(),
        2
    );
    assert!(
        artifact
            .tensor_contract()
            .descriptors()
            .keys()
            .all(|tensor_id| {
                !matches!(
                    tensor_id,
                    LagunaTensorId::Layer {
                        component: LagunaTensorComponent::AttentionKeyScaleMetadata
                            | LagunaTensorComponent::AttentionValueScaleMetadata,
                        ..
                    }
                )
            })
    );
}

#[test]
fn should_reject_missing_wrong_or_asymmetric_compressed_sidecars() {
    let cases = [
        "missing_scale",
        "zero_point",
        "wrong_scale_dtype",
        "wrong_shape_dtype",
    ];
    for malformed_case in cases {
        let model_directory = tempfile::tempdir().expect("the test should create a directory");
        let mut fixture = dense_fixture("", CompressedFixtureFormat::SymmetricPackedI32);
        match malformed_case {
            "missing_scale" => {
                fixture.remove_tensor_completely("model.layers.0.self_attn.q_proj.weight_scale")
            }
            "zero_point" => fixture.add_tensor(
                FIRST_SHARD_FILE_NAME,
                SyntheticTensor {
                    name: "model.layers.0.self_attn.q_proj.weight_zero_point".to_owned(),
                    dtype: "I32",
                    shape: vec![128, 4],
                },
            ),
            "wrong_scale_dtype" => {
                fixture
                    .tensor_mut("model.layers.0.self_attn.q_proj.weight_scale")
                    .dtype = "U8";
            }
            "wrong_shape_dtype" => {
                fixture
                    .tensor_mut("model.layers.0.self_attn.q_proj.weight_shape")
                    .dtype = "F32";
            }
            _ => unreachable!("the malformed case is exhaustive"),
        }
        fixture.write(model_directory.path());
        assert!(matches!(
            validation_error(model_directory.path()),
            LagunaArtifactValidationError::ExpectedTensorMissing { .. }
                | LagunaArtifactValidationError::UnsupportedAsymmetricStorage { .. }
                | LagunaArtifactValidationError::TensorDtypeMismatch { .. }
        ));
    }
}

#[test]
fn should_reject_wrong_physical_shapes_and_dtypes_for_every_exact_encoding() {
    for (format, tensor_name, malformed_shape, malformed_dtype) in [
        (
            CompressedFixtureFormat::SymmetricPackedU8,
            "model.layers.0.self_attn.q_proj.weight_packed",
            vec![128, 63],
            "F32",
        ),
        (
            CompressedFixtureFormat::NativeNvfp4,
            "model.layers.0.self_attn.q_proj.scales",
            vec![128, 7],
            "F32",
        ),
        (
            CompressedFixtureFormat::TwoLevelNvfp4,
            "model.layers.0.self_attn.q_proj.weight_scale",
            vec![128, 7],
            "F32",
        ),
        (
            CompressedFixtureFormat::BlockFp8 {
                block_row_extent: 128,
                block_column_extent: 128,
            },
            "model.layers.0.self_attn.q_proj.weight",
            vec![127, 128],
            "BF16",
        ),
    ] {
        let shape_directory = tempfile::tempdir().expect("the test should create a directory");
        let mut shape_fixture = dense_fixture("", format);
        shape_fixture.tensor_mut(tensor_name).shape = malformed_shape;
        shape_fixture.write(shape_directory.path());
        assert!(matches!(
            validation_error(shape_directory.path()),
            LagunaArtifactValidationError::TensorShapeMismatch { .. }
        ));

        let dtype_directory = tempfile::tempdir().expect("the test should create a directory");
        let mut dtype_fixture = dense_fixture("", format);
        dtype_fixture.tensor_mut(tensor_name).dtype = malformed_dtype;
        dtype_fixture.write(dtype_directory.path());
        assert!(matches!(
            validation_error(dtype_directory.path()),
            LagunaArtifactValidationError::TensorDtypeMismatch { .. }
        ));
    }
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
