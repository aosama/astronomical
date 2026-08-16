use serde_json::json;

use super::artifact_support::{SyntheticLagunaArtifact, SyntheticTensor};

// Verbatim quantization_config from poolside/Laguna-M.1-NVFP4 at the public main revision.
pub(super) const PUBLISHED_M1_NVFP4_QUANTIZATION_CONFIG: &str = r#"{
  "config_groups": {
    "group_0": {
      "format": "nvfp4-pack-quantized",
      "input_activations": {
        "actorder": null,
        "block_structure": null,
        "dynamic": "local",
        "group_size": 16,
        "num_bits": 4,
        "observer": "static_minmax",
        "observer_kwargs": {},
        "scale_dtype": "torch.float8_e4m3fn",
        "strategy": "tensor_group",
        "symmetric": true,
        "type": "float",
        "zp_dtype": null
      },
      "output_activations": null,
      "targets": [
        "re:.*mlp\\.(gate_proj|up_proj|down_proj)$",
        "re:.*experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)$",
        "re:.*shared_expert\\.(gate_proj|up_proj|down_proj)$",
        "re:.*mlp\\.(gate_up_proj|down_proj)$",
        "re:.*experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)$",
        "re:.*shared_expert\\.(gate_up_proj|down_proj)$"
      ],
      "weights": {
        "actorder": null,
        "block_structure": null,
        "dynamic": false,
        "group_size": 16,
        "num_bits": 4,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
        "scale_dtype": "torch.float8_e4m3fn",
        "strategy": "tensor_group",
        "symmetric": true,
        "type": "float",
        "zp_dtype": null
      }
    }
  },
  "format": "nvfp4-pack-quantized",
  "global_compression_ratio": null,
  "ignore": [
    "lm_head",
    "re:.*\\.self_attn\\.q_proj$",
    "re:.*\\.self_attn\\.k_proj$",
    "re:.*\\.self_attn\\.v_proj$",
    "re:.*\\.self_attn\\.o_proj$",
    "re:.*\\.self_attn\\.g_proj$",
    "re:.*\\.mlp\\.gate$"
  ],
  "kv_cache_scheme": {
    "actorder": null,
    "block_structure": null,
    "dynamic": false,
    "group_size": null,
    "num_bits": 8,
    "observer": "minmax",
    "observer_kwargs": {},
    "scale_dtype": null,
    "strategy": "tensor",
    "symmetric": true,
    "type": "float",
    "zp_dtype": null
  },
  "quant_method": "compressed-tensors",
  "quantization_status": "compressed",
  "sparsity_config": {},
  "transform_config": {},
  "version": "0.14.1.dev11+gf2ee47b"
}"#;

/// Exact upstream storage layouts represented by synthetic header-only fixtures.
#[derive(Clone, Copy)]
pub(super) enum CompressedFixtureFormat {
    SymmetricPackedI32,
    SymmetricPackedU8,
    NativeNvfp4,
    TwoLevelNvfp4,
    BlockFp8 {
        block_row_extent: usize,
        block_column_extent: usize,
    },
}

pub(super) fn dense_fixture(
    namespace_prefix: &str,
    format: CompressedFixtureFormat,
) -> SyntheticLagunaArtifact {
    let mut fixture = SyntheticLagunaArtifact::dense(namespace_prefix);
    fixture.config["vocab_size"] = json!(256);
    fixture.config["hidden_size"] = json!(128);
    fixture.config["intermediate_size"] = json!(128);
    fixture.config["num_attention_heads"] = json!(4);
    fixture.config["num_key_value_heads"] = json!(4);
    fixture.config["head_dim"] = json!(32);
    fixture.replace_dense_logical_shapes(namespace_prefix);
    use_block_compatible_attention_geometry(&mut fixture, namespace_prefix);
    apply_format(&mut fixture, format);
    fixture
}

/// Builds the M.1 module-selection journey without materializing model payload values.
pub(super) fn published_m1_nvfp4_dense_fixture() -> SyntheticLagunaArtifact {
    let mut fixture = SyntheticLagunaArtifact::dense("");
    fixture.config["vocab_size"] = json!(256);
    fixture.config["hidden_size"] = json!(128);
    fixture.config["intermediate_size"] = json!(128);
    fixture.config["num_attention_heads"] = json!(4);
    fixture.config["num_key_value_heads"] = json!(4);
    fixture.config["head_dim"] = json!(32);
    fixture.replace_dense_logical_shapes("");
    use_block_compatible_attention_geometry(&mut fixture, "");
    fixture.config["quantization_config"] =
        serde_json::from_str(PUBLISHED_M1_NVFP4_QUANTIZATION_CONFIG)
            .expect("the published quantization_config fixture should remain valid JSON");
    apply_format_to_selected_modules(
        &mut fixture,
        CompressedFixtureFormat::TwoLevelNvfp4,
        is_feed_forward_projection,
    );
    fixture
}

/// Builds selected sparse experts while leaving the router and model-wide matrices direct.
pub(super) fn published_m1_nvfp4_sparse_fixture() -> SyntheticLagunaArtifact {
    let mut fixture = SyntheticLagunaArtifact::direct_affine_sparse_fixture(false, false);
    use_block_compatible_attention_geometry(&mut fixture, "");
    fixture.config["quantization_config"] =
        serde_json::from_str(PUBLISHED_M1_NVFP4_QUANTIZATION_CONFIG)
            .expect("the published quantization_config fixture should remain valid JSON");
    apply_format_to_selected_modules(
        &mut fixture,
        CompressedFixtureFormat::TwoLevelNvfp4,
        is_feed_forward_projection,
    );
    fixture
}

pub(super) fn sparse_fixture(
    is_per_expert: bool,
    format: CompressedFixtureFormat,
) -> SyntheticLagunaArtifact {
    let mut fixture = SyntheticLagunaArtifact::direct_affine_sparse_fixture(is_per_expert, false);
    use_block_compatible_attention_geometry(&mut fixture, "");
    apply_format(&mut fixture, format);
    fixture
}

fn use_block_compatible_attention_geometry(
    fixture: &mut SyntheticLagunaArtifact,
    namespace_prefix: &str,
) {
    fixture.config["num_key_value_heads"] = json!(4);
    for projection_name in ["k_proj", "v_proj"] {
        fixture
            .tensor_mut(&format!(
                "{namespace_prefix}model.layers.0.self_attn.{projection_name}.weight"
            ))
            .shape = vec![128, 128];
    }
}

pub(super) fn apply_format(fixture: &mut SyntheticLagunaArtifact, format: CompressedFixtureFormat) {
    fixture.config["quantization_config"] = match format {
        CompressedFixtureFormat::SymmetricPackedI32
        | CompressedFixtureFormat::SymmetricPackedU8 => json!({
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {
                "num_bits": 4, "group_size": 32, "type": "int"
            }}}
        }),
        CompressedFixtureFormat::NativeNvfp4 => {
            json!({"group_size": 16, "bits": 4, "mode": "nvfp4"})
        }
        CompressedFixtureFormat::TwoLevelNvfp4 => json!({
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {"group_0": {
                "format": "nvfp4-pack-quantized",
                "weights": {"num_bits": 4, "group_size": 16}
            }}
        }),
        CompressedFixtureFormat::BlockFp8 {
            block_row_extent,
            block_column_extent,
        } => json!({
            "quant_method": "compressed-tensors",
            "kv_cache_scheme": {
                "dynamic": false,
                "num_bits": 8,
                "observer": "minmax",
                "strategy": "tensor",
                "symmetric": true,
                "type": "float"
            },
            "config_groups": {"group_0": {
                "format": "float-quantized",
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "block_structure": [block_row_extent, block_column_extent]
                }
            }}
        }),
    };

    apply_format_to_selected_modules(fixture, format, is_compressed_matrix);
}

fn apply_format_to_selected_modules(
    fixture: &mut SyntheticLagunaArtifact,
    format: CompressedFixtureFormat,
    is_selected_module: fn(&str) -> bool,
) {
    for (shard_file_name, shard_tensors) in &mut fixture.tensors_by_shard {
        let mut sidecars = Vec::new();
        for weight_tensor in shard_tensors.iter_mut() {
            let Some(raw_module_name) = weight_tensor
                .name
                .strip_suffix(".weight")
                .map(str::to_owned)
            else {
                continue;
            };
            if !is_selected_module(&raw_module_name) {
                continue;
            }
            let logical_shape = weight_tensor.shape.clone();
            let logical_input_width = *logical_shape.last().expect("a matrix has an input axis");
            let logical_output_width = logical_shape[logical_shape.len() - 2];
            fixture.indexed_shard_by_tensor.remove(&weight_tensor.name);
            match format {
                CompressedFixtureFormat::SymmetricPackedI32
                | CompressedFixtureFormat::SymmetricPackedU8 => {
                    weight_tensor.name = format!("{raw_module_name}.weight_packed");
                    weight_tensor.dtype =
                        if matches!(format, CompressedFixtureFormat::SymmetricPackedI32) {
                            "I32"
                        } else {
                            "U8"
                        };
                    *weight_tensor
                        .shape
                        .last_mut()
                        .expect("a matrix has an input axis") =
                        if matches!(format, CompressedFixtureFormat::SymmetricPackedI32) {
                            logical_input_width / 8
                        } else {
                            logical_input_width / 2
                        };
                    sidecars.push(SyntheticTensor {
                        name: format!("{raw_module_name}.weight_scale"),
                        dtype: "F32",
                        shape: replace_last(&logical_shape, logical_input_width / 32),
                    });
                    sidecars.push(shape_metadata(
                        &raw_module_name,
                        logical_output_width,
                        logical_input_width,
                    ));
                }
                CompressedFixtureFormat::NativeNvfp4 => {
                    weight_tensor.dtype = "U32";
                    *weight_tensor
                        .shape
                        .last_mut()
                        .expect("a matrix has an input axis") = logical_input_width / 8;
                    sidecars.push(SyntheticTensor {
                        name: format!("{raw_module_name}.scales"),
                        dtype: "U8",
                        shape: replace_last(&logical_shape, logical_input_width / 16),
                    });
                }
                CompressedFixtureFormat::TwoLevelNvfp4 => {
                    weight_tensor.name = format!("{raw_module_name}.weight_packed");
                    weight_tensor.dtype = "U8";
                    *weight_tensor
                        .shape
                        .last_mut()
                        .expect("a matrix has an input axis") = logical_input_width / 2;
                    sidecars.extend([
                        SyntheticTensor {
                            name: format!("{raw_module_name}.weight_scale"),
                            dtype: "U8",
                            shape: replace_last(&logical_shape, logical_input_width / 16),
                        },
                        SyntheticTensor {
                            name: format!("{raw_module_name}.weight_global_scale"),
                            dtype: "F32",
                            shape: vec![1],
                        },
                        SyntheticTensor {
                            name: format!("{raw_module_name}.input_global_scale"),
                            dtype: "F32",
                            shape: vec![1],
                        },
                        shape_metadata(&raw_module_name, logical_output_width, logical_input_width),
                    ]);
                }
                CompressedFixtureFormat::BlockFp8 {
                    block_row_extent,
                    block_column_extent,
                } => {
                    weight_tensor.dtype = "F8_E4M3";
                    let mut scale_shape = logical_shape.clone();
                    let row_axis = scale_shape.len() - 2;
                    scale_shape[row_axis] = logical_output_width / block_row_extent;
                    scale_shape[row_axis + 1] = logical_input_width / block_column_extent;
                    sidecars.push(SyntheticTensor {
                        name: format!("{raw_module_name}.weight_scale"),
                        dtype: "F32",
                        shape: scale_shape,
                    });
                }
            }
            fixture
                .indexed_shard_by_tensor
                .insert(weight_tensor.name.clone(), shard_file_name.clone());
        }
        for sidecar in sidecars {
            fixture
                .indexed_shard_by_tensor
                .insert(sidecar.name.clone(), shard_file_name.clone());
            shard_tensors.push(sidecar);
        }
    }
    if fixture.config["quantization_config"]["kv_cache_scheme"].is_object() {
        let shard_file_name = fixture
            .tensors_by_shard
            .keys()
            .next()
            .expect("the synthetic artifact should have one shard")
            .clone();
        for metadata_name in [
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.v_scale",
        ] {
            if !fixture.indexed_shard_by_tensor.contains_key(metadata_name) {
                fixture.add_tensor(
                    &shard_file_name,
                    SyntheticTensor {
                        name: metadata_name.to_owned(),
                        dtype: "F32",
                        shape: vec![1],
                    },
                );
            }
        }
    }
}

fn is_feed_forward_projection(raw_module_name: &str) -> bool {
    let canonical_module_name = raw_module_name
        .strip_prefix("language_model.")
        .unwrap_or(raw_module_name);
    canonical_module_name.contains(".mlp.")
        && matches!(
            canonical_module_name.rsplit('.').next(),
            Some("gate_proj" | "up_proj" | "gate_up_proj" | "down_proj")
        )
}

fn is_compressed_matrix(raw_module_name: &str) -> bool {
    let canonical_module_name = raw_module_name
        .strip_prefix("language_model.")
        .unwrap_or(raw_module_name);
    canonical_module_name == "model.embed_tokens"
        || canonical_module_name == "lm_head"
        || canonical_module_name.contains(".self_attn.") && canonical_module_name.ends_with("_proj")
        || canonical_module_name.contains(".mlp.") && canonical_module_name.ends_with("_proj")
}

fn replace_last(shape: &[usize], replacement: usize) -> Vec<usize> {
    let mut replaced_shape = shape.to_vec();
    *replaced_shape
        .last_mut()
        .expect("a matrix has an input axis") = replacement;
    replaced_shape
}

fn shape_metadata(module_name: &str, _output_width: usize, _input_width: usize) -> SyntheticTensor {
    SyntheticTensor {
        name: format!("{module_name}.weight_shape"),
        dtype: "I64",
        shape: vec![2],
    }
}
