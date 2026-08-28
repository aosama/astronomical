//! Mixed native and affine Qwen expert-page construction.
//!
//! An affine OptiQ document does not make every sparse expert affine. Some
//! modules stay native floating-point (`weight` only). Discovery already accepts
//! that layout. These tests prove paging follows each module profile instead of
//! rejecting native experts as zero-bit affine.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use astronomical_model_serving::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, Qwen3_5Config,
    build_quantized_expert_layer_plan,
};
use serde_json::json;

use crate::common::qwen3_5_moe::certified_ornith_config_bytes;

const EXPERT_CAPACITY: usize = 2;
const OUTPUT_DIMENSION: usize = 4;
const INPUT_DIMENSION: usize = 64;
const AFFINE_BITS: usize = 6;
const AFFINE_GROUP_SIZE: usize = 64;
const NATIVE_LAYER_PREFIX: &str = "language_model.model.layers.0.mlp";
const AFFINE_LAYER_PREFIX: &str = "language_model.model.layers.1.mlp";
const MIXED_PROJECTION_LAYER_PREFIX: &str = "language_model.model.layers.2.mlp";
const SHARD_FILE_NAME: &str = "model.safetensors";
const PROJECTION_NAMES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

#[test]
fn should_plan_native_expert_layers_from_unquantized_module_profiles() {
    let mixed_expert_artifact = write_mixed_native_and_affine_layers();

    let native_layer_plan = build_quantized_expert_layer_plan(
        mixed_expert_artifact.model_directory.path(),
        &mixed_expert_artifact.weight_map,
        NATIVE_LAYER_PREFIX,
        &mixed_expert_artifact.config,
    )
    .expect("a native expert layer should plan without treating bits 0 as affine");

    assert_eq!(
        native_layer_plan.quantization_mode,
        QuantizationMode::NativeBfloat16
    );
    for projection_name in PROJECTION_NAMES {
        assert_native_projection(&native_layer_plan, projection_name);
    }
}

#[test]
fn should_plan_affine_expert_layers_from_affine_module_profiles() {
    let mixed_expert_artifact = write_mixed_native_and_affine_layers();

    let affine_layer_plan = build_quantized_expert_layer_plan(
        mixed_expert_artifact.model_directory.path(),
        &mixed_expert_artifact.weight_map,
        AFFINE_LAYER_PREFIX,
        &mixed_expert_artifact.config,
    )
    .expect("an affine expert layer should keep packed weights, scales, and biases");

    assert_eq!(
        affine_layer_plan.quantization_mode,
        QuantizationMode::Affine
    );
    for projection_name in PROJECTION_NAMES {
        assert_affine_projection(&affine_layer_plan, projection_name);
    }
}

#[test]
fn should_plan_mixed_native_and_affine_expert_layers_in_one_artifact() {
    let mixed_expert_artifact = write_mixed_native_and_affine_layers();

    let native_layer_plan = build_quantized_expert_layer_plan(
        mixed_expert_artifact.model_directory.path(),
        &mixed_expert_artifact.weight_map,
        NATIVE_LAYER_PREFIX,
        &mixed_expert_artifact.config,
    )
    .expect("the native expert layer in a mixed pack should load");
    let affine_layer_plan = build_quantized_expert_layer_plan(
        mixed_expert_artifact.model_directory.path(),
        &mixed_expert_artifact.weight_map,
        AFFINE_LAYER_PREFIX,
        &mixed_expert_artifact.config,
    )
    .expect("the affine expert layer in a mixed pack should load");

    for projection_name in PROJECTION_NAMES {
        assert_native_projection(&native_layer_plan, projection_name);
        assert_affine_projection(&affine_layer_plan, projection_name);
    }
}

#[test]
fn should_plan_mixed_native_and_affine_projections_in_one_layer() {
    let mixed_projection_artifact = write_mixed_projections_in_one_layer();

    let mixed_layer_plan = build_quantized_expert_layer_plan(
        mixed_projection_artifact.model_directory.path(),
        &mixed_projection_artifact.weight_map,
        MIXED_PROJECTION_LAYER_PREFIX,
        &mixed_projection_artifact.config,
    )
    .expect("gate native and up/down affine in one layer should plan");

    assert_native_projection(&mixed_layer_plan, "gate_proj");
    assert_affine_projection(&mixed_layer_plan, "up_proj");
    assert_affine_projection(&mixed_layer_plan, "down_proj");
}

#[test]
fn should_reject_affine_profiles_when_companion_tensors_are_absent() {
    let affine_profile_without_companions = write_weight_only_tensors_without_unquantized_resolve();

    let layer_plan_result = build_quantized_expert_layer_plan(
        affine_profile_without_companions.model_directory.path(),
        &affine_profile_without_companions.weight_map,
        NATIVE_LAYER_PREFIX,
        &affine_profile_without_companions.config,
    );

    assert!(
        matches!(
            layer_plan_result,
            Err(ExpertManifestError::MissingTensorEntry { .. })
        ),
        "an affine profile without scales and biases must fail closed: {layer_plan_result:?}"
    );
}

fn assert_native_projection(layer_plan: &QuantizedExpertLayerPlan, projection_name: &str) {
    assert_eq!(
        layer_plan.quantization_mode_for_projection(projection_name),
        QuantizationMode::NativeBfloat16
    );
    assert_eq!(
        projection_parameter_names(layer_plan, projection_name),
        ["weight"]
    );
}

fn assert_affine_projection(layer_plan: &QuantizedExpertLayerPlan, projection_name: &str) {
    assert_eq!(
        layer_plan.quantization_mode_for_projection(projection_name),
        QuantizationMode::Affine
    );
    assert_eq!(
        projection_parameter_names(layer_plan, projection_name),
        ["weight", "scales", "biases"]
    );
}

fn projection_parameter_names<'plan>(
    layer_plan: &'plan QuantizedExpertLayerPlan,
    projection_name: &str,
) -> Vec<&'plan str> {
    layer_plan
        .tensor_sources
        .iter()
        .filter(|tensor_source| tensor_source.projection_name == projection_name)
        .map(|tensor_source| tensor_source.parameter_name.as_str())
        .collect()
}

struct MixedExpertArtifact {
    model_directory: tempfile::TempDir,
    config: Qwen3_5Config,
    weight_map: HashMap<String, String>,
}

fn write_mixed_native_and_affine_layers() -> MixedExpertArtifact {
    write_expert_artifact(
        &[
            LayerStorage::Native {
                layer_prefix: NATIVE_LAYER_PREFIX,
            },
            LayerStorage::Affine {
                layer_prefix: AFFINE_LAYER_PREFIX,
            },
        ],
        true,
    )
}

fn write_mixed_projections_in_one_layer() -> MixedExpertArtifact {
    write_expert_artifact(
        &[LayerStorage::MixedProjections {
            layer_prefix: MIXED_PROJECTION_LAYER_PREFIX,
            native_projection_names: &["gate_proj"],
        }],
        true,
    )
}

fn write_weight_only_tensors_without_unquantized_resolve() -> MixedExpertArtifact {
    write_expert_artifact(
        &[LayerStorage::AffineWeightWithoutCompanions {
            layer_prefix: NATIVE_LAYER_PREFIX,
        }],
        false,
    )
}

enum LayerStorage<'layer> {
    Native {
        layer_prefix: &'layer str,
    },
    Affine {
        layer_prefix: &'layer str,
    },
    AffineWeightWithoutCompanions {
        layer_prefix: &'layer str,
    },
    MixedProjections {
        layer_prefix: &'layer str,
        native_projection_names: &'layer [&'layer str],
    },
}

fn write_expert_artifact(
    layers: &[LayerStorage<'_>],
    resolve_unquantized_modules: bool,
) -> MixedExpertArtifact {
    let model_directory =
        tempfile::tempdir().expect("the mixed expert fixture should create a temp directory");
    let mut tensors = Vec::new();
    let mut weight_map = HashMap::new();
    for layer in layers {
        match *layer {
            LayerStorage::Native { layer_prefix } => {
                append_native_layer(&mut tensors, &mut weight_map, layer_prefix);
            }
            LayerStorage::Affine { layer_prefix } => {
                append_affine_layer(&mut tensors, &mut weight_map, layer_prefix);
            }
            LayerStorage::AffineWeightWithoutCompanions { layer_prefix } => {
                append_affine_weight_without_companions(
                    &mut tensors,
                    &mut weight_map,
                    layer_prefix,
                );
            }
            LayerStorage::MixedProjections {
                layer_prefix,
                native_projection_names,
            } => {
                for projection_name in PROJECTION_NAMES {
                    if native_projection_names.contains(&projection_name) {
                        append_native_projection(
                            &mut tensors,
                            &mut weight_map,
                            layer_prefix,
                            projection_name,
                        );
                    } else {
                        append_affine_projection(
                            &mut tensors,
                            &mut weight_map,
                            layer_prefix,
                            projection_name,
                        );
                    }
                }
            }
        }
    }
    write_safetensors_file(&model_directory.path().join(SHARD_FILE_NAME), &tensors);
    let mut config = Qwen3_5Config::from_json_bytes(&certified_ornith_config_bytes())
        .expect("the certified Ornith config should parse");
    if resolve_unquantized_modules {
        let shard_tensor_names = weight_map.keys().cloned().collect::<BTreeSet<_>>();
        config.resolve_unquantized_modules_from_shard_index(&shard_tensor_names);
    }
    MixedExpertArtifact {
        model_directory,
        config,
        weight_map,
    }
}

struct ShardTensor {
    tensor_name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    payload_bytes: Vec<u8>,
}

fn append_native_layer(
    tensors: &mut Vec<ShardTensor>,
    weight_map: &mut HashMap<String, String>,
    layer_prefix: &str,
) {
    for projection_name in PROJECTION_NAMES {
        append_native_projection(tensors, weight_map, layer_prefix, projection_name);
    }
}

fn append_affine_layer(
    tensors: &mut Vec<ShardTensor>,
    weight_map: &mut HashMap<String, String>,
    layer_prefix: &str,
) {
    for projection_name in PROJECTION_NAMES {
        append_affine_projection(tensors, weight_map, layer_prefix, projection_name);
    }
}

fn append_affine_weight_without_companions(
    tensors: &mut Vec<ShardTensor>,
    weight_map: &mut HashMap<String, String>,
    layer_prefix: &str,
) {
    let packed_width = INPUT_DIMENSION * AFFINE_BITS / 32;
    for projection_name in PROJECTION_NAMES {
        let weight_name = format!("{layer_prefix}.switch_mlp.{projection_name}.weight");
        weight_map.insert(weight_name.clone(), SHARD_FILE_NAME.to_owned());
        tensors.push(ShardTensor {
            tensor_name: weight_name,
            dtype: "U32",
            shape: vec![EXPERT_CAPACITY, OUTPUT_DIMENSION, packed_width],
            payload_bytes: vec![0_u8; EXPERT_CAPACITY * OUTPUT_DIMENSION * packed_width * 4],
        });
    }
}

fn append_native_projection(
    tensors: &mut Vec<ShardTensor>,
    weight_map: &mut HashMap<String, String>,
    layer_prefix: &str,
    projection_name: &str,
) {
    let tensor_name = format!("{layer_prefix}.switch_mlp.{projection_name}.weight");
    weight_map.insert(tensor_name.clone(), SHARD_FILE_NAME.to_owned());
    tensors.push(ShardTensor {
        tensor_name,
        dtype: "BF16",
        shape: vec![EXPERT_CAPACITY, OUTPUT_DIMENSION, INPUT_DIMENSION],
        payload_bytes: vec![0_u8; EXPERT_CAPACITY * OUTPUT_DIMENSION * INPUT_DIMENSION * 2],
    });
}

fn append_affine_projection(
    tensors: &mut Vec<ShardTensor>,
    weight_map: &mut HashMap<String, String>,
    layer_prefix: &str,
    projection_name: &str,
) {
    let packed_width = INPUT_DIMENSION * AFFINE_BITS / 32;
    let scale_width = INPUT_DIMENSION / AFFINE_GROUP_SIZE;
    let weight_name = format!("{layer_prefix}.switch_mlp.{projection_name}.weight");
    let scales_name = format!("{layer_prefix}.switch_mlp.{projection_name}.scales");
    let biases_name = format!("{layer_prefix}.switch_mlp.{projection_name}.biases");
    weight_map.insert(weight_name.clone(), SHARD_FILE_NAME.to_owned());
    weight_map.insert(scales_name.clone(), SHARD_FILE_NAME.to_owned());
    weight_map.insert(biases_name.clone(), SHARD_FILE_NAME.to_owned());
    tensors.push(ShardTensor {
        tensor_name: weight_name,
        dtype: "U32",
        shape: vec![EXPERT_CAPACITY, OUTPUT_DIMENSION, packed_width],
        payload_bytes: vec![0_u8; EXPERT_CAPACITY * OUTPUT_DIMENSION * packed_width * 4],
    });
    let companion_payload_bytes = vec![0_u8; EXPERT_CAPACITY * OUTPUT_DIMENSION * scale_width * 2];
    tensors.push(ShardTensor {
        tensor_name: scales_name,
        dtype: "BF16",
        shape: vec![EXPERT_CAPACITY, OUTPUT_DIMENSION, scale_width],
        payload_bytes: companion_payload_bytes.clone(),
    });
    tensors.push(ShardTensor {
        tensor_name: biases_name,
        dtype: "BF16",
        shape: vec![EXPERT_CAPACITY, OUTPUT_DIMENSION, scale_width],
        payload_bytes: companion_payload_bytes,
    });
}

fn write_safetensors_file(file_path: &Path, tensors: &[ShardTensor]) {
    let mut payload_bytes = Vec::new();
    let mut header = serde_json::Map::new();
    for tensor in tensors {
        let payload_start = payload_bytes.len();
        payload_bytes.extend_from_slice(&tensor.payload_bytes);
        let payload_end = payload_bytes.len();
        header.insert(
            tensor.tensor_name.clone(),
            json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [payload_start, payload_end],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
        .expect("the mixed expert safetensors header should serialize");
    let mut shard_file = File::create(file_path).expect("the mixed expert shard should be created");
    shard_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .expect("the mixed expert shard should write its header length");
    shard_file
        .write_all(&header_bytes)
        .expect("the mixed expert shard should write its header");
    shard_file
        .write_all(&payload_bytes)
        .expect("the mixed expert shard should write its payload");
}
