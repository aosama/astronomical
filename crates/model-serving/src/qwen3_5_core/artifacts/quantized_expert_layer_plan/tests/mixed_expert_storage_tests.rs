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

use serde_json::json;

use crate::expert_paging::quantized_expert_manifest::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan,
};
use crate::expert_paging::safetensors_header::SafetensorsDtype;
use crate::qwen3_5_core::artifacts::quantized_expert_layer_plan::{
    build_quantized_expert_layer_plan, build_quantized_expert_layer_plans,
};
use crate::qwen3_5_core::artifacts::{
    Qwen3_5ArtifactValidator, Qwen3_5ShardIndex, qwen3_5_language_tensor_profiles,
};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config_bytes;

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
fn should_build_layer_plans_for_mixed_storage_across_multiple_layers() {
    let mixed_expert_artifact = write_mixed_native_and_affine_layers();

    let layer_plans = build_quantized_expert_layer_plans(
        mixed_expert_artifact.model_directory.path(),
        &mixed_expert_artifact.weight_map,
        &HashMap::new(),
        &mixed_expert_artifact.config,
        2,
    )
    .expect("all sparse decoder layers should be planned from shared artifact metadata");

    assert_eq!(layer_plans.len(), 2);
    assert_eq!(
        layer_plans[0].quantization_mode,
        QuantizationMode::NativeBfloat16
    );
    assert_eq!(layer_plans[1].quantization_mode, QuantizationMode::Affine);
    assert!(
        layer_plans
            .iter()
            .all(|layer_plan| layer_plan.expert_capacity == EXPERT_CAPACITY)
    );
    assert_eq!(
        layer_plans
            .iter()
            .map(|layer_plan| layer_plan.complete_expert_payload_byte_count())
            .collect::<Result<Vec<_>, _>>()
            .expect("every layer payload should have valid geometry")
            .len(),
        layer_plans.len()
    );
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
fn should_plan_stacked_jangtq2_codebook_projections_from_float16_row_scales() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    let jangtq2_artifact = write_jangtq2_expert_artifact();

    let jangtq2_layer_plan = build_quantized_expert_layer_plan(
        jangtq2_artifact.model_directory.path(),
        &jangtq2_artifact.weight_map,
        NATIVE_LAYER_PREFIX,
        &jangtq2_artifact.config,
    )
    .expect("stacked codebook projections should build a validated page plan");

    assert_eq!(
        jangtq2_layer_plan.quantization_mode,
        QuantizationMode::Jangtq2Codebook
    );
    assert_eq!(
        jangtq2_layer_plan.jangtq2_projection_rotations.len(),
        PROJECTION_NAMES.len()
    );
    for projection_name in PROJECTION_NAMES {
        assert_eq!(
            jangtq2_layer_plan.quantization_mode_for_projection(projection_name),
            QuantizationMode::Jangtq2Codebook
        );
        assert_eq!(
            jangtq2_layer_plan.jangtq2_rotation_for_projection(projection_name),
            Some(astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32)
        );
        assert_eq!(
            projection_parameter_names(&jangtq2_layer_plan, projection_name),
            ["tq2_packed", "tq2_scales"]
        );
    }
    assert_eq!(
        jangtq2_layer_plan
            .jangtq2_decoded_weight_page_shapes(1)
            .expect("one selected expert should have valid decoded matrix shapes"),
        [vec![1, 512, 2048], vec![1, 512, 2048], vec![1, 2048, 512]]
    );
    assert_eq!(
        jangtq2_layer_plan
            .tensor_sources
            .iter()
            .filter(|tensor_source| tensor_source.parameter_name == "tq2_scales")
            .map(|tensor_source| tensor_source.dtype)
            .collect::<Vec<_>>(),
        [SafetensorsDtype::Float16; 3]
    );
}

#[test]
#[ignore = "requires ASTRONOMICAL_JANGTQ2_MODEL_DIR pointing to the downloaded JANGH2 artifact"]
fn should_plan_every_jangtq2_projection_from_the_real_artifact_headers() {
    let model_directory = std::env::var_os("ASTRONOMICAL_JANGTQ2_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("ASTRONOMICAL_JANGTQ2_MODEL_DIR should name the local JANGH2 artifact");
    let config_bytes = std::fs::read(model_directory.join("config.json"))
        .expect("the JANGH2 config should be readable");
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the JANGH2 config should parse through the production parser");
    Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 1)
        .expect("the complete JANGH2 artifact should pass production validation");
    let shard_index_bytes = std::fs::read(model_directory.join("model.safetensors.index.json"))
        .expect("the JANGH2 shard index should be readable");
    let shard_index_value: serde_json::Value = serde_json::from_slice(&shard_index_bytes)
        .expect("the JANGH2 shard index should be valid JSON");
    let language_tensor_profiles = qwen3_5_language_tensor_profiles(&config)
        .expect("the JANGH2 language tensor contract should build");
    let shard_index =
        Qwen3_5ShardIndex::from_json_bytes(&shard_index_bytes, &language_tensor_profiles)
            .expect("the JANGH2 shard index should match the language tensor contract");
    let weight_map = shard_index_value["weight_map"]
        .as_object()
        .expect("the JANGH2 index should contain a weight map")
        .iter()
        .map(|(tensor_name, shard_name)| {
            (
                tensor_name.clone(),
                shard_name
                    .as_str()
                    .expect("shard names should be strings")
                    .to_owned(),
            )
        })
        .collect::<HashMap<_, _>>();

    let layer_plans = build_quantized_expert_layer_plans(
        &model_directory,
        &weight_map,
        &HashMap::new(),
        &config,
        config.layer_count() as usize,
    )
    .expect("every sparse expert layer should validate against the JANGH2 shard headers");
    let packed_codebook_source_count = layer_plans
        .iter()
        .flat_map(|layer_plan| &layer_plan.tensor_sources)
        .filter(|tensor_source| tensor_source.parameter_name == "tq2_packed")
        .count();
    let codebook_scale_sources = layer_plans
        .iter()
        .flat_map(|layer_plan| &layer_plan.tensor_sources)
        .filter(|tensor_source| tensor_source.parameter_name == "tq2_scales")
        .collect::<Vec<_>>();

    assert_eq!(layer_plans.len(), config.layer_count() as usize);
    assert_eq!(
        shard_index.language_tensor_name_to_shard_file_name().len(),
        language_tensor_profiles.len()
    );
    assert_eq!(packed_codebook_source_count, 144);
    assert_eq!(codebook_scale_sources.len(), packed_codebook_source_count);
    assert!(
        codebook_scale_sources
            .iter()
            .all(|tensor_source| tensor_source.dtype == SafetensorsDtype::Float16)
    );
    assert_eq!(
        layer_plans
            .iter()
            .map(|layer_plan| layer_plan.jangtq2_projection_rotations.len())
            .sum::<usize>(),
        packed_codebook_source_count
    );
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

fn write_jangtq2_expert_artifact() -> MixedExpertArtifact {
    let mut config_value =
        serde_json::from_slice::<serde_json::Value>(&frozen_ornith_1_0_config_bytes())
            .expect("the frozen Qwen3.5 config should decode as JSON");
    config_value["text_config"]["num_experts"] = json!(EXPERT_CAPACITY);
    config_value["text_config"]["num_experts_per_tok"] = json!(EXPERT_CAPACITY);
    let mut quantization_overrides = config_value["quantization"]
        .as_object()
        .expect("the frozen quantization profile should be an object")
        .clone();
    for projection_name in PROJECTION_NAMES {
        quantization_overrides.insert(
            format!("{NATIVE_LAYER_PREFIX}.switch_mlp.{projection_name}"),
            json!({"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"}),
        );
    }
    let quantization_value = serde_json::Value::Object(quantization_overrides);
    config_value["quantization"] = quantization_value.clone();
    config_value["quantization_config"] = quantization_value;
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the synthetic jangtq2 config should serialize as JSON");
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the synthetic jangtq2 config should parse");

    let model_directory =
        tempfile::tempdir().expect("the jangtq2 fixture should create a temporary directory");
    let mut tensors = Vec::new();
    let mut weight_map = HashMap::new();
    for projection_name in PROJECTION_NAMES {
        append_jangtq2_projection(
            &mut tensors,
            &mut weight_map,
            NATIVE_LAYER_PREFIX,
            projection_name,
            &config,
        );
    }
    write_safetensors_file(&model_directory.path().join(SHARD_FILE_NAME), &tensors);
    MixedExpertArtifact {
        model_directory,
        config,
        weight_map,
    }
}

fn append_jangtq2_projection(
    tensors: &mut Vec<ShardTensor>,
    weight_map: &mut HashMap<String, String>,
    layer_prefix: &str,
    projection_name: &str,
    config: &Qwen3_5Config,
) {
    let (input_dimension, output_dimension) = if projection_name == "down_proj" {
        (
            config.expert_intermediate_size() as usize,
            config.hidden_size() as usize,
        )
    } else {
        (
            config.hidden_size() as usize,
            config.expert_intermediate_size() as usize,
        )
    };
    let packed_width = input_dimension * 2 / 32;
    let packed_tensor_name = format!("{layer_prefix}.switch_mlp.{projection_name}.tq2_packed");
    let scales_tensor_name = format!("{layer_prefix}.switch_mlp.{projection_name}.tq2_scales");
    weight_map.insert(packed_tensor_name.clone(), SHARD_FILE_NAME.to_owned());
    weight_map.insert(scales_tensor_name.clone(), SHARD_FILE_NAME.to_owned());
    tensors.push(ShardTensor {
        tensor_name: packed_tensor_name,
        dtype: "U32",
        shape: vec![EXPERT_CAPACITY, output_dimension, packed_width],
        payload_bytes: vec![0_u8; EXPERT_CAPACITY * output_dimension * packed_width * 4],
    });
    tensors.push(ShardTensor {
        tensor_name: scales_tensor_name,
        dtype: "F16",
        shape: vec![EXPERT_CAPACITY, output_dimension],
        payload_bytes: vec![0_u8; EXPERT_CAPACITY * output_dimension * 2],
    });
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
    let mut config = Qwen3_5Config::from_json_bytes(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen Ornith 1.0 config should parse");
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
