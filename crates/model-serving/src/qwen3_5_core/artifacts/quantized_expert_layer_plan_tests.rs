//! Unit contracts for the pure quantized expert layer plan assembly steps.
//!
//! `drain_projection_tensor_sources()` and `assemble_expert_layer_plan()`
//! answer without opening files or touching shared caches, so their
//! ordering, error, and boundary behavior is covered here directly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::quantized_expert_layer_plan_steps::{
    ProjectionPlanInputs, assemble_expert_layer_plan, drain_projection_tensor_sources,
};
use super::{PARAMETER_NAMES, ProjectionStorageContract};
use crate::expert_paging::quantized_expert_manifest::{
    ExpertManifestError, QuantizationMode, QuantizedTensorSource,
};
use crate::expert_paging::safetensors_header::SafetensorsDtype;
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config_bytes;

const LAYER_PREFIX: &str = "language_model.model.layers.0.mlp";
const PROJECTION_NAME: &str = "gate_proj";

fn frozen_ornith_config() -> Qwen3_5Config {
    Qwen3_5Config::from_json_bytes(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen Ornith 1.0 configuration should parse")
}

fn affine_projection_storage() -> ProjectionStorageContract {
    ProjectionStorageContract {
        quantization_bits: 4,
        quantization_group_size: 64,
        parameter_names: PARAMETER_NAMES,
        quantization_mode: QuantizationMode::Affine,
    }
}

fn expert_tensor_source(parameter_name: &str, expert_capacity: usize) -> QuantizedTensorSource {
    QuantizedTensorSource {
        tensor_name: format!("{LAYER_PREFIX}.switch_mlp.{PROJECTION_NAME}.{parameter_name}"),
        projection_name: PROJECTION_NAME.to_owned(),
        parameter_name: parameter_name.to_owned(),
        quantization_bits: 4,
        quantization_group_size: 64,
        source_file: PathBuf::from("model.safetensors"),
        source_file_size_bytes: 1_024,
        dtype: SafetensorsDtype::Uint32,
        full_shape: vec![expert_capacity, 4, 16],
        tensor_payload_offset: 128,
        bytes_per_expert: 256,
        expert_capacity,
    }
}

fn layer_plan_inputs<'a>(
    layer_prefix: &'a str,
    qwen3_5_config: &'a Qwen3_5Config,
    empty_weight_map: &'a HashMap<String, String>,
    empty_stored_names: &'a HashMap<String, String>,
) -> ProjectionPlanInputs<'a> {
    ProjectionPlanInputs {
        model_dir: Path::new("unused-model-dir"),
        weight_map: empty_weight_map,
        stored_tensor_name_by_canonical_name: empty_stored_names,
        layer_prefix,
        qwen3_5_config,
    }
}

#[test]
fn should_drain_projection_sources_in_storage_parameter_order() {
    let qwen3_5_config = frozen_ornith_config();
    let empty_weight_map: HashMap<String, String> = HashMap::new();
    let empty_stored_names: HashMap<String, String> = HashMap::new();
    let plan_inputs = layer_plan_inputs(
        LAYER_PREFIX,
        &qwen3_5_config,
        &empty_weight_map,
        &empty_stored_names,
    );
    let mut projection_sources = HashMap::new();
    projection_sources.insert("scales", Some(expert_tensor_source("scales", 8)));
    projection_sources.insert("weight", Some(expert_tensor_source("weight", 8)));
    projection_sources.insert("biases", Some(expert_tensor_source("biases", 8)));

    let drained_tensor_sources = drain_projection_tensor_sources(
        &plan_inputs,
        PROJECTION_NAME,
        projection_sources,
        &affine_projection_storage(),
    )
    .expect("a fully populated projection map should drain without errors");

    let drained_parameter_names: Vec<&str> = drained_tensor_sources
        .iter()
        .map(|tensor_source| tensor_source.parameter_name.as_str())
        .collect();
    assert_eq!(drained_parameter_names, vec!["weight", "scales", "biases"]);
}

#[test]
fn should_fail_the_drain_when_a_storage_parameter_has_no_source() {
    let qwen3_5_config = frozen_ornith_config();
    let empty_weight_map: HashMap<String, String> = HashMap::new();
    let empty_stored_names: HashMap<String, String> = HashMap::new();
    let plan_inputs = layer_plan_inputs(
        LAYER_PREFIX,
        &qwen3_5_config,
        &empty_weight_map,
        &empty_stored_names,
    );
    let mut projection_sources = HashMap::new();
    projection_sources.insert("weight", Some(expert_tensor_source("weight", 8)));

    let drain_failure = drain_projection_tensor_sources(
        &plan_inputs,
        PROJECTION_NAME,
        projection_sources,
        &affine_projection_storage(),
    )
    .expect_err("a missing scales source must fail the drain");

    assert!(matches!(
        drain_failure,
        ExpertManifestError::MissingShardTensor { .. }
    ));
}

#[test]
fn should_drain_an_empty_projection_storage_to_no_tensor_sources() {
    let qwen3_5_config = frozen_ornith_config();
    let empty_weight_map: HashMap<String, String> = HashMap::new();
    let empty_stored_names: HashMap<String, String> = HashMap::new();
    let plan_inputs = layer_plan_inputs(
        LAYER_PREFIX,
        &qwen3_5_config,
        &empty_weight_map,
        &empty_stored_names,
    );
    let empty_projection_storage = ProjectionStorageContract {
        quantization_bits: 0,
        quantization_group_size: 0,
        parameter_names: &[],
        quantization_mode: QuantizationMode::NativeBfloat16,
    };

    let drained_tensor_sources = drain_projection_tensor_sources(
        &plan_inputs,
        PROJECTION_NAME,
        HashMap::new(),
        &empty_projection_storage,
    )
    .expect("an empty storage contract should drain to an empty list");

    assert!(drained_tensor_sources.is_empty());
}

#[test]
fn should_assemble_an_affine_plan_with_the_configured_default_quantization() {
    let qwen3_5_config = frozen_ornith_config();
    let empty_weight_map: HashMap<String, String> = HashMap::new();
    let empty_stored_names: HashMap<String, String> = HashMap::new();
    let plan_inputs = layer_plan_inputs(
        LAYER_PREFIX,
        &qwen3_5_config,
        &empty_weight_map,
        &empty_stored_names,
    );
    let tensor_sources = vec![
        expert_tensor_source("weight", 8),
        expert_tensor_source("scales", 8),
        expert_tensor_source("biases", 8),
    ];

    let layer_plan = assemble_expert_layer_plan(
        &plan_inputs,
        tensor_sources,
        vec![
            QuantizationMode::Affine,
            QuantizationMode::Affine,
            QuantizationMode::Affine,
        ],
        Vec::new(),
    )
    .expect("unanimous affine projections should assemble into an affine plan");

    assert_eq!(layer_plan.layer_prefix, LAYER_PREFIX);
    assert_eq!(layer_plan.quantization_mode, QuantizationMode::Affine);
    assert_eq!(
        layer_plan.quantization_bits,
        i32::try_from(qwen3_5_config.default_quantization_bits())
            .expect("the frozen configuration's bits should fit an i32")
    );
    assert_eq!(
        layer_plan.quantization_group_size,
        i32::try_from(qwen3_5_config.default_quantization_group_size())
            .expect("the frozen configuration's group size should fit an i32")
    );
    assert_eq!(layer_plan.expert_capacity, 8);
    assert_eq!(layer_plan.tensor_sources.len(), 3);
}

#[test]
fn should_assemble_a_zero_bit_plan_when_every_projection_is_native() {
    let qwen3_5_config = frozen_ornith_config();
    let empty_weight_map: HashMap<String, String> = HashMap::new();
    let empty_stored_names: HashMap<String, String> = HashMap::new();
    let plan_inputs = layer_plan_inputs(
        LAYER_PREFIX,
        &qwen3_5_config,
        &empty_weight_map,
        &empty_stored_names,
    );
    let native_tensor_source = QuantizedTensorSource {
        dtype: SafetensorsDtype::BFloat16,
        quantization_bits: 0,
        quantization_group_size: 0,
        ..expert_tensor_source("weight", 4)
    };

    let layer_plan = assemble_expert_layer_plan(
        &plan_inputs,
        vec![native_tensor_source],
        vec![QuantizationMode::NativeBfloat16],
        Vec::new(),
    )
    .expect("unanimous native projections should assemble into a native plan");

    assert_eq!(
        layer_plan.quantization_mode,
        QuantizationMode::NativeBfloat16
    );
    assert_eq!(layer_plan.quantization_bits, 0);
    assert_eq!(layer_plan.quantization_group_size, 0);
    assert_eq!(layer_plan.expert_capacity, 4);
}

#[test]
fn should_fail_assembly_when_projections_disagree_on_expert_capacity() {
    let qwen3_5_config = frozen_ornith_config();
    let empty_weight_map: HashMap<String, String> = HashMap::new();
    let empty_stored_names: HashMap<String, String> = HashMap::new();
    let plan_inputs = layer_plan_inputs(
        LAYER_PREFIX,
        &qwen3_5_config,
        &empty_weight_map,
        &empty_stored_names,
    );
    let tensor_sources = vec![
        expert_tensor_source("weight", 8),
        expert_tensor_source("scales", 16),
    ];

    let assembly_failure = assemble_expert_layer_plan(
        &plan_inputs,
        tensor_sources,
        vec![QuantizationMode::Affine, QuantizationMode::Affine],
        Vec::new(),
    )
    .expect_err("disagreeing expert capacities must fail the assembly");

    match assembly_failure {
        ExpertManifestError::InconsistentExpertCapacity { found_capacities } => {
            let mut sorted_found_capacities = found_capacities;
            sorted_found_capacities.sort_unstable();
            assert_eq!(sorted_found_capacities, vec![8, 16]);
        }
        other => panic!("expected an inconsistent expert capacity error, got {other:?}"),
    }
}

#[test]
fn should_fail_assembly_for_a_layer_with_no_tensor_sources() {
    let qwen3_5_config = frozen_ornith_config();
    let empty_weight_map: HashMap<String, String> = HashMap::new();
    let empty_stored_names: HashMap<String, String> = HashMap::new();
    let plan_inputs = layer_plan_inputs(
        LAYER_PREFIX,
        &qwen3_5_config,
        &empty_weight_map,
        &empty_stored_names,
    );

    let assembly_failure =
        assemble_expert_layer_plan(&plan_inputs, Vec::new(), Vec::new(), Vec::new())
            .expect_err("a layer without tensor sources cannot name an expert capacity");

    match assembly_failure {
        ExpertManifestError::InconsistentExpertCapacity { found_capacities } => {
            assert!(found_capacities.is_empty());
        }
        other => panic!("expected an inconsistent expert capacity error, got {other:?}"),
    }
}
