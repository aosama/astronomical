//! Parity between the resident shard read and the bounded paged read.
//!
//! Issue #503 asks whether one complete expert layer already retained as a
//! packed page can become complete-resident-owner weights with exact parity
//! against the current shard-read route. The retained cache stores a complete
//! layer's streamed page as-is (no padding, identity slot map), so the question
//! reduces to one claim: the bounded source-interval read used by the pager
//! returns the same bytes as the whole-shard read used by the resident loader.
//!
//! Both paths below are production. The resident path is exactly what
//! `Qwen3_5ResidentExpertWeights::load` does (load the shard, take the named
//! tensor, validate it against the plan). The paged path is exactly what
//! `Qwen3_5ExpertPager::load_rust_streamed_experts` does (build a page manifest
//! from the validated plan, then load it through bounded ranges).

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use astronomical_model_serving::{
    QuantizationMode, QuantizedExpertLayerPlan, ResidentProjectionArraysForTests,
    build_quantized_expert_layer_plan, build_quantized_expert_page_manifest_from_plan,
    load_quantized_expert_page, resident_layer_arrays_for_tests,
};
use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime, MlxSafetensors,
};
use serde_json::json;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    qwen3_5_moe::frozen_ornith_1_0_config_bytes,
};

const EXPERT_CAPACITY: usize = 2;
const OUTPUT_DIMENSION: usize = 4;
const INPUT_DIMENSION: usize = 64;
const AFFINE_BITS: usize = 6;
const AFFINE_GROUP_SIZE: usize = 64;
const LAYER_PREFIX: &str = "language_model.model.layers.0.mlp";
const SHARD_FILE_NAME: &str = "model.safetensors";
const PROJECTION_NAMES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Deterministic nonzero payload byte, distinct per tensor and expert.
///
/// A zero-filled fixture would make the two reads agree for the wrong reason:
/// unread or mis-offset ranges would both evaluate to zero.
fn fixture_byte(tensor_index: usize, byte_index: u64) -> u8 {
    ((tensor_index * 37 + byte_index as usize * 11 + 1) % 251) as u8
}

struct FixtureTensor {
    tensor_name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    payload_bytes: Vec<u8>,
}

/// Writes a real safetensors shard plus the weight map the plan builder needs.
fn write_fixture(model_directory: &Path) -> HashMap<String, String> {
    let mut tensors = Vec::new();
    let mut weight_map = HashMap::new();
    let packed_width = INPUT_DIMENSION * AFFINE_BITS / 32;
    let companion_width = INPUT_DIMENSION / AFFINE_GROUP_SIZE;
    for projection_name in PROJECTION_NAMES {
        let base = format!("{LAYER_PREFIX}.switch_mlp.{projection_name}");
        let weight_name = format!("{base}.weight");
        let scales_name = format!("{base}.scales");
        let biases_name = format!("{base}.biases");
        for tensor_name in [&weight_name, &scales_name, &biases_name] {
            weight_map.insert(tensor_name.clone(), SHARD_FILE_NAME.to_owned());
        }
        tensors.push(FixtureTensor {
            tensor_name: weight_name,
            dtype: "U32",
            shape: vec![EXPERT_CAPACITY, OUTPUT_DIMENSION, packed_width],
            payload_bytes: vec![0_u8; EXPERT_CAPACITY * OUTPUT_DIMENSION * packed_width * 4],
        });
        let companion_shape = vec![EXPERT_CAPACITY, OUTPUT_DIMENSION, companion_width];
        let companion_byte_count = EXPERT_CAPACITY * OUTPUT_DIMENSION * companion_width * 2;
        for (tensor_name, dtype) in [(scales_name, "BF16"), (biases_name, "BF16")] {
            tensors.push(FixtureTensor {
                tensor_name,
                dtype,
                shape: companion_shape.clone(),
                payload_bytes: vec![0_u8; companion_byte_count],
            });
        }
    }
    for (tensor_index, tensor) in tensors.iter_mut().enumerate() {
        for (byte_index, byte) in tensor.payload_bytes.iter_mut().enumerate() {
            *byte = fixture_byte(tensor_index, byte_index as u64);
        }
    }

    let mut payload_bytes = Vec::new();
    let mut header = serde_json::Map::new();
    for tensor in &tensors {
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
        .expect("the parity safetensors header should serialize");
    let mut shard_file = File::create(model_directory.join(SHARD_FILE_NAME))
        .expect("the parity shard should create");
    shard_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .expect("the parity shard should write its header length");
    shard_file
        .write_all(&header_bytes)
        .expect("the parity shard should write its header");
    shard_file
        .write_all(&payload_bytes)
        .expect("the parity shard should write its payload");
    weight_map
}

fn parity_layer_plan(model_directory: &Path) -> QuantizedExpertLayerPlan {
    let weight_map = write_fixture(model_directory);
    let config_bytes = frozen_ornith_1_0_config_bytes();
    let configuration = astronomical_model_serving::Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Ornith configuration should parse");
    build_quantized_expert_layer_plan(model_directory, &weight_map, LAYER_PREFIX, &configuration)
        .expect("the parity layer plan should build from the validated header")
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("direct MLX test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Evaluates one array to its exact host representation for comparison.
fn evaluated_values(
    runtime: &MlxRuntime,
    array: &astronomical_runtime_integration::MlxArray,
) -> Vec<f64> {
    match array.dtype() {
        MlxDtype::UInt32 => array
            .to_vec_u32()
            .expect("a UInt32 parity tensor should evaluate")
            .into_iter()
            .map(f64::from)
            .collect(),
        _ => {
            let float32 = runtime
                .astype(array, MlxDtype::Float32)
                .expect("a floating parity tensor should cast for inspection");
            float32
                .to_vec_f32()
                .expect("a floating parity tensor should evaluate")
                .into_iter()
                .map(f64::from)
                .collect()
        }
    }
}

#[tokio::test]
async fn should_read_identical_expert_tensor_bytes_from_shard_and_bounded_page() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("the parity model directory should create");
    let layer_plan = parity_layer_plan(model_directory.path());
    let runtime = test_runtime();

    // Resident route: the whole-shard load the resident loader performs.
    let shard = runtime
        .load_safetensors(
            File::open(model_directory.path().join(SHARD_FILE_NAME))
                .expect("the parity shard should open"),
            None,
        )
        .expect("the parity shard should load through the production reader");

    // Paged route: the bounded source-interval read the pager performs.
    let complete_expert_ids = (0..EXPERT_CAPACITY).collect::<Vec<_>>();
    let page_manifest =
        build_quantized_expert_page_manifest_from_plan(&layer_plan, &complete_expert_ids)
            .expect("the complete-layer page manifest should build from the validated plan");
    let page_tensors = load_quantized_expert_page(&runtime, &page_manifest, None)
        .expect("the complete-layer page should load through bounded ranges");

    assert_eq!(
        page_tensors.len(),
        layer_plan.tensor_sources.len(),
        "the bounded page must expose every planned expert tensor"
    );

    for tensor_source in &layer_plan.tensor_sources {
        let shard_array = shard
            .tensor(&tensor_source.tensor_name)
            .unwrap_or_else(|_| panic!("the shard must expose {}", tensor_source.tensor_name));
        // The bounded reader names page tensors by projection and parameter
        // (for example "gate_proj.weight"), not by the artifact tensor name.
        let page_tensor_name = format!(
            "{}.{}",
            tensor_source.projection_name, tensor_source.parameter_name
        );
        let page_array = page_tensors
            .get(&page_tensor_name)
            .unwrap_or_else(|| panic!("the bounded page must expose {page_tensor_name}"));
        assert_eq!(
            shard_array.shape(),
            page_array.shape(),
            "tensor {} shape must match between the shard and bounded reads",
            tensor_source.tensor_name
        );
        assert_eq!(
            shard_array.dtype(),
            page_array.dtype(),
            "tensor {} dtype must match between the shard and bounded reads",
            tensor_source.tensor_name
        );
        assert_eq!(
            evaluated_values(&runtime, &shard_array),
            evaluated_values(&runtime, &page_array),
            "tensor {} values must be bit-identical between the shard and bounded reads",
            tensor_source.tensor_name
        );
    }
}

/// Per-projection arrays read through one provenance.
struct ProjectionRead {
    packed_weight: MlxArray,
    quantization_scales: Option<MlxArray>,
    quantization_biases: Option<MlxArray>,
}

/// Reads one projection through one provenance.
///
/// `shard` yields arrays by artifact tensor name; the bounded page yields them
/// by `projection.parameter`. Both are the production reads, so this adapter
/// only maps names, never values.
fn read_projection(
    _runtime: &MlxRuntime,
    layer_plan: &QuantizedExpertLayerPlan,
    projection_name: &str,
    shard: Option<&MlxSafetensors>,
    page_tensors: &HashMap<String, MlxArray>,
) -> ProjectionRead {
    let quantization_mode = layer_plan.quantization_mode_for_projection(projection_name);
    let read_parameter = |parameter_name: &str| -> MlxArray {
        match shard {
            Some(shard) => {
                let tensor_name =
                    format!("{LAYER_PREFIX}.switch_mlp.{projection_name}.{parameter_name}");
                shard
                    .tensor(&tensor_name)
                    .unwrap_or_else(|_| panic!("the shard must expose {tensor_name}"))
                    .retain()
                    .expect("the shard tensor should retain")
            }
            None => {
                let page_tensor_name = format!("{projection_name}.{parameter_name}");
                page_tensors
                    .get(&page_tensor_name)
                    .unwrap_or_else(|| panic!("the bounded page must expose {page_tensor_name}"))
                    .retain()
                    .expect("the page tensor should retain")
            }
        }
    };
    let packed_weight = read_parameter("weight");
    if quantization_mode == QuantizationMode::NativeBfloat16 {
        return ProjectionRead {
            packed_weight,
            quantization_scales: None,
            quantization_biases: None,
        };
    }
    ProjectionRead {
        packed_weight,
        quantization_scales: Some(read_parameter("scales")),
        quantization_biases: Some(read_parameter("biases")),
    }
}

fn assert_arrays_identical(
    runtime: &MlxRuntime,
    label: &str,
    shard_array: &MlxArray,
    page_array: &MlxArray,
) {
    assert_eq!(
        shard_array.shape(),
        page_array.shape(),
        "{label} shape must match between the shard and bounded constructions"
    );
    assert_eq!(
        shard_array.dtype(),
        page_array.dtype(),
        "{label} dtype must match between the shard and bounded constructions"
    );
    assert_eq!(
        evaluated_values(runtime, shard_array),
        evaluated_values(runtime, page_array),
        "{label} values must be bit-identical between the shard and bounded constructions"
    );
}

#[tokio::test]
async fn should_build_identical_resident_layers_from_shard_and_bounded_reads() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("the parity model directory should create");
    let layer_plan = parity_layer_plan(model_directory.path());
    let runtime = test_runtime();

    let shard = runtime
        .load_safetensors(
            File::open(model_directory.path().join(SHARD_FILE_NAME))
                .expect("the parity shard should open"),
            None,
        )
        .expect("the parity shard should load through the production reader");

    let complete_expert_ids = (0..EXPERT_CAPACITY).collect::<Vec<_>>();
    let page_manifest =
        build_quantized_expert_page_manifest_from_plan(&layer_plan, &complete_expert_ids)
            .expect("the complete-layer page manifest should build from the validated plan");
    let page_tensors = load_quantized_expert_page(&runtime, &page_manifest, None)
        .expect("the complete-layer page should load through bounded ranges");

    // Build the resident layer through the production construction twice: once
    // from the whole-shard arrays the resident loader reads, once from the
    // bounded arrays the pager reads.
    let build_from = |shard: Option<&MlxSafetensors>| {
        let to_owner_input = |read: ProjectionRead| ResidentProjectionArraysForTests {
            packed_weight: read.packed_weight,
            quantization_scales: read.quantization_scales,
            quantization_biases: read.quantization_biases,
        };
        let gate = to_owner_input(read_projection(
            &runtime,
            &layer_plan,
            "gate_proj",
            shard,
            &page_tensors,
        ));
        let up = to_owner_input(read_projection(
            &runtime,
            &layer_plan,
            "up_proj",
            shard,
            &page_tensors,
        ));
        let down = to_owner_input(read_projection(
            &runtime,
            &layer_plan,
            "down_proj",
            shard,
            &page_tensors,
        ));
        resident_layer_arrays_for_tests(&runtime, &layer_plan, gate, up, down)
            .expect("the production resident construction should build")
    };
    let shard_built = build_from(Some(&shard));
    let page_built = build_from(None);

    assert_eq!(
        shard_built.is_fused, page_built.is_fused,
        "both provenances must reach the same gate/up fusion decision"
    );
    assert_eq!(
        shard_built.gate_up.len(),
        page_built.gate_up.len(),
        "both provenances must retain the same gate/up entry count"
    );
    for (entry_index, (shard_entry, page_entry)) in shard_built
        .gate_up
        .iter()
        .zip(&page_built.gate_up)
        .enumerate()
    {
        let label = format!("gate_up entry {entry_index} packed weight");
        assert_arrays_identical(
            &runtime,
            &label,
            &shard_entry.packed_weight,
            &page_entry.packed_weight,
        );
        for (companion_name, shard_companion, page_companion) in [
            (
                "scales",
                &shard_entry.quantization_scales,
                &page_entry.quantization_scales,
            ),
            (
                "biases",
                &shard_entry.quantization_biases,
                &page_entry.quantization_biases,
            ),
        ] {
            assert_eq!(
                shard_companion.is_some(),
                page_companion.is_some(),
                "{label} {companion_name} presence must match between provenances"
            );
            if let (Some(shard_companion), Some(page_companion)) = (shard_companion, page_companion)
            {
                assert_arrays_identical(
                    &runtime,
                    &format!("{label} {companion_name}"),
                    shard_companion,
                    page_companion,
                );
            }
        }
    }
    assert_arrays_identical(
        &runtime,
        "down packed weight",
        &shard_built.down.packed_weight,
        &page_built.down.packed_weight,
    );
    for (companion_name, shard_companion, page_companion) in [
        (
            "scales",
            &shard_built.down.quantization_scales,
            &page_built.down.quantization_scales,
        ),
        (
            "biases",
            &shard_built.down.quantization_biases,
            &page_built.down.quantization_biases,
        ),
    ] {
        assert_eq!(
            shard_companion.is_some(),
            page_companion.is_some(),
            "down {companion_name} presence must match between provenances"
        );
        if let (Some(shard_companion), Some(page_companion)) = (shard_companion, page_companion) {
            assert_arrays_identical(
                &runtime,
                &format!("down {companion_name}"),
                shard_companion,
                page_companion,
            );
        }
    }
}
