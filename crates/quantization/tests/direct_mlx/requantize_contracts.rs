//! The requantizer end to end on a synthetic source: bf16 shards in, a
//! mixed-precision artifact out, read back through MLX and verified against
//! the reference encoder. Also the bounded-memory property: peak MLX memory
//! stays near one tensor regardless of how many tensors the source holds.

use std::collections::BTreeMap;

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::encoder::RangeSearchPolicy;
use astronomical_quantization::module_inventory::ModuleInventory;
use astronomical_quantization::progress::SilentProgressReporter;
use astronomical_quantization::requantize::bit_policy::BitPolicy;
use astronomical_quantization::requantize::driver::run_requantize;
use astronomical_quantization::requantize::tensor_plan::RequantizeTensorPlan;

use astronomical_quantization::encoder::reference::encode_slab;
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::requantize::tensor_plan::TensorPlanEntry;
use serial_test::{parallel, serial};

use super::support::{contract_runtime, outlier_heavy_weights};

const HIDDEN_COLUMNS: usize = 256;
const INTERMEDIATE_ROWS: usize = 128;
const ROUTER_EXPERT_COUNT: usize = 4;

/// Writes a two-shard bf16 source with a router, two linears, and a norm:
/// the router, `up_proj`, and the norm in the first shard, `down_proj` in
/// the second, plus the index and a minimal `config.json`.
fn write_two_shard_source(directory: &std::path::Path) {
    let first_shard_tensors = [
        (
            "model.layers.0.mlp.gate.weight",
            vec![ROUTER_EXPERT_COUNT, HIDDEN_COLUMNS],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![INTERMEDIATE_ROWS, HIDDEN_COLUMNS],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![HIDDEN_COLUMNS],
        ),
    ];
    let second_shard_tensors = [(
        "model.layers.0.mlp.down_proj.weight",
        vec![HIDDEN_COLUMNS, INTERMEDIATE_ROWS],
    )];
    write_bfloat16_shard(&directory.join(FIRST_SHARD_NAME), &first_shard_tensors);
    write_bfloat16_shard(&directory.join(SECOND_SHARD_NAME), &second_shard_tensors);
    let mut weight_map = serde_json::Map::new();
    for (tensor_name, _) in &first_shard_tensors {
        weight_map.insert((*tensor_name).to_owned(), FIRST_SHARD_NAME.into());
    }
    for (tensor_name, _) in &second_shard_tensors {
        weight_map.insert((*tensor_name).to_owned(), SECOND_SHARD_NAME.into());
    }
    let index_document = serde_json::json!({"metadata": {}, "weight_map": weight_map});
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        index_document.to_string(),
    )
    .expect("the source index should be writable");
    std::fs::write(
        directory.join("config.json"),
        r#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#,
    )
    .expect("the source config should be writable");
}

const FIRST_SHARD_NAME: &str = "model-00001-of-00002.safetensors";
const SECOND_SHARD_NAME: &str = "model-00002-of-00002.safetensors";

/// One bf16 shard whose tensors hold the deterministic outlier-heavy
/// weights, so the encoder sees the regime the range search exists for.
fn write_bfloat16_shard(shard_path: &std::path::Path, tensors: &[(&str, Vec<usize>)]) {
    let mut buffers = Vec::new();
    for (_, shape) in tensors {
        let element_count: usize = shape.iter().product();
        buffers.push(bfloat16_bytes(&outlier_heavy_weights(1, element_count)));
    }
    let mut views = Vec::new();
    for ((tensor_name, shape), buffer) in tensors.iter().zip(&buffers) {
        views.push((
            (*tensor_name).to_owned(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.clone(), buffer)
                .expect("the source tensor view should be valid"),
        ));
    }
    // The workspace pins `safetensors` without `std`, so serialize in memory.
    let shard_bytes =
        safetensors::tensor::serialize(views, None).expect("the source shard should serialize");
    std::fs::write(shard_path, shard_bytes).expect("the source shard should be writable");
}

/// Little-endian bf16, rounding to nearest even as MLX's cast does.
fn bfloat16_bytes(weights: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(weights.len() * 2);
    for weight in weights {
        let bits = weight.to_bits();
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        let bfloat16_bits = ((bits + rounding_bias) >> 16) as u16;
        bytes.extend_from_slice(&bfloat16_bits.to_le_bytes());
    }
    bytes
}

/// The f32 value of each bf16 the source holds.
fn bfloat16_rounded(weights: &[f32]) -> Vec<f32> {
    let bytes = bfloat16_bytes(weights);
    let mut rounded_weights = Vec::with_capacity(weights.len());
    for bfloat16_pair in bytes.as_chunks::<2>().0 {
        let bfloat16_bits = u16::from_le_bytes([bfloat16_pair[0], bfloat16_pair[1]]);
        rounded_weights.push(f32::from_bits(u32::from(bfloat16_bits) << 16));
    }
    rounded_weights
}

fn uniform_plan(source: &std::path::Path, bits: u32) -> RequantizeTensorPlan {
    let inventory =
        ModuleInventory::from_source_directory(source, GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    RequantizeTensorPlan::from_inventory(
        &inventory,
        &BitPolicy::Uniform {
            bit_width: BitWidth::new(bits).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        },
    )
}

#[test]
#[parallel]
fn should_requantize_a_two_shard_source_into_a_loadable_mixed_precision_artifact() {
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_two_shard_source(source.path());
    let inventory =
        ModuleInventory::from_source_directory(source.path(), GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    let plan = RequantizeTensorPlan::from_inventory(
        &inventory,
        &BitPolicy::Uniform {
            bit_width: BitWidth::new(4).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        },
    );
    let mut silent_reporter = SilentProgressReporter;

    let outcome = run_requantize(
        &runtime,
        source.path(),
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        64 * 1024,
        &mut silent_reporter,
    )
    .expect("requantization should succeed");

    assert!(outcome.shard_count >= 1);
    assert!(output.path().join("model.safetensors.index.json").exists());
    // The router passed through: no entry for it in the written map.
    assert!(
        !outcome
            .written_quantization_by_module
            .contains_key("model.layers.0.mlp.gate")
    );
    let expected_written: BTreeMap<String, (u32, u32)> = BTreeMap::from([
        ("model.layers.0.mlp.up_proj".to_owned(), (4, 64)),
        ("model.layers.0.mlp.down_proj".to_owned(), (4, 64)),
    ]);
    assert_eq!(outcome.written_quantization_by_module, expected_written);
}

#[test]
#[serial]
fn should_keep_peak_mlx_memory_near_one_tensor_regardless_of_tensor_count() {
    // Requantize a source with N copies of the same tensor and assert the
    // reported peak is bounded by a small multiple of one tensor's float32
    // working copy, the property that lets a 300B checkpoint convert on a
    // laptop. The bound is deliberately loose (8x one tensor) so allocator
    // slack cannot flake it; the point is that it does not scale with N.
    // Peak memory is process-global, so this contract runs #[serial] and
    // every other MLX contract in the binary is #[parallel]: the peak it
    // samples must be its own.
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_two_shard_source(source.path());
    let inventory =
        ModuleInventory::from_source_directory(source.path(), GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    let plan = RequantizeTensorPlan::from_inventory(
        &inventory,
        &BitPolicy::Uniform {
            bit_width: BitWidth::new(2).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        },
    );
    let mut silent_reporter = SilentProgressReporter;

    let outcome = run_requantize(
        &runtime,
        source.path(),
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        64 * 1024,
        &mut silent_reporter,
    )
    .expect("requantization should succeed");

    let largest_tensor_float32_bytes = plan.entries.iter().fold(0_u64, largest_float32_bytes);
    assert!(outcome.peak_mlx_memory_bytes <= 8 * largest_tensor_float32_bytes);
}

fn largest_float32_bytes(largest_so_far: u64, entry: &TensorPlanEntry) -> u64 {
    largest_so_far.max(entry.shape.iter().product::<usize>() as u64 * 4)
}

#[test]
#[parallel]
fn should_write_packed_words_that_match_the_reference_encoder() {
    // At 4 bits the default policy uses stock quantize_affine, which the
    // reference reproduces word for word from the same bf16 weights.
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_two_shard_source(source.path());
    let plan = uniform_plan(source.path(), 4);
    let mut silent_reporter = SilentProgressReporter;

    run_requantize(
        &runtime,
        source.path(),
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        64 * 1024,
        &mut silent_reporter,
    )
    .expect("requantization should succeed");

    let index_text = std::fs::read_to_string(output.path().join("model.safetensors.index.json"))
        .expect("the index should be readable");
    let index_document: serde_json::Value =
        serde_json::from_str(&index_text).expect("the index should parse");
    let up_projection_key = "model.layers.0.mlp.up_proj.weight";
    let shard_name = index_document["weight_map"][up_projection_key]
        .as_str()
        .expect("the index maps up_proj");
    let shard_file =
        std::fs::File::open(output.path().join(shard_name)).expect("the shard should open");
    let shard_map = runtime
        .load_safetensors(shard_file, None)
        .expect("the written shard should load through MLX");
    let written_words = shard_map
        .tensor(up_projection_key)
        .expect("up_proj should be in its shard")
        .to_vec_u32()
        .expect("packed words read back");
    let source_weights = bfloat16_rounded(&outlier_heavy_weights(
        1,
        INTERMEDIATE_ROWS * HIDDEN_COLUMNS,
    ));
    let reference = encode_slab(
        &source_weights,
        INTERMEDIATE_ROWS,
        HIDDEN_COLUMNS,
        BitWidth::new(4).expect("valid"),
        GroupSize::new(64).expect("valid"),
        false,
    )
    .expect("the reference encoder should succeed");

    assert_eq!(written_words, reference.packed_words);
}

#[test]
#[parallel]
fn should_fail_with_a_typed_error_when_a_planned_shard_is_missing() {
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    write_two_shard_source(source.path());
    let plan = uniform_plan(source.path(), 4);
    std::fs::remove_file(source.path().join(SECOND_SHARD_NAME))
        .expect("the second shard should be removable");
    let mut silent_reporter = SilentProgressReporter;

    let failure = run_requantize(
        &runtime,
        source.path(),
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        64 * 1024,
        &mut silent_reporter,
    );

    assert!(matches!(
        failure,
        Err(QuantizationError::SafetensorsHeaderUnreadable { .. })
    ));
}
