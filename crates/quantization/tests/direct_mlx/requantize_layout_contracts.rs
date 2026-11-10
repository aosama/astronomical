//! The requantizer's source layouts, each built with MLX itself so the
//! fixture is exactly what a real checkpoint holds: an already-packed
//! affine source, a fused gate-up expert tensor, unfused per-expert
//! tensors, and a block-scaled FP8 weight. Each contract compares the
//! written words with stock `quantize_affine` of the weight the layout
//! should decode to, computed independently in the test.

use std::path::Path;

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::encoder::RangeSearchPolicy;
use astronomical_quantization::module_inventory::{ModuleInventory, ModuleWeightSource};
use astronomical_quantization::progress::{
    ProgressReporter, SilentProgressReporter, StageProgress,
};
use astronomical_quantization::requantize::bit_policy::BitPolicy;
use astronomical_quantization::requantize::driver::run_requantize;
use astronomical_quantization::requantize::tensor_plan::RequantizeTensorPlan;
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::parallel;

use super::support::{contract_runtime, outlier_heavy_weights};

const ROWS: usize = 128;
const COLUMNS: usize = 256;
const SHARD_BYTES_LIMIT: u64 = 64 * 1024;

#[test]
#[parallel]
fn should_dequantize_a_packed_source_with_its_own_parameters_before_re_encoding() {
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let dense_weight = bfloat16_weight(&runtime, &[ROWS, COLUMNS]);
    let (packed, scales, biases) = runtime
        .quantize_affine(&dense_weight, 32, 8)
        .expect("the source should quantize");
    save_source(
        &runtime,
        source.path(),
        &[
            ("model.layers.0.mlp.up_proj.weight", &packed),
            ("model.layers.0.mlp.up_proj.scales", &scales),
            ("model.layers.0.mlp.up_proj.biases", &biases),
        ],
        r#"{"model_type": "qwen3", "quantization": {"bits": 8, "group_size": 32}}"#,
    );
    let decoded_source = runtime
        .dequantize_affine(&packed, &scales, &biases, 32, 8)
        .expect("the source should dequantize");

    let output = requantize_at_four_bits(&runtime, source.path());

    assert_eq!(
        written_words(&runtime, output.path(), "model.layers.0.mlp.up_proj.weight"),
        stock_four_bit_words(&runtime, &decoded_source)
    );
}

#[test]
#[parallel]
fn should_split_a_fused_gate_up_tensor_into_its_lower_and_upper_halves() {
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let expert_count = 2;
    let fused_weight = bfloat16_weight(&runtime, &[expert_count, 2 * ROWS, COLUMNS]);
    save_source(
        &runtime,
        source.path(),
        &[("model.layers.0.mlp.experts.gate_up_proj", &fused_weight)],
        r#"{"model_type": "qwen3_moe"}"#,
    );
    let upper_half = runtime
        .slice(
            &fused_weight,
            &[0, ROWS as i32, 0],
            &[expert_count as i32, 2 * ROWS as i32, COLUMNS as i32],
            &[1, 1, 1],
        )
        .expect("the upper half should slice");
    let up_projection_name = fused_half_module_name(source.path(), 1);

    let output = requantize_at_four_bits(&runtime, source.path());

    assert_eq!(
        written_words(
            &runtime,
            output.path(),
            &format!("{up_projection_name}.weight")
        ),
        stock_four_bit_words(&runtime, &upper_half)
    );
}

#[test]
#[parallel]
fn should_stack_unfused_per_expert_tensors_in_expert_order() {
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let first_expert = bfloat16_weight(&runtime, &[ROWS, COLUMNS]);
    let second_expert = runtime
        .multiply(&first_expert, &MlxArray::from_scalar_f32(-0.5))
        .expect("the second expert should differ");
    let second_expert = runtime
        .astype(&second_expert, MlxDtype::BFloat16)
        .expect("cast");
    save_source(
        &runtime,
        source.path(),
        &[
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                &first_expert,
            ),
            (
                "model.layers.0.mlp.experts.1.gate_proj.weight",
                &second_expert,
            ),
        ],
        r#"{"model_type": "qwen3_moe"}"#,
    );
    let expected_stack = runtime
        .stack_axis(&[&first_expert, &second_expert], 0)
        .expect("stack");
    let stacked_module_name = only_module_name(source.path());

    let output = requantize_at_four_bits(&runtime, source.path());

    assert_eq!(
        written_words(
            &runtime,
            output.path(),
            &format!("{stacked_module_name}.weight")
        ),
        stock_four_bit_words(&runtime, &expected_stack)
    );
}

#[test]
#[parallel]
fn should_multiply_each_fp8_tile_by_its_block_scale() {
    // Powers of two as block scales keep the bf16 products exact, so the
    // host-side expectation and the GPU dequantization agree bit for bit.
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let dense_weight = bfloat16_weight(&runtime, &[ROWS, COLUMNS]);
    let fp8_weight = runtime.to_fp8(&dense_weight).expect("to fp8");
    let block_scales = [0.5_f32, 2.0];
    let block_scale_array = runtime
        .array_from_f32(&block_scales, &[1, 2])
        .expect("scales");
    save_source(
        &runtime,
        source.path(),
        &[
            ("model.layers.0.mlp.up_proj.weight", &fp8_weight),
            (
                "model.layers.0.mlp.up_proj.weight_scale_inv",
                &block_scale_array,
            ),
        ],
        r#"{"model_type": "qwen3"}"#,
    );
    let widened_values = runtime
        .astype(
            &runtime
                .from_fp8(&fp8_weight, MlxDtype::BFloat16)
                .expect("from fp8"),
            MlxDtype::Float32,
        )
        .expect("cast")
        .to_vec_f32()
        .expect("read back");
    let expected_dense = block_scaled_on_host(&runtime, &widened_values, &block_scales);

    let output = requantize_at_four_bits(&runtime, source.path());

    assert_eq!(
        written_words(&runtime, output.path(), "model.layers.0.mlp.up_proj.weight"),
        stock_four_bit_words(&runtime, &expected_dense)
    );
}

#[test]
#[parallel]
fn should_report_progress_ending_at_every_planned_tensor() {
    let runtime = contract_runtime();
    let source = tempfile::tempdir().expect("a temporary directory should be creatable");
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    let dense_weight = bfloat16_weight(&runtime, &[ROWS, COLUMNS]);
    save_source(
        &runtime,
        source.path(),
        &[
            ("model.layers.0.mlp.up_proj.weight", &dense_weight),
            ("model.layers.0.mlp.down_proj.weight", &dense_weight),
        ],
        r#"{"model_type": "qwen3"}"#,
    );
    let plan = four_bit_plan(source.path());
    let mut recording_reporter = RecordingProgressReporter::default();

    run_requantize(
        &runtime,
        source.path(),
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        SHARD_BYTES_LIMIT,
        &mut recording_reporter,
    )
    .expect("requantization should succeed");

    let final_report = recording_reporter
        .reports
        .last()
        .expect("at least one report");
    assert_eq!(final_report.stage_name, "requantize");
    assert_eq!(
        final_report.completed_work_units,
        plan.tensor_count() as u64
    );
    assert_eq!(final_report.total_work_units, plan.tensor_count() as u64);
}

#[derive(Default)]
struct RecordingProgressReporter {
    reports: Vec<StageProgress>,
}

impl ProgressReporter for RecordingProgressReporter {
    fn report(&mut self, progress: &StageProgress) {
        self.reports.push(progress.clone());
    }
}

/// Deterministic bf16 weights of the given shape.
fn bfloat16_weight(runtime: &MlxRuntime, shape: &[usize]) -> MlxArray {
    let element_count: usize = shape.iter().product();
    let mut signed_shape = Vec::with_capacity(shape.len());
    for extent in shape {
        signed_shape.push(*extent as i32);
    }
    let float32_weight = runtime
        .array_from_f32(&outlier_heavy_weights(1, element_count), &signed_shape)
        .expect("the weights array should be created");
    runtime
        .astype(&float32_weight, MlxDtype::BFloat16)
        .expect("cast")
}

/// One-shard source written through MLX's own safetensors writer.
fn save_source(
    runtime: &MlxRuntime,
    directory: &Path,
    named_arrays: &[(&str, &MlxArray)],
    config_json: &str,
) {
    let shard_file = std::fs::File::create(directory.join("model.safetensors"))
        .expect("the source shard should be creatable");
    runtime
        .save_safetensors(shard_file, named_arrays, &[("format", "mlx")])
        .expect("the source shard should save");
    std::fs::write(directory.join("config.json"), config_json)
        .expect("the source config should be writable");
}

fn four_bit_plan(source: &Path) -> RequantizeTensorPlan {
    let inventory =
        ModuleInventory::from_source_directory(source, GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    RequantizeTensorPlan::from_inventory(
        &inventory,
        &BitPolicy::Uniform {
            bit_width: BitWidth::new(4).expect("valid"),
            group_size: GroupSize::new(64).expect("valid"),
        },
    )
}

/// Requantizes at 4 bits, where the default policy is stock
/// `quantize_affine`, so the expectation needs no search.
fn requantize_at_four_bits(runtime: &MlxRuntime, source: &Path) -> tempfile::TempDir {
    let output = tempfile::tempdir().expect("a temporary directory should be creatable");
    let plan = four_bit_plan(source);
    let mut silent_reporter = SilentProgressReporter;
    run_requantize(
        runtime,
        source,
        output.path(),
        &plan,
        RangeSearchPolicy::default_policy(),
        SHARD_BYTES_LIMIT,
        &mut silent_reporter,
    )
    .expect("requantization should succeed");
    output
}

/// The packed words written under `tensor_key`, found through the index.
fn written_words(runtime: &MlxRuntime, output: &Path, tensor_key: &str) -> Vec<u32> {
    let index_text = std::fs::read_to_string(output.join("model.safetensors.index.json"))
        .expect("the index should be readable");
    let index_document: serde_json::Value =
        serde_json::from_str(&index_text).expect("the index should parse");
    let shard_name = index_document["weight_map"][tensor_key]
        .as_str()
        .expect("the index maps the tensor");
    let shard_file = std::fs::File::open(output.join(shard_name)).expect("the shard should open");
    runtime
        .load_safetensors(shard_file, None)
        .expect("the shard should load")
        .tensor(tensor_key)
        .expect("the tensor should be in its shard")
        .to_vec_u32()
        .expect("packed words read back")
}

fn stock_four_bit_words(runtime: &MlxRuntime, expected_weight: &MlxArray) -> Vec<u32> {
    let (packed, _, _) = runtime
        .quantize_affine(expected_weight, 64, 4)
        .expect("stock quantization should succeed");
    packed.to_vec_u32().expect("read back")
}

/// The inventory's name for the fused tensor's lower (0) or upper (1) half.
fn fused_half_module_name(source: &Path, wanted_half_index: u8) -> String {
    let inventory =
        ModuleInventory::from_source_directory(source, GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    for module in inventory.modules() {
        if let ModuleWeightSource::FusedGateUpHalf { half_index, .. } = module.weight_source
            && half_index == wanted_half_index
        {
            return module.module_name.clone();
        }
    }
    panic!("the inventory has no fused half {wanted_half_index}");
}

fn only_module_name(source: &Path) -> String {
    let inventory =
        ModuleInventory::from_source_directory(source, GroupSize::new(64).expect("valid"))
            .expect("the inventory should build");
    assert_eq!(inventory.module_count(), 1);
    inventory
        .modules()
        .next()
        .expect("one module")
        .module_name
        .clone()
}

/// Each 128x128 tile multiplied by its scale, computed on the host.
fn block_scaled_on_host(
    runtime: &MlxRuntime,
    widened_values: &[f32],
    block_scales: &[f32],
) -> MlxArray {
    let column_blocks = COLUMNS.div_ceil(128);
    let mut scaled_values = Vec::with_capacity(widened_values.len());
    for (element_index, widened_value) in widened_values.iter().enumerate() {
        let row_block = element_index / COLUMNS / 128;
        let column_block = element_index % COLUMNS / 128;
        scaled_values.push(widened_value * block_scales[row_block * column_blocks + column_block]);
    }
    let scaled_array = runtime
        .array_from_f32(&scaled_values, &[ROWS as i32, COLUMNS as i32])
        .expect("array");
    runtime
        .astype(&scaled_array, MlxDtype::BFloat16)
        .expect("cast")
}
