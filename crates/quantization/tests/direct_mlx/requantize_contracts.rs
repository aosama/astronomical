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

use super::support::{contract_runtime, outlier_heavy_weights};

/// Writes a two-shard bf16 source with a router, two linears, and a norm.
fn write_two_shard_source(directory: &std::path::Path) {
    let _ = (directory, outlier_heavy_weights(1, 1));
    todo!(
        "write model-00001-of-00002 and model-00002-of-00002 with the safetensors crate (bf16 bytes from f32), the index, and a config.json"
    )
}

#[test]
#[ignore = "pending implementation of run_requantize"]
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
#[ignore = "pending implementation of run_requantize"]
fn should_keep_peak_mlx_memory_near_one_tensor_regardless_of_tensor_count() {
    // Requantize a source with N copies of the same tensor and assert the
    // reported peak is bounded by a small multiple of one tensor's float32
    // working copy, the property that lets a 300B checkpoint convert on a
    // laptop. The bound is deliberately loose (8x one tensor) so allocator
    // slack cannot flake it; the point is that it does not scale with N.
    // Peak memory is process-global: when this contract gains its
    // implementation and other MLX tests run concurrently in this binary,
    // it needs #[serial] for the same reason as model-serving's
    // memory_behavior_tests (the sample, not the load rule).
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

    let largest_tensor_float32_bytes = plan
        .entries
        .iter()
        .map(|entry| entry.shape.iter().product::<usize>() as u64 * 4)
        .max()
        .expect("the plan has entries");
    assert!(outcome.peak_mlx_memory_bytes <= 8 * largest_tensor_float32_bytes);
}
