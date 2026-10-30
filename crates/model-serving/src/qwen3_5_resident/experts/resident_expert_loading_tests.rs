//! Beside-source unit contracts for the pure resident-expert plan summaries in
//! `resident_expert_loading.rs`. These run before any model file is touched:
//! the payload total, the expert-entry count, and the distinct source-file set
//! are derived purely from the validated layer plans, so their overflow,
//! deduplication, and empty-input behavior are pinned here without a graphics
//! processor.

use std::path::PathBuf;

use crate::{QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype};

use super::{
    collect_source_file_paths, complete_model_expert_entry_count, complete_model_payload_bytes,
};

fn tensor_source(
    source_file: &str,
    bytes_per_expert: usize,
    expert_capacity: usize,
) -> QuantizedTensorSource {
    QuantizedTensorSource {
        tensor_name: "fictional.expert.weight".to_owned(),
        projection_name: "fictional_projection".to_owned(),
        parameter_name: "weight".to_owned(),
        quantization_bits: 4,
        quantization_group_size: 64,
        source_file: PathBuf::from(source_file),
        source_file_size_bytes: 1_000,
        dtype: SafetensorsDtype::Uint32,
        full_shape: vec![4, 2, 2],
        tensor_payload_offset: 0,
        bytes_per_expert,
        expert_capacity,
    }
}

fn layer_plan(
    layer_prefix: &str,
    source_file: &str,
    bytes_per_expert: usize,
    expert_capacity: usize,
) -> QuantizedExpertLayerPlan {
    QuantizedExpertLayerPlan {
        layer_prefix: layer_prefix.to_owned(),
        tensor_sources: vec![tensor_source(
            source_file,
            bytes_per_expert,
            expert_capacity,
        )],
        jangtq2_projection_rotations: Vec::new(),
        expert_capacity,
        quantization_bits: 4,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}

#[test]
fn should_sum_complete_payload_bytes_across_layer_plans() {
    // Two consistent layers: each contributes bytes_per_expert x capacity.
    let layer_plans = vec![
        layer_plan("fictional.layers.0", "shard-a.safetensors", 10, 4),
        layer_plan("fictional.layers.1", "shard-a.safetensors", 5, 4),
    ];

    let complete_model_payload_bytes = complete_model_payload_bytes(&layer_plans)
        .expect("consistent fictional layer plans should total cleanly");

    assert_eq!(complete_model_payload_bytes, 10 * 4 + 5 * 4);
}

#[test]
fn should_fail_the_payload_total_when_a_layer_plan_has_inconsistent_geometry() {
    // The tensor source claims capacity 2 while the layer claims 4; the
    // plan-level consistency check must reject it before any storage is
    // reserved for the resident owner.
    let mut inconsistent_layer_plan =
        layer_plan("fictional.layers.0", "shard-a.safetensors", 10, 4);
    inconsistent_layer_plan.tensor_sources[0].expert_capacity = 2;

    let payload_total = complete_model_payload_bytes(&[inconsistent_layer_plan]);

    assert!(
        payload_total.is_err(),
        "inconsistent fictional geometry must fail the payload total"
    );
}

#[test]
fn should_count_expert_entries_by_summing_layer_capacities() {
    let layer_plans = vec![
        layer_plan("fictional.layers.0", "shard-a.safetensors", 10, 4),
        layer_plan("fictional.layers.1", "shard-b.safetensors", 10, 7),
    ];

    assert_eq!(complete_model_expert_entry_count(&layer_plans), 11);
}

#[test]
fn should_return_zero_counts_and_no_source_files_for_an_empty_plan_set() {
    let empty_layer_plans: Vec<QuantizedExpertLayerPlan> = Vec::new();

    assert_eq!(complete_model_expert_entry_count(&empty_layer_plans), 0);
    assert_eq!(
        complete_model_payload_bytes(&empty_layer_plans)
            .expect("an empty plan set should total to zero bytes"),
        0
    );
    assert!(collect_source_file_paths(&empty_layer_plans).is_empty());
}

#[test]
fn should_deduplicate_shared_source_files_across_layer_plans() {
    // Both layers read shard-a; one also reads shard-b. The distinct file set
    // must contain shard-a exactly once so the shard-loading loop opens it a
    // single time.
    let layer_plans = vec![
        layer_plan("fictional.layers.0", "shard-a.safetensors", 10, 4),
        layer_plan("fictional.layers.1", "shard-a.safetensors", 5, 4),
        layer_plan("fictional.layers.2", "shard-b.safetensors", 8, 4),
    ];

    let source_file_paths = collect_source_file_paths(&layer_plans);

    assert_eq!(source_file_paths.len(), 2);
    assert!(source_file_paths.contains(&PathBuf::from("shard-a.safetensors")));
    assert!(source_file_paths.contains(&PathBuf::from("shard-b.safetensors")));
}
