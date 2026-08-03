//! Hermetic tests for quantized expert manifest validation and construction.
//!
//! These tests exercise the pure validation functions and the manifest builder
//! `build_quantized_expert_page_manifest_from_plan` from the expert_paging module.

use super::support::{make_source_interval, synthetic_layer_plan};
use astronomical_model_serving::{
    ExpertManifestError, QuantizationMode, QuantizedTensorSource, SafetensorsDtype,
    build_quantized_expert_page_manifest_from_plan, validate_expert_ids,
    validate_quantization_contract, validate_source_intervals, validate_virtual_intervals,
};

// --- validate_expert_ids ---

#[test]
fn should_accept_valid_ascending_expert_ids() {
    let result = validate_expert_ids(&[0, 5, 42], 256);
    assert!(
        result.is_ok(),
        "should accept valid ascending IDs: {:?}",
        result
    );
    let normalized = result.unwrap();
    assert_eq!(normalized, vec![0, 5, 42]);
}

#[test]
fn should_reject_empty_expert_ids() {
    let result = validate_expert_ids(&[], 256);
    assert!(
        matches!(result, Err(ExpertManifestError::EmptyExpertIds)),
        "should reject empty expert IDs: {:?}",
        result
    );
}

#[test]
fn should_reject_non_ascending_expert_ids() {
    let result = validate_expert_ids(&[3, 1, 7], 256);
    assert!(
        matches!(result, Err(ExpertManifestError::NonAscendingExpertIds)),
        "should reject non-ascending IDs: {:?}",
        result
    );
}

#[test]
fn should_reject_duplicate_expert_ids() {
    // Duplicates violate the "strictly ascending" rule
    let result = validate_expert_ids(&[0, 5, 5, 10], 256);
    assert!(
        matches!(result, Err(ExpertManifestError::NonAscendingExpertIds)),
        "should reject duplicate IDs: {:?}",
        result
    );
}

#[test]
fn should_reject_expert_id_exceeding_capacity() {
    let result = validate_expert_ids(&[0, 255, 300], 256);
    assert!(
        matches!(
            result,
            Err(ExpertManifestError::ExpertIdExceedsCapacity { .. })
        ),
        "should reject IDs exceeding capacity: {:?}",
        result
    );
}

#[test]
fn should_accept_single_expert_id() {
    let result = validate_expert_ids(&[42], 256);
    assert!(result.is_ok(), "should accept single ID: {:?}", result);
}

#[test]
fn should_accept_expert_ids_at_capacity_boundary() {
    let result = validate_expert_ids(&[0, 255], 256);
    assert!(
        result.is_ok(),
        "should accept ID 255 when capacity is 256: {:?}",
        result
    );
}

#[test]
fn should_reject_expert_id_equal_to_capacity() {
    // capacity=256 means max valid ID is 255
    let result = validate_expert_ids(&[256], 256);
    assert!(
        matches!(
            result,
            Err(ExpertManifestError::ExpertIdExceedsCapacity { .. })
        ),
        "should reject ID equal to capacity: {:?}",
        result
    );
}

// --- validate_quantization_contract ---

#[test]
fn should_accept_valid_affine_quantization_contract() {
    let result = validate_quantization_contract(6, 128, QuantizationMode::Affine);
    assert!(
        result.is_ok(),
        "should accept valid 6-bit/128-group affine contract: {:?}",
        result
    );
}

#[test]
fn should_reject_zero_bits() {
    let result = validate_quantization_contract(0, 128, QuantizationMode::Affine);
    assert!(
        matches!(result, Err(ExpertManifestError::InvalidBits)),
        "should reject zero bits: {:?}",
        result
    );
}

#[test]
fn should_reject_negative_bits() {
    let result = validate_quantization_contract(-1, 128, QuantizationMode::Affine);
    assert!(
        matches!(result, Err(ExpertManifestError::InvalidBits)),
        "should reject negative bits: {:?}",
        result
    );
}

#[test]
fn should_reject_zero_group_size() {
    let result = validate_quantization_contract(6, 0, QuantizationMode::Affine);
    assert!(
        matches!(result, Err(ExpertManifestError::InvalidGroupSize)),
        "should reject zero group size: {:?}",
        result
    );
}

#[test]
fn should_reject_negative_group_size() {
    let result = validate_quantization_contract(6, -1, QuantizationMode::Affine);
    assert!(
        matches!(result, Err(ExpertManifestError::InvalidGroupSize)),
        "should reject negative group size: {:?}",
        result
    );
}

#[test]
fn should_reject_group_size_not_packing_into_u32() {
    // 6 bits × 7 groups = 42 bits, which does not pack into whole u32 (32 bits)
    let result = validate_quantization_contract(6, 7, QuantizationMode::Affine);
    assert!(
        matches!(
            result,
            Err(ExpertManifestError::GroupsNotPackedIntoU32 { .. })
        ),
        "should reject group size that doesn't pack into u32: {:?}",
        result
    );
}

#[test]
fn should_reject_quantization_contract_when_packed_bit_count_overflows() {
    let validation_outcome =
        validate_quantization_contract(i32::MAX, i32::MAX, QuantizationMode::Affine);

    assert!(matches!(
        validation_outcome,
        Err(ExpertManifestError::GroupsNotPackedIntoU32 { .. })
    ));
}

#[test]
fn should_accept_4_bit_8_group_contract() {
    // 4 bits × 8 groups = 32 bits = 1 u32 — packs cleanly
    let result = validate_quantization_contract(4, 8, QuantizationMode::Affine);
    assert!(
        result.is_ok(),
        "should accept 4-bit/8-group affine contract: {:?}",
        result
    );
}

// --- validate_source_intervals ---

#[test]
fn should_accept_non_overlapping_source_intervals() {
    let intervals = vec![
        make_source_interval("gate_proj.weight", 100, 200, 0),
        make_source_interval("up_proj.weight", 400, 100, 200),
    ];
    let result = validate_source_intervals(&intervals, 0);
    assert!(
        result.is_ok(),
        "should accept non-overlapping source intervals: {:?}",
        result
    );
}

#[test]
fn should_reject_overlapping_source_intervals() {
    let intervals = vec![
        make_source_interval("gate_proj.weight", 100, 350, 0),
        make_source_interval("up_proj.weight", 300, 100, 350),
    ];
    let result = validate_source_intervals(&intervals, 0);
    assert!(
        matches!(
            result,
            Err(ExpertManifestError::OverlappingSourceIntervals { .. })
        ),
        "should reject overlapping source intervals: {:?}",
        result
    );
}

#[test]
fn should_reject_zero_length_source_interval() {
    let intervals = vec![make_source_interval("gate_proj.weight", 100, 0, 0)];
    let result = validate_source_intervals(&intervals, 0);
    assert!(
        result.is_err(),
        "should reject zero-length source interval: {:?}",
        result
    );
}

#[test]
fn should_reject_source_interval_when_its_end_offset_overflows() {
    let source_intervals = vec![make_source_interval("gate_proj.weight", u64::MAX, 1, 0)];

    let validation_outcome = validate_source_intervals(&source_intervals, 0);

    assert!(matches!(
        validation_outcome,
        Err(ExpertManifestError::SourceIntervalExceedsFile { .. })
    ));
}

// --- validate_virtual_intervals ---

#[test]
fn should_accept_contiguous_virtual_intervals() {
    let intervals = vec![
        make_source_interval("gate_proj.weight", 100, 200, 0),
        make_source_interval("up_proj.weight", 400, 100, 200),
    ];
    let result = validate_virtual_intervals(&intervals, 300);
    assert!(
        result.is_ok(),
        "should accept contiguous virtual intervals: {:?}",
        result
    );
}

#[test]
fn should_reject_non_contiguous_virtual_intervals() {
    let intervals = vec![
        make_source_interval("gate_proj.weight", 100, 200, 0),
        // Gap: expected offset 200, actual 250
        make_source_interval("up_proj.weight", 400, 100, 250),
    ];
    let result = validate_virtual_intervals(&intervals, 350);
    assert!(
        matches!(
            result,
            Err(ExpertManifestError::NonContiguousVirtualIntervals { .. })
        ),
        "should reject non-contiguous virtual intervals: {:?}",
        result
    );
}

#[test]
fn should_reject_virtual_intervals_shortfall() {
    let intervals = vec![make_source_interval("gate_proj.weight", 100, 200, 0)];
    // Declared 500 bytes but only 200 covered
    let result = validate_virtual_intervals(&intervals, 500);
    assert!(
        matches!(
            result,
            Err(ExpertManifestError::VirtualIntervalsShortfall { .. })
        ),
        "should reject virtual intervals shortfall: {:?}",
        result
    );
}

#[test]
fn should_reject_virtual_intervals_when_covered_byte_count_overflows() {
    let virtual_intervals = vec![
        make_source_interval("gate_proj.weight", 0, usize::MAX, 0),
        make_source_interval("up_proj.weight", 0, 1, u64::MAX),
    ];

    let validation_outcome = validate_virtual_intervals(&virtual_intervals, u64::MAX);

    assert!(matches!(
        validation_outcome,
        Err(ExpertManifestError::VirtualIntervalsShortfall {
            actual_bytes: u64::MAX,
            ..
        })
    ));
}

// --- build_quantized_expert_page_manifest_from_plan ---

/// Constructs a synthetic layer plan for a single projection with 8 experts,
/// 6-bit affine quantization, group size 128, hidden dimension 1024.
///
/// The tensor sources have non-overlapping payload offsets ranges to satisfy
/// the validation that source intervals must not overlap.
#[test]
fn should_build_a_native_bfloat16_page_with_only_uncompressed_weight_intervals() {
    let mut native_bfloat16_layer_plan = synthetic_layer_plan("native_bfloat16_layer");
    let native_weight_source = native_bfloat16_layer_plan.tensor_sources.remove(0);
    native_bfloat16_layer_plan.tensor_sources = vec![QuantizedTensorSource {
        tensor_name: native_weight_source.tensor_name,
        projection_name: native_weight_source.projection_name,
        parameter_name: "weight".to_owned(),
        quantization_bits: 0,
        quantization_group_size: 0,
        source_file: native_weight_source.source_file,
        source_file_size_bytes: 16_484,
        dtype: SafetensorsDtype::BFloat16,
        full_shape: vec![8, 4, 1_024],
        tensor_payload_offset: 100,
        bytes_per_expert: 8_192,
        expert_capacity: 8,
    }];
    native_bfloat16_layer_plan.quantization_mode = QuantizationMode::NativeBfloat16;

    let native_bfloat16_page_manifest =
        build_quantized_expert_page_manifest_from_plan(&native_bfloat16_layer_plan, &[1, 3])
            .expect("native BF16 experts should require only their weight intervals");

    assert_eq!(native_bfloat16_page_manifest.payload_byte_count, 16_384);
    assert_eq!(native_bfloat16_page_manifest.source_manifests.len(), 1);
    assert_eq!(
        native_bfloat16_page_manifest.source_manifests[0]
            .tensor_ranges
            .len(),
        1,
        "native BF16 pages must not synthesize affine scale or bias tensors"
    );
}

#[test]
fn should_build_page_manifest_from_synthetic_layer_plan_with_two_experts() {
    let layer_plan = synthetic_layer_plan("language_model.model.layers.0");
    let result = build_quantized_expert_page_manifest_from_plan(&layer_plan, &[0, 7]);
    assert!(
        result.is_ok(),
        "should build page manifest from synthetic layer plan: {:?}",
        result
    );
    let manifest = result.unwrap();
    assert_eq!(manifest.expert_ids, vec![0, 7]);
    assert_eq!(manifest.page_slot_by_global_expert_id.len(), 8);
    assert_eq!(manifest.page_slot_by_global_expert_id[0], 0);
    assert_eq!(manifest.page_slot_by_global_expert_id[7], 1);
    assert_eq!(manifest.page_slot_by_global_expert_id[3], u32::MAX);
    assert!(
        !manifest.source_manifests.is_empty(),
        "should have source manifests"
    );
    assert!(
        manifest.payload_byte_count > 0,
        "should have non-zero payload bytes"
    );
}

#[test]
fn should_map_expert_ids_to_page_slots_correctly() {
    let layer_plan = synthetic_layer_plan("language_model.model.layers.5");
    let result = build_quantized_expert_page_manifest_from_plan(&layer_plan, &[3, 5, 7]);
    let manifest = result.unwrap();
    // Page slots are assigned in order of the sorted expert IDs
    assert_eq!(manifest.page_slot_by_global_expert_id[3], 0);
    assert_eq!(manifest.page_slot_by_global_expert_id[5], 1);
    assert_eq!(manifest.page_slot_by_global_expert_id[7], 2);
}

#[test]
fn should_compute_correct_payload_byte_count() {
    let layer_plan = synthetic_layer_plan("language_model.model.layers.0");
    let result = build_quantized_expert_page_manifest_from_plan(&layer_plan, &[0, 1]);
    let manifest = result.unwrap();
    // Payload byte count should be the sum of all source manifest payload bytes
    assert!(
        manifest.payload_byte_count > 0,
        "payload_byte_count should be positive, got {}",
        manifest.payload_byte_count
    );
}

#[test]
fn should_rebase_loaded_tensor_names_to_projection_parameter_names() {
    eprintln!("[expert-manifest] status=start case=rebase_tensor_names");
    let layer_plan = synthetic_layer_plan("language_model.model.layers.0");
    let manifest = build_quantized_expert_page_manifest_from_plan(&layer_plan, &[0])
        .expect("synthetic layer plan should build a one-expert page manifest");

    let loaded_tensor_names: std::collections::BTreeSet<_> = manifest
        .source_manifests
        .iter()
        .flat_map(|source_manifest| {
            source_manifest
                .tensor_ranges
                .iter()
                .map(|tensor_range| tensor_range.tensor_name.as_str())
        })
        .collect();

    eprintln!("[expert-manifest] status=success loaded_tensor_names={loaded_tensor_names:?}");
    assert!(loaded_tensor_names.contains("gate_proj.weight"));
    assert!(loaded_tensor_names.contains("gate_proj.scales"));
    assert!(loaded_tensor_names.contains("gate_proj.biases"));
    assert!(
        !loaded_tensor_names.contains("language_model.model.layers.0.switch_mlp.weight"),
        "rebased page headers must not expose full checkpoint tensor names to the MLX load result"
    );
}

#[test]
fn should_reject_empty_expert_ids_through_manifest() {
    let layer_plan = synthetic_layer_plan("language_model.model.layers.0");
    let result = build_quantized_expert_page_manifest_from_plan(&layer_plan, &[]);
    assert!(
        matches!(result, Err(ExpertManifestError::EmptyExpertIds)),
        "should reject empty expert IDs: {:?}",
        result
    );
}

#[test]
fn should_reject_non_ascending_expert_ids_through_manifest() {
    let layer_plan = synthetic_layer_plan("language_model.model.layers.0");
    let result = build_quantized_expert_page_manifest_from_plan(&layer_plan, &[3, 1, 7]);
    assert!(
        matches!(result, Err(ExpertManifestError::NonAscendingExpertIds)),
        "should reject non-ascending IDs: {:?}",
        result
    );
}

#[test]
fn should_reject_expert_id_exceeding_capacity_through_manifest() {
    let layer_plan = synthetic_layer_plan("language_model.model.layers.0");
    // expert_capacity is 8, so expert ID 8 exceeds it (valid: 0..7)
    let result = build_quantized_expert_page_manifest_from_plan(&layer_plan, &[0, 5, 8]);
    assert!(
        matches!(
            result,
            Err(ExpertManifestError::ExpertIdExceedsCapacity { .. })
        ),
        "should reject IDs exceeding capacity: {:?}",
        result
    );
}
