use std::path::PathBuf;

use super::prefill_complete_layer_prefetch::{
    largest_prefill_prefetchable_complete_source_payload_bytes, layer_plan_allows_prefill_prefetch,
    next_prefill_prefetch_layer_index, prefill_chunk_routes_complete_layer,
    prefill_complete_layer_stream_slot_bytes, residency_target_allows_prefill_prefetch,
};
use crate::{
    ExpertLayerResidencyTarget, QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
    SafetensorsDtype,
};

const PREFILL_CHUNK_TOKEN_COUNT: i32 = 2_048;
const SINGLE_DECODE_TOKEN_COUNT: i32 = 1;

/// A three-projection layer plan whose projections store the given
/// `(projection_name, parameter_names)` tensors; geometry is irrelevant here.
fn layer_plan_storing(projection_parameters: &[(&str, &[&str])]) -> QuantizedExpertLayerPlan {
    let tensor_sources = projection_parameters
        .iter()
        .flat_map(|(projection_name, parameter_names)| {
            parameter_names
                .iter()
                .map(move |parameter_name| QuantizedTensorSource {
                    tensor_name: format!(
                        "layers.0.mlp.switch_mlp.{projection_name}.{parameter_name}"
                    ),
                    projection_name: (*projection_name).to_owned(),
                    parameter_name: (*parameter_name).to_owned(),
                    quantization_bits: 2,
                    quantization_group_size: 64,
                    source_file: PathBuf::from("fictional-model.safetensors"),
                    source_file_size_bytes: 1_000,
                    dtype: SafetensorsDtype::Uint32,
                    full_shape: vec![4, 2, 2],
                    tensor_payload_offset: 0,
                    bytes_per_expert: 8,
                    expert_capacity: 4,
                })
        })
        .collect();
    QuantizedExpertLayerPlan {
        layer_prefix: "layers.0.mlp.switch_mlp".to_owned(),
        tensor_sources,
        jangtq2_projection_rotations: Vec::new(),
        expert_capacity: 4,
        quantization_bits: 2,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}

const AFFINE_PARAMETERS: &[&str] = &["weight", "scales", "biases"];
const CODEBOOK_PARAMETERS: &[&str] = &["tq2_packed", "tq2_scales"];

/// Forward-time page peak of the 122B JANGTQ2 model (packed plus decoded).
const JANGTQ2_FORWARD_TIME_EXPERT_PAGE_PEAK_BYTES: u64 = 5_740_429_312;

#[test]
fn should_charge_the_read_ahead_layer_its_packed_source_not_the_decoded_peak() {
    let packed_source_bytes = 700_000_000;

    let stream_slot_bytes = prefill_complete_layer_stream_slot_bytes(
        JANGTQ2_FORWARD_TIME_EXPERT_PAGE_PEAK_BYTES,
        packed_source_bytes,
    );

    assert_eq!(
        stream_slot_bytes,
        JANGTQ2_FORWARD_TIME_EXPERT_PAGE_PEAK_BYTES + packed_source_bytes
    );
    // The regression this pins: doubling the decoded peak reserved 11.48 GB.
    assert!(stream_slot_bytes < JANGTQ2_FORWARD_TIME_EXPERT_PAGE_PEAK_BYTES * 2);
}

#[test]
fn should_reserve_two_whole_layers_when_pages_need_no_decode() {
    // Plain MLX quantized layers multiply their packed page directly, so the
    // forward-time peak equals the packed source and both slots are equal.
    assert_eq!(
        prefill_complete_layer_stream_slot_bytes(855_638_016, 855_638_016),
        1_711_276_032
    );
}

#[test]
fn should_saturate_instead_of_overflowing_the_stream_slot() {
    assert_eq!(
        prefill_complete_layer_stream_slot_bytes(u64::MAX, 1),
        u64::MAX
    );
    assert_eq!(prefill_complete_layer_stream_slot_bytes(0, 0), 0);
}

#[test]
fn should_treat_a_full_prefill_chunk_as_routing_every_expert() {
    assert!(prefill_chunk_routes_complete_layer(2_048, 8, 256));
}

#[test]
fn should_require_exactly_eight_assignments_per_expert_at_the_boundary() {
    assert!(prefill_chunk_routes_complete_layer(256, 8, 256));
    assert!(!prefill_chunk_routes_complete_layer(255, 8, 256));
}

#[test]
fn should_never_prefetch_for_a_decode_token_or_an_invalid_shape() {
    assert!(!prefill_chunk_routes_complete_layer(1, 2_048, 256));
    assert!(!prefill_chunk_routes_complete_layer(0, 8, 256));
    assert!(!prefill_chunk_routes_complete_layer(-4, 8, 256));
    assert!(!prefill_chunk_routes_complete_layer(2_048, 8, 0));
}

#[test]
fn should_refuse_prefetch_only_for_layers_the_plan_is_releasing() {
    assert!(!residency_target_allows_prefill_prefetch(Some(
        ExpertLayerResidencyTarget::ReleasePartial
    )));
    assert!(!residency_target_allows_prefill_prefetch(Some(
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
    )));
    for streaming_target in [
        None,
        Some(ExpertLayerResidencyTarget::StreamOperationLocal),
        Some(ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead),
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
    ] {
        assert!(residency_target_allows_prefill_prefetch(streaming_target));
    }
}

#[test]
fn should_pick_the_nearest_later_candidate_layer() {
    let streamed_layers = [3_usize, 7, 9];
    assert_eq!(
        next_prefill_prefetch_layer_index(3, 10, |layer_index| streamed_layers
            .contains(&layer_index)),
        Some(7)
    );
    assert_eq!(
        next_prefill_prefetch_layer_index(0, 10, |layer_index| streamed_layers
            .contains(&layer_index)),
        Some(3)
    );
}

#[test]
fn should_find_no_candidate_after_the_last_layer_or_when_none_stream() {
    assert_eq!(next_prefill_prefetch_layer_index(9, 10, |_| true), None);
    assert_eq!(
        next_prefill_prefetch_layer_index(usize::MAX, 10, |_| true),
        None
    );
    assert_eq!(next_prefill_prefetch_layer_index(0, 10, |_| false), None);
}

#[test]
fn should_allow_prefill_prefetch_for_an_affine_layer_with_or_without_the_fused_kernel() {
    let affine_layer_plan = layer_plan_storing(&[
        ("gate_proj", AFFINE_PARAMETERS),
        ("up_proj", AFFINE_PARAMETERS),
        ("down_proj", AFFINE_PARAMETERS),
    ]);

    assert!(layer_plan_allows_prefill_prefetch(
        &affine_layer_plan,
        PREFILL_CHUNK_TOKEN_COUNT,
        false
    ));
    assert!(layer_plan_allows_prefill_prefetch(
        &affine_layer_plan,
        PREFILL_CHUNK_TOKEN_COUNT,
        true
    ));
}

#[test]
fn should_refuse_prefill_prefetch_for_a_codebook_layer_even_with_the_fused_kernel() {
    // The regression this pins: every codebook layer was planned, admitted and
    // booked as a full-layer source plan, then dropped, once per chunk.
    let codebook_layer_plan = layer_plan_storing(&[
        ("gate_proj", CODEBOOK_PARAMETERS),
        ("up_proj", CODEBOOK_PARAMETERS),
        ("down_proj", CODEBOOK_PARAMETERS),
    ]);

    assert!(!layer_plan_allows_prefill_prefetch(
        &codebook_layer_plan,
        PREFILL_CHUNK_TOKEN_COUNT,
        true
    ));
    assert!(!layer_plan_allows_prefill_prefetch(
        &codebook_layer_plan,
        2,
        true
    ));
}

#[test]
fn should_refuse_prefill_prefetch_when_any_projection_of_a_mixed_layer_is_codebook() {
    let mixed_layer_plan = layer_plan_storing(&[
        ("gate_proj", AFFINE_PARAMETERS),
        ("up_proj", AFFINE_PARAMETERS),
        ("down_proj", CODEBOOK_PARAMETERS),
    ]);

    assert!(!layer_plan_allows_prefill_prefetch(
        &mixed_layer_plan,
        PREFILL_CHUNK_TOKEN_COUNT,
        true
    ));
}

#[test]
fn should_mirror_the_page_rule_for_a_single_codebook_decode_token() {
    let codebook_layer_plan = layer_plan_storing(&[("down_proj", CODEBOOK_PARAMETERS)]);

    assert!(layer_plan_allows_prefill_prefetch(
        &codebook_layer_plan,
        SINGLE_DECODE_TOKEN_COUNT,
        true
    ));
    assert!(!layer_plan_allows_prefill_prefetch(
        &codebook_layer_plan,
        SINGLE_DECODE_TOKEN_COUNT,
        false
    ));
}

#[test]
fn should_allow_prefill_prefetch_for_a_native_bfloat16_layer() {
    let bfloat16_layer_plan = layer_plan_storing(&[
        ("gate_proj", &["weight"]),
        ("up_proj", &["weight"]),
        ("down_proj", &["weight"]),
    ]);

    assert!(layer_plan_allows_prefill_prefetch(
        &bfloat16_layer_plan,
        PREFILL_CHUNK_TOKEN_COUNT,
        false
    ));
}

/// Each fixture tensor holds 8 bytes per expert across 4 experts.
const FIXTURE_TENSOR_COMPLETE_PAYLOAD_BYTES: u64 = 32;

#[test]
fn should_reserve_no_read_ahead_slot_for_an_all_codebook_bundle() {
    let codebook_layer_plan = layer_plan_storing(&[
        ("gate_proj", CODEBOOK_PARAMETERS),
        ("up_proj", CODEBOOK_PARAMETERS),
        ("down_proj", CODEBOOK_PARAMETERS),
    ]);

    assert_eq!(
        largest_prefill_prefetchable_complete_source_payload_bytes(&[
            codebook_layer_plan.clone(),
            codebook_layer_plan,
        ])
        .expect("fictional codebook geometry should be valid"),
        0
    );
}

#[test]
fn should_size_the_read_ahead_slot_from_the_largest_prefetchable_layer_only() {
    // The codebook layer is the largest on disk but can never be read ahead,
    // so it must not set the slot size.
    let large_codebook_layer_plan = layer_plan_storing(&[
        (
            "gate_proj",
            &["tq2_packed", "tq2_scales", "tq2_extra_one", "tq2_extra_two"],
        ),
        ("up_proj", CODEBOOK_PARAMETERS),
        ("down_proj", CODEBOOK_PARAMETERS),
    ]);
    let small_affine_layer_plan = layer_plan_storing(&[("gate_proj", AFFINE_PARAMETERS)]);
    let larger_affine_layer_plan = layer_plan_storing(&[
        ("gate_proj", AFFINE_PARAMETERS),
        ("up_proj", AFFINE_PARAMETERS),
    ]);

    assert_eq!(
        largest_prefill_prefetchable_complete_source_payload_bytes(&[
            large_codebook_layer_plan,
            small_affine_layer_plan,
            larger_affine_layer_plan,
        ])
        .expect("fictional mixed geometry should be valid"),
        6 * FIXTURE_TENSOR_COMPLETE_PAYLOAD_BYTES
    );
}

#[test]
fn should_reserve_no_read_ahead_slot_for_a_model_without_expert_layers() {
    assert_eq!(
        largest_prefill_prefetchable_complete_source_payload_bytes(&[])
            .expect("an empty layer list has no geometry to overflow"),
        0
    );
}
