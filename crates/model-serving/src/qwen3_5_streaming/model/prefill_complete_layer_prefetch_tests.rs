use std::path::PathBuf;

use super::prefill_complete_layer_prefetch::{
    layer_plan_allows_prefill_prefetch, next_prefill_prefetch_layer_index,
    prefill_chunk_routes_complete_layer,
};
use crate::{QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype};

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
                    expert_stride_bytes: 8,
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
