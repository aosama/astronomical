//! Unit tests for the layer-input reuse arithmetic.

use super::*;
use crate::quantization_probe::module_locations::{AttentionProjection, FeedForwardProjection};

#[test]
fn should_keep_every_layer_input_for_a_small_base() {
    assert_eq!(recording_layer_stride(24, 1_024, 248_320), 1);
}

#[test]
fn should_keep_fewer_layer_inputs_for_a_wide_deep_base() {
    assert_eq!(recording_layer_stride(32, 4_096, 248_320), 3);
}

#[test]
fn should_keep_every_layer_input_at_the_budget_boundary() {
    // Four parts of 10 layers x 25 wide is exactly a vocabulary of 1,000.
    assert_eq!(recording_layer_stride(10, 25, 1_000), 1);
    assert_eq!(recording_layer_stride(10, 25, 999), 2);
}

#[test]
fn should_never_return_a_zero_stride() {
    assert_eq!(recording_layer_stride(0, 0, 0), 1);
}

#[test]
fn should_resume_a_decoder_layer_swap_at_its_own_layer() {
    let attention_location = ModuleLocation::DecoderLayerAttention {
        decoder_layer_index: 7,
        projection: AttentionProjection::Output,
    };
    let router_location = ModuleLocation::DecoderLayerRouter {
        decoder_layer_index: 3,
    };
    let expert_location = ModuleLocation::DecoderLayerRoutedExperts {
        decoder_layer_index: 5,
        projection: FeedForwardProjection::Down,
    };

    assert_eq!(
        first_affected_decoder_layer(attention_location, 24),
        Some(7)
    );
    assert_eq!(first_affected_decoder_layer(router_location, 24), Some(3));
    assert_eq!(first_affected_decoder_layer(expert_location, 24), Some(5));
}

#[test]
fn should_resume_a_head_swap_after_the_last_layer() {
    assert_eq!(
        first_affected_decoder_layer(ModuleLocation::LanguageModelHead, 24),
        Some(24)
    );
}

#[test]
fn should_not_resume_an_embedding_swap() {
    assert_eq!(
        first_affected_decoder_layer(ModuleLocation::Embedding, 24),
        None
    );
}

#[test]
fn should_resume_from_the_latest_record_at_or_before_the_swapped_layer() {
    let recorded_layer_indices = [0, 3, 6, 9, 12];

    assert_eq!(resume_record_position(&recorded_layer_indices, 0), Some(0));
    assert_eq!(resume_record_position(&recorded_layer_indices, 5), Some(1));
    assert_eq!(resume_record_position(&recorded_layer_indices, 6), Some(2));
    assert_eq!(resume_record_position(&recorded_layer_indices, 12), Some(4));
}

#[test]
fn should_find_no_record_when_none_was_kept() {
    assert_eq!(resume_record_position(&[], 4), None);
    assert_eq!(resume_record_position(&[6], 4), None);
}
