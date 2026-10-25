//! Visibility-contract tests for the DFlash draft attention routing.

use super::draft_attention_masks::{
    DFlashDraftAttentionRoute, attention_route_for, position_is_visible_to_layer,
};
use super::draft_configuration::DFlashDraftLayerAttentionKind;

const SLIDING: DFlashDraftLayerAttentionKind =
    DFlashDraftLayerAttentionKind::SlidingWindowCausal { window_size: 4096 };
const BIDIRECTIONAL: DFlashDraftLayerAttentionKind =
    DFlashDraftLayerAttentionKind::FullBidirectional;

#[test]
fn should_route_sliding_layers_to_the_masked_kernel_and_full_layers_to_the_unmasked_kernel() {
    assert_eq!(
        attention_route_for(SLIDING),
        DFlashDraftAttentionRoute::SlidingWindowMasked { window_size: 4096 }
    );
    assert_eq!(
        attention_route_for(BIDIRECTIONAL),
        DFlashDraftAttentionRoute::Unmasked
    );
}

#[test]
fn should_let_sliding_layers_see_earlier_positions_inside_the_window() {
    // Query at position 5000 with a 4096 window sees keys from 905 (the
    // first position inside the window) through 5000 inclusive.
    assert!(position_is_visible_to_layer(SLIDING, 5000, 5000));
    assert!(position_is_visible_to_layer(SLIDING, 5000, 905));
    // Position 0 lies 5000 back — beyond the 4096 window — so it is hidden.
    assert!(!position_is_visible_to_layer(SLIDING, 5000, 0));
}

#[test]
fn should_hide_later_positions_from_sliding_layers() {
    // A mask position at 5001 must not see its block successor at 5002:
    // sliding layers stay causal inside the speculation block.
    assert!(!position_is_visible_to_layer(SLIDING, 5001, 5002));
    // Exactly at the window edge (4096 tokens back) visibility is false;
    // one position inside it is true.
    assert!(!position_is_visible_to_layer(SLIDING, 5000, 904));
    assert!(position_is_visible_to_layer(SLIDING, 5000, 905));
}

#[test]
fn should_let_bidirectional_layers_see_every_position() {
    // The full-attention layer sees the whole context and the whole block,
    // including later block positions — the block-diffusion parallelism.
    assert!(position_is_visible_to_layer(BIDIRECTIONAL, 5001, 5002));
    assert!(position_is_visible_to_layer(BIDIRECTIONAL, 5001, 0));
    assert!(position_is_visible_to_layer(BIDIRECTIONAL, 0, 900000));
}
