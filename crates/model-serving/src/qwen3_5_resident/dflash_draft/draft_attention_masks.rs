//! Per-layer attention routing for the draft trunk.
//!
//! The two layer kinds map onto two existing, separately tested attention
//! paths: sliding-window causal layers reuse the tree's sliding-window mask
//! builder verbatim with the masked attention kernel, and the bidirectional
//! full-attention layer attends to everything resident, which is exactly the
//! unmasked attention kernel. Keeping the mapping in one pure unit makes the
//! visibility contract testable without a graphics processor.

use super::draft_configuration::DFlashDraftLayerAttentionKind;

/// How one draft layer's attention call is routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashDraftAttentionRoute {
    /// Causal within the block and windowed against the context: build the
    /// sliding-window mask and call masked attention.
    SlidingWindowMasked { window_size: i32 },
    /// Every block position sees the whole context and the whole block:
    /// the unmasked attention kernel is the exact semantics.
    Unmasked,
}

/// Resolves one layer kind to its attention route.
#[must_use]
pub fn attention_route_for(
    attention_kind: DFlashDraftLayerAttentionKind,
) -> DFlashDraftAttentionRoute {
    match attention_kind {
        DFlashDraftLayerAttentionKind::SlidingWindowCausal { window_size } => {
            DFlashDraftAttentionRoute::SlidingWindowMasked { window_size }
        }
        DFlashDraftLayerAttentionKind::FullBidirectional => DFlashDraftAttentionRoute::Unmasked,
    }
}

/// The visibility rule one layer kind implies, as a pure predicate.
///
/// `query_absolute_position` and `key_absolute_position` are token positions;
/// `block_first_absolute_position` is where the speculation block begins.
/// This documents the contract the two routes implement and is the unit the
/// tests lock down.
#[must_use]
pub fn position_is_visible_to_layer(
    attention_kind: DFlashDraftLayerAttentionKind,
    query_absolute_position: i32,
    key_absolute_position: i32,
) -> bool {
    match attention_kind {
        DFlashDraftLayerAttentionKind::SlidingWindowCausal { window_size } => {
            key_absolute_position <= query_absolute_position
                && key_absolute_position > query_absolute_position - window_size
        }
        DFlashDraftLayerAttentionKind::FullBidirectional => true,
    }
}
