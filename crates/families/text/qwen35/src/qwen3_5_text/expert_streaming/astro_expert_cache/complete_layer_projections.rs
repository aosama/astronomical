//! One routed-expert projection of a complete layer, read or replaced across
//! every expert at once: what the quantization sensitivity sweep does when it
//! re-encodes a layer's gate, up or down projection at another width, runs
//! the model and puts the original back.
//!
//! Only a complete layer can be read or replaced this way, one whose table holds every
//! expert with each expert in the slot of its own id (a complete layer is
//! seated with experts `0..capacity` in order). A replacement is then laid out
//! exactly as the router's expert ids index it, the same as the artifact's
//! own expert axis, so the caller can build it from the artifact.

use super::AstroExpertCache;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;

/// One of a routed expert's three projections.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedExpertProjection {
    Gate,
    Up,
    Down,
}

impl AstroExpertCache {
    /// The projection's weights across every expert of a complete layer, or
    /// `None` when the layer is not held complete in expert order.
    #[must_use]
    pub fn complete_layer_projection(
        &self,
        layer_index: usize,
        expert_capacity: usize,
        projection: RoutedExpertProjection,
    ) -> Option<&Qwen3_5AffineWeights> {
        if !self.holds_complete_layer_in_expert_order(layer_index, expert_capacity) {
            return None;
        }
        let table = self.tables_by_layer.get(layer_index)?.as_ref()?;
        Some(match projection {
            RoutedExpertProjection::Gate => &table.weights.gate_projection,
            RoutedExpertProjection::Up => &table.weights.up_projection,
            RoutedExpertProjection::Down => &table.weights.down_projection,
        })
    }

    /// Puts `replacement` in place of the projection across every expert of
    /// a complete layer and returns what was there, or gives `replacement`
    /// back unused when the layer is not held complete in expert order.
    ///
    /// The table's byte accounting is left as it was: a replacement lasts
    /// one probe and the original goes back before anything is admitted.
    pub fn replace_complete_layer_projection(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        projection: RoutedExpertProjection,
        replacement: Qwen3_5AffineWeights,
    ) -> Result<Qwen3_5AffineWeights, Qwen3_5AffineWeights> {
        if !self.holds_complete_layer_in_expert_order(layer_index, expert_capacity) {
            return Err(replacement);
        }
        let Some(Some(table)) = self.tables_by_layer.get_mut(layer_index) else {
            return Err(replacement);
        };
        let held_projection = match projection {
            RoutedExpertProjection::Gate => &mut table.weights.gate_projection,
            RoutedExpertProjection::Up => &mut table.weights.up_projection,
            RoutedExpertProjection::Down => &mut table.weights.down_projection,
        };
        Ok(std::mem::replace(held_projection, replacement))
    }

    /// Whether the layer's table holds every expert, each in the slot of its
    /// own id.
    fn holds_complete_layer_in_expert_order(
        &self,
        layer_index: usize,
        expert_capacity: usize,
    ) -> bool {
        let Some(Some(table)) = self.tables_by_layer.get(layer_index) else {
            return false;
        };
        expert_capacity > 0
            && table.slot_count() == expert_capacity
            && is_identity_slot_order(&table.expert_id_by_slot)
    }
}

/// Pure: whether every slot holds the expert whose id is its slot number.
pub(super) fn is_identity_slot_order(expert_id_by_slot: &[Option<usize>]) -> bool {
    expert_id_by_slot
        .iter()
        .enumerate()
        .all(|(slot, expert_id)| *expert_id == Some(slot))
}

#[cfg(test)]
#[path = "complete_layer_projections_tests.rs"]
mod complete_layer_projections_tests;
