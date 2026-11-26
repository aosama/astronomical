//! Which layers of an in-graph forward get a lookup, and the counts of
//! those that do not.
//!
//! A layer of an in-graph forward is served by a lookup only when the
//! reply's plan keeps experts on it and its table is one a lookup can
//! serve through. Every other layer is served some other way, and all but
//! a layer held whole stop at the host for it. A fallback nobody counts
//! cannot be measured, so each is counted by its reason here, with the
//! tables grown ahead of a forward.

use text_generation::execution_error::TextExecutionError;

use super::Qwen3_5Model;
use crate::qwen3_5_text::expert_streaming::TableNotLookedUp;

/// A layer's table with the layer's size, as a lookup is built from it.
pub(super) struct LookupTable {
    pub(super) table: crate::qwen3_5_text::expert_streaming::PartlyHeldTable,
    pub(super) expert_capacity: usize,
}

/// Why a layer of an in-graph forward got no lookup.
#[derive(Clone, Copy, Debug)]
pub(super) enum LayerNotLookedUp {
    /// The reply's plan keeps nothing new on the layer.
    PlanKeepsNothing,
    /// Its table is not one a lookup can serve through.
    Table(TableNotLookedUp),
    /// Multiplying its page for this many rows would decode the whole page.
    PageWouldDecode,
}

impl Qwen3_5Model {
    /// The layer's table when a lookup of the forward in progress can serve
    /// the layer through it; otherwise counts why not, so a layer that falls
    /// back to another way is never a silent one.
    pub(super) fn table_a_lookup_can_serve(
        &self,
        layer_index: usize,
    ) -> Result<Option<LookupTable>, TextExecutionError> {
        if !self.keeps_reply_experts_on(layer_index) {
            self.count_layer_not_looked_up(LayerNotLookedUp::PlanKeepsNothing);
            return Ok(None);
        }
        let Some((astro_expert_cache, expert_pager)) = self.expert_tables_and_pager() else {
            return Ok(None);
        };
        let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
        let partly_held_table = astro_expert_cache
            .borrow()
            .partly_held_table_or_why_not(layer_index, expert_capacity);
        match partly_held_table {
            Ok(table) => Ok(Some(LookupTable {
                table,
                expert_capacity,
            })),
            Err(why_not) => {
                self.count_layer_not_looked_up(LayerNotLookedUp::Table(why_not));
                Ok(None)
            }
        }
    }

    pub(super) fn count_layer_not_looked_up(&self, why_not: LayerNotLookedUp) {
        let in_graph_serving = self.in_graph_table_serving();
        let mut counts = in_graph_serving.counts.get();
        let count = match why_not {
            LayerNotLookedUp::PlanKeepsNothing => &mut counts.layers_the_plan_keeps_nothing_on,
            LayerNotLookedUp::PageWouldDecode => &mut counts.layers_whose_page_would_decode,
            LayerNotLookedUp::Table(TableNotLookedUp::NoTable) => {
                &mut counts.layers_without_a_table
            }
            LayerNotLookedUp::Table(TableNotLookedUp::HoldsNothing) => {
                &mut counts.layers_whose_table_holds_nothing
            }
            LayerNotLookedUp::Table(TableNotLookedUp::HoldsWholeLayer) => {
                &mut counts.layers_held_whole
            }
            LayerNotLookedUp::Table(TableNotLookedUp::CannotBeFilledFromStorage) => {
                &mut counts.layers_the_reader_cannot_fill
            }
        };
        *count += 1;
        in_graph_serving.counts.set(counts);
    }

    pub(super) fn count_table_growth(
        &self,
        layer_index: usize,
        growth: Result<bool, runtime_integration::MlxRuntimeError>,
    ) {
        match growth {
            Ok(true) => {
                let in_graph_serving = self.in_graph_table_serving();
                let mut counts = in_graph_serving.counts.get();
                counts.tables_grown_ahead_count += 1;
                in_graph_serving.counts.set(counts);
            }
            Ok(false) => {}
            Err(growth_error) => tracing::warn!(
                layer_index,
                %growth_error,
                "a table could not be grown ahead of its lookup"
            ),
        }
    }
}
