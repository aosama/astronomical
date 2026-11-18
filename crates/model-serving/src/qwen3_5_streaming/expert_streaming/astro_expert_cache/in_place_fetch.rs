//! Fetches a reply token's missing experts straight into their table slots.
//!
//! The other way in is `insert` and `flush`: the token multiplies from a page
//! MLX loaded, and after the forward its rows are copied into the table.
//! Here the same slots are planned by the same rules, before the multiply,
//! and the experts are read from storage directly into them; the token then
//! multiplies from the table like any expert already held. Which experts are
//! kept and which give way is decided exactly as before; only how the bytes
//! arrive changes.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::AstroExpertCache;
use super::insert_planning::{
    PlannedRowSource, PlannedSlotInsert, snapshot_insert_planning_inputs, write_planned_experts,
};
use super::slot_writes::materialize_and_detach_table_weights;
use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::qwen3_5_streaming::expert_streaming::ExpertSlotStorageFiller;

impl AstroExpertCache {
    /// Puts every expert of `missing_expert_ids` into the layer's table by
    /// reading it from storage into a slot planned as an insert would plan
    /// it, never taking a slot from `protected_expert_ids` (the experts this
    /// token found in the table). Returns `true` when all of them are now in
    /// the table.
    ///
    /// Returns `false`, having changed nothing an insert would not, when the
    /// layer has no table yet, its experts are stored in an encoding the
    /// filler does not write, or the table cannot take them all; the caller
    /// then serves the token the copying way.
    ///
    /// The caller must have evaluated every graph that reads this layer's
    /// table: a slot given to a new expert is overwritten in place.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fetch_missing_experts_into_table(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        layer_plan: &QuantizedExpertLayerPlan,
        missing_expert_ids: &[usize],
        protected_expert_ids: &[usize],
        warm_slot_count: usize,
        slot_filler: &ExpertSlotStorageFiller,
    ) -> Result<bool, MlxRuntimeError> {
        let Some(table) = self
            .tables_by_layer
            .get(layer_index)
            .and_then(|table| table.as_ref())
        else {
            return Ok(false);
        };
        if missing_expert_ids.is_empty() || !ExpertSlotStorageFiller::can_fill(&table.weights) {
            return Ok(false);
        }
        let candidate_decode_step_epoch = (self.decode_astro_expert_cache_policy
            == astronomical_ipc_protocol::WorkerDecodeAstroExpertCachePolicy::Candidate)
            .then_some(self.decode_step_epoch);
        let (layer_prefetched_unconsumed_expert_ids, historically_hot_expert_ids) =
            snapshot_insert_planning_inputs(self, layer_index, missing_expert_ids);
        let Some(PlannedSlotInsert {
            new_expert_rows,
            planned_slots,
        }) = self.plan_slots_in_existing_table(
            runtime,
            layer_index,
            missing_expert_ids,
            protected_expert_ids,
            &layer_prefetched_unconsumed_expert_ids,
            warm_slot_count,
            candidate_decode_step_epoch,
        )?
        else {
            return Ok(false);
        };
        if new_expert_rows.len() != missing_expert_ids.len() {
            // Planning skips experts it will not admit; a token served from
            // the table needs every one of them there.
            return Ok(false);
        }
        let table = self
            .tables_by_layer
            .get_mut(layer_index)
            .and_then(|table| table.as_mut())
            .expect("slot table was just planned");
        let (claimed_expert_slots, slot_replacement_count) = write_planned_experts(
            runtime,
            layer_index,
            table,
            PlannedRowSource::Storage,
            new_expert_rows,
            planned_slots,
            &historically_hot_expert_ids,
            candidate_decode_step_epoch,
        )?;
        // Growth leaves the table as a lazy copy of the smaller one; its
        // storage exists only once that copy has run.
        materialize_and_detach_table_weights(runtime, &table.weights)?;
        if let Err(fill_error) =
            slot_filler.fill_slots(layer_plan, &mut table.weights, &claimed_expert_slots)
        {
            // The slots hold whatever the failed read left: give them up so
            // no token multiplies from them.
            for (expert_id, slot) in claimed_expert_slots {
                table.slot_by_expert_id.remove(&expert_id);
                table.expert_id_by_slot[slot] = None;
                table.occupied_slot_count = table.occupied_slot_count.saturating_sub(1);
                table.payload_bytes = table
                    .payload_bytes
                    .saturating_sub(table.per_expert_payload_bytes);
            }
            return Err(fill_error);
        }
        self.record_planned_insert_counters(
            claimed_expert_slots.len(),
            slot_replacement_count,
            warm_slot_count,
            candidate_decode_step_epoch,
        );
        Ok(true)
    }
}
