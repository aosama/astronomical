//! Warm-table creation, growth, and routed-expert slot inserts for the
//! retained cache.
//!
//! A decode miss streams one token's routed experts, and this module decides
//! whether that page becomes a retained warm table. The first insert adopts
//! the streamed page as-is (no padding); when the table fills, it doubles in
//! place while the budget admits the growth, so resident bytes always track
//! demonstrated demand within a factor of two. Eagerly padding a new table to
//! the whole affordable capacity charged the budget for zeros and starved
//! sibling layers' warming after the first token (issue #955). Budget
//! admission is the only refusal reason: a page the machine cannot hold stays
//! operation-local.
//!
//! Every write path ends in
//! [`materialize_and_detach_table_weights`][super::slot_writes::materialize_and_detach_table_weights]:
//! MLX retains an array's graph inputs for its lifetime, so a retained table
//! built from lazy row-writes would pin each source buffer — the streamed
//! page of a fill, the retired tensor of a growth — in active memory until
//! the table dies. Evaluating and detaching keeps the published memory
//! decomposition exact.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::decode_cache_policy::{
    DecodeExpertSlotState, HISTORICAL_HOT_DEMAND_COUNT, MAXIMUM_PROBATIONARY_EXPERT_SLOTS_PER_LAYER,
};
use super::insert_planning::{
    assemble_planned_slots, grow_free_slots_within_budget, plan_evictable_slots,
    plan_new_expert_rows, snapshot_insert_planning_inputs, write_planned_experts,
};
use super::slot_writes::{
    RetainedReferenceOk, create_warm_table_weights, materialize_and_detach_table_weights,
    write_expert_into_slot,
};
use super::{ExpertSlotTable, WarmExpertSlotTableCache};
use crate::expert_paging::ExpertWeightPage;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;

impl WarmExpertSlotTableCache {
    /// Inserts streamed experts into the layer's slot table. The first insert
    /// adopts the streamed page as-is so the table holds exactly the routed
    /// experts; later inserts grow the table (doubling, bounded by the policy
    /// warm capacity) or write into free and least-read slots. Returns how
    /// many experts were newly written.
    pub fn insert_streamed_experts(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        protected_expert_ids: &[usize],
        warm_slot_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        self.insert_streamed_experts_internal(
            runtime,
            layer_index,
            expert_ids,
            streamed_weights,
            protected_expert_ids,
            warm_slot_count,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn insert_candidate_decode_experts(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        protected_expert_ids: &[usize],
        warm_slot_count: usize,
        decode_step_epoch: u64,
    ) -> Result<usize, MlxRuntimeError> {
        self.insert_streamed_experts_internal(
            runtime,
            layer_index,
            expert_ids,
            streamed_weights,
            protected_expert_ids,
            warm_slot_count,
            Some(decode_step_epoch),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_streamed_experts_internal(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        protected_expert_ids: &[usize],
        warm_slot_count: usize,
        candidate_decode_step_epoch: Option<u64>,
    ) -> Result<usize, MlxRuntimeError> {
        if expert_ids.is_empty() {
            return Ok(0);
        }
        // Snapshot before the mutable table borrow below: victims must exclude
        // prefetched-but-unconsumed experts (colibri's PILOT_EVICT_GUARD
        // default) — evicting a page the next step is expected to consume
        // converts a hit into a re-read. Persisted-usage hotness is snapshotted
        // for the same reason: the incoming set's history decides probation.
        let (layer_prefetched_unconsumed_expert_ids, historically_hot_expert_ids) =
            snapshot_insert_planning_inputs(self, layer_index, expert_ids);
        if self
            .tables_by_layer
            .get(layer_index)
            .is_none_or(|table| table.is_none())
        {
            if let Some(decode_step_epoch) = candidate_decode_step_epoch {
                return self.insert_first_candidate_decode_table(
                    runtime,
                    layer_index,
                    expert_ids,
                    streamed_weights,
                    warm_slot_count,
                    decode_step_epoch,
                );
            }
            return self.adopt_streamed_page_as_first_table(
                runtime,
                layer_index,
                expert_ids,
                streamed_weights,
                warm_slot_count,
            );
        }
        let cache_ceiling_bytes = self.effective_maximum_resident_payload_bytes();
        let table = self
            .tables_by_layer
            .get_mut(layer_index)
            .and_then(|table| table.as_mut())
            .expect("slot table was just checked");
        let mut resident_payload_bytes = self.resident_payload_bytes;
        // Plan every slot before writing: free slots first, then growth, then
        // least-read victims. Planning up front keeps experts inserted by the
        // same flush from evicting each other — with equal (zero) read counts a
        // naive pick-lowest-slot loop would let the second insert evict the
        // first. Victims also exclude slots holding any incoming routed expert:
        // such an expert is contained now, and an eviction would demote it to a
        // late surprise miss that exhausts the plan.
        let new_expert_rows = plan_new_expert_rows(table, expert_ids, candidate_decode_step_epoch);
        let new_expert_count = new_expert_rows.len();
        if new_expert_count == 0 {
            return Ok(0);
        }
        let mut free_slots: Vec<usize> = (0..table.slot_count())
            .filter(|slot| table.expert_id_by_slot[*slot].is_none())
            .collect();
        let free_slot_deficit = new_expert_count.saturating_sub(free_slots.len());
        if free_slot_deficit > 0 {
            let (grown_free_slots, grown_resident_payload_bytes) = grow_free_slots_within_budget(
                runtime,
                table,
                resident_payload_bytes,
                cache_ceiling_bytes,
                warm_slot_count,
                free_slot_deficit,
                free_slots,
            )?;
            free_slots = grown_free_slots;
            if let Some(grown_resident_payload_bytes) = grown_resident_payload_bytes {
                resident_payload_bytes = grown_resident_payload_bytes;
                // The table tensor already grew, so the cache-wide accounting
                // must move with it even if slot planning below gives up:
                // undercounting here would let later can_admit checks overfill
                // the ceiling.
                self.resident_payload_bytes = resident_payload_bytes;
            }
        }
        let evictable_slots = plan_evictable_slots(
            table,
            expert_ids,
            protected_expert_ids,
            &layer_prefetched_unconsumed_expert_ids,
            candidate_decode_step_epoch,
            free_slots.len(),
            new_expert_count,
        );
        if new_expert_count > free_slots.len() + evictable_slots.len() {
            return Ok(0);
        }
        let planned_slots = assemble_planned_slots(&free_slots, &evictable_slots, new_expert_count);
        let (written_expert_count, slot_replacement_count) = write_planned_experts(
            runtime,
            layer_index,
            table,
            streamed_weights,
            new_expert_rows,
            planned_slots,
            &historically_hot_expert_ids,
            candidate_decode_step_epoch,
        )?;
        if warm_slot_count > 0 {
            self.warm_expert_insert_count = self
                .warm_expert_insert_count
                .saturating_add(u64::try_from(written_expert_count).unwrap_or(u64::MAX));
        }
        self.slot_replacement_count = self
            .slot_replacement_count
            .saturating_add(slot_replacement_count);
        if candidate_decode_step_epoch.is_some() {
            self.decode_slot_eviction_count = self
                .decode_slot_eviction_count
                .saturating_add(slot_replacement_count);
        }
        self.resident_payload_bytes = resident_payload_bytes;
        // The fills left the table arrays with lazy graphs that pin this
        // token's streamed page for the table's lifetime. Materialize the
        // writes and strip the provenance so the page frees with the forward.
        materialize_and_detach_table_weights(runtime, &table.weights)?;
        Ok(written_expert_count)
    }

    /// Adopts the streamed page as the layer's first table, refusing only when
    /// the machine cannot hold the page. The table starts at the routed set,
    /// and growth expands it only as demand proves itself.
    fn adopt_streamed_page_as_first_table(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        warm_slot_count: usize,
    ) -> Result<usize, MlxRuntimeError> {
        let routed_expert_count = expert_ids.len();
        let per_expert_payload_bytes = streamed_weights
            .resident_payload_byte_count()
            .checked_div(u64::try_from(routed_expert_count).unwrap_or(1))
            .unwrap_or(0);
        if !self.can_admit(streamed_weights.resident_payload_byte_count()) {
            // Leave the streamed page operation-local. Evicting another complete
            // layer to seat this one would thrash the sequential decoder order.
            return Ok(0);
        }
        // Adopt the streamed page as-is: the table starts at the routed set,
        // and growth below expands it only as demand proves itself.
        let weights = streamed_weights.retained_reference_ok();
        // Today's streamed pages are materialized file reads so this is a
        // no-op evaluation, but a future lazy streaming pipeline must not
        // pin its sources for the table's lifetime: adoption gets the same
        // graph hygiene as every write path.
        materialize_and_detach_table_weights(runtime, &weights)?;
        // The residency plan validates payload as per-expert bytes times
        // occupancy, so the accounting must use exactly that product even
        // if the streamed page's raw byte count carries extra alignment.
        let payload_bytes = per_expert_payload_bytes
            .saturating_mul(u64::try_from(routed_expert_count).unwrap_or(0));
        let full_padded_payload_bytes = weights.resident_payload_byte_count();
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_add(full_padded_payload_bytes);
        let mut slot_by_expert_id = HashMap::with_capacity(routed_expert_count);
        let mut expert_id_by_slot = Vec::with_capacity(routed_expert_count);
        for (slot, expert_id) in expert_ids.iter().enumerate() {
            expert_id_by_slot.push(Some(*expert_id));
            slot_by_expert_id.insert(*expert_id, slot);
        }
        let table = ExpertSlotTable {
            weights,
            expert_id_by_slot,
            slot_by_expert_id,
            read_count_by_slot: vec![0; routed_expert_count],
            decode_policy_by_slot: vec![DecodeExpertSlotState::default(); routed_expert_count],
            occupied_slot_count: routed_expert_count,
            per_expert_payload_bytes,
            payload_bytes,
            full_padded_payload_bytes,
        };
        if let Some(slot) = self.tables_by_layer.get_mut(layer_index) {
            *slot = Some(table);
        }
        // Warm-insert evidence counts only hot-expert warming (a nonzero
        // warm capacity); complete-layer adoption is whole-layer caching.
        if warm_slot_count > 0 {
            self.warm_expert_insert_count = self
                .warm_expert_insert_count
                .saturating_add(u64::try_from(routed_expert_count).unwrap_or(u64::MAX));
        }
        Ok(routed_expert_count)
    }

    fn insert_first_candidate_decode_table(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        expert_ids: &[usize],
        streamed_weights: &Qwen3_5PagedExpertWeights,
        warm_slot_count: usize,
        decode_step_epoch: u64,
    ) -> Result<usize, MlxRuntimeError> {
        let admitted_expert_count = expert_ids
            .len()
            .min(warm_slot_count)
            .min(MAXIMUM_PROBATIONARY_EXPERT_SLOTS_PER_LAYER);
        if admitted_expert_count == 0 {
            return Ok(0);
        }
        let per_expert_payload_bytes = streamed_weights
            .resident_payload_byte_count()
            .checked_div(u64::try_from(expert_ids.len()).unwrap_or(1))
            .unwrap_or(0);
        let estimated_table_payload_bytes = per_expert_payload_bytes
            .saturating_mul(u64::try_from(admitted_expert_count).unwrap_or(u64::MAX));
        if !self.can_admit(estimated_table_payload_bytes) {
            return Ok(0);
        }

        let mut table_weights =
            create_warm_table_weights(runtime, streamed_weights, admitted_expert_count)?;
        for expert_row in 0..admitted_expert_count {
            write_expert_into_slot(
                runtime,
                &mut table_weights,
                streamed_weights,
                expert_row,
                expert_row,
            )?;
        }
        materialize_and_detach_table_weights(runtime, &table_weights)?;
        let full_padded_payload_bytes = table_weights.resident_payload_byte_count();
        if !self.can_admit(full_padded_payload_bytes) {
            return Ok(0);
        }

        let occupied_payload_bytes = per_expert_payload_bytes
            .saturating_mul(u64::try_from(admitted_expert_count).unwrap_or(u64::MAX));
        let expert_id_by_slot: Vec<Option<usize>> = expert_ids
            .iter()
            .take(admitted_expert_count)
            .copied()
            .map(Some)
            .collect();
        let slot_by_expert_id = expert_id_by_slot
            .iter()
            .enumerate()
            .filter_map(|(slot, expert_id)| expert_id.map(|expert_id| (expert_id, slot)))
            .collect();
        let mut decode_policy_by_slot =
            vec![DecodeExpertSlotState::default(); admitted_expert_count];
        for (decode_policy_slot, admitted_expert_id) in expert_ids
            .iter()
            .take(admitted_expert_count)
            .copied()
            .enumerate()
        {
            if self.historical_demand_count(layer_index, admitted_expert_id)
                >= HISTORICAL_HOT_DEMAND_COUNT
            {
                decode_policy_by_slot[decode_policy_slot].mark_historically_hot(decode_step_epoch);
            } else {
                decode_policy_by_slot[decode_policy_slot].mark_probationary(decode_step_epoch);
            }
        }
        let table = ExpertSlotTable {
            weights: table_weights,
            expert_id_by_slot,
            slot_by_expert_id,
            read_count_by_slot: vec![0; admitted_expert_count],
            decode_policy_by_slot,
            occupied_slot_count: admitted_expert_count,
            per_expert_payload_bytes,
            payload_bytes: occupied_payload_bytes,
            full_padded_payload_bytes,
        };
        if let Some(table_slot) = self.tables_by_layer.get_mut(layer_index) {
            *table_slot = Some(table);
        } else {
            return Ok(0);
        }
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_add(full_padded_payload_bytes);
        if warm_slot_count > 0 {
            self.warm_expert_insert_count = self
                .warm_expert_insert_count
                .saturating_add(u64::try_from(admitted_expert_count).unwrap_or(u64::MAX));
        }
        Ok(admitted_expert_count)
    }
}
