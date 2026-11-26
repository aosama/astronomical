//! Post-evaluation drain of queued slot inserts.
//!
//! Inserts are queued during a forward and drained after its arrays evaluate,
//! so `slice_update` can donate the destination buffer instead of copying a live
//! gather buffer. Two kinds of insert share this drain: ordinary hot-expert
//! warming, and the leftover-only previous-token prefetch from issue #537. The
//! prefetch kind is scored separately because it is an experiment, not warming.

use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::{AstroExpertCache, PendingSlotInsert};
use engine_core::expert_paging::ExpertWeightPage;

impl AstroExpertCache {
    /// Writes queued miss experts into their layer tables after GPU evaluation
    /// and returns how many experts were newly written.
    pub fn flush_pending_inserts(&mut self, runtime: &MlxRuntime) -> Result<u64, MlxRuntimeError> {
        let flush_started_at = std::time::Instant::now();
        let pending_inserts = std::mem::take(&mut self.pending_inserts);
        let pending_count = pending_inserts.len();
        let total_expert_count: usize = pending_inserts.iter().map(|pi| pi.expert_ids.len()).sum();
        let mut written_expert_count = 0_u64;
        for pending_insert in pending_inserts {
            let insert_started_at = std::time::Instant::now();
            let expert_count = pending_insert.expert_ids.len();
            if pending_insert.no_evict {
                self.flush_leftover_only_prefetch(runtime, &pending_insert)?;
            } else {
                let written_count = self.flush_ordinary_insert(runtime, &pending_insert)?;
                written_expert_count = written_expert_count
                    .saturating_add(u64::try_from(written_count).unwrap_or(u64::MAX));
            }
            let insert_elapsed = insert_started_at.elapsed();
            if insert_elapsed > std::time::Duration::from_millis(10) {
                tracing::info!(
                    layer_index = pending_insert.layer_index,
                    expert_count,
                    insert_elapsed_millis = insert_elapsed.as_millis(),
                    "slow slot table insert after flush"
                );
            }
        }
        let flush_elapsed = flush_started_at.elapsed();
        if flush_elapsed > std::time::Duration::from_millis(100) {
            tracing::info!(
                pending_count,
                total_expert_count,
                flush_elapsed_millis = flush_elapsed.as_millis(),
                "flushed pending expert slot inserts"
            );
        }
        Ok(written_expert_count)
    }

    /// Issue #537: parks the previous token's experts in leftover slots only,
    /// then scores the attempt. Experts written count as issued and experts
    /// that found no free slot count as capacity drops. When anything was
    /// written, the experts that were missing beforehand are marked prefetched
    /// until a later route demands them.
    fn flush_leftover_only_prefetch(
        &mut self,
        runtime: &MlxRuntime,
        pending_insert: &PendingSlotInsert,
    ) -> Result<(), MlxRuntimeError> {
        let missing_before_insert = self
            .expert_ids_missing_from_table(pending_insert.layer_index, &pending_insert.expert_ids);
        let written_count = self.insert_into_free_slots_only(
            runtime,
            pending_insert.layer_index,
            &pending_insert.expert_ids,
            &pending_insert.weights,
            pending_insert.warm_slot_count,
        )?;
        let drop_count = missing_before_insert.len().saturating_sub(written_count);
        self.prefetch_issue_count = self
            .prefetch_issue_count
            .saturating_add(u64::try_from(written_count).unwrap_or(u64::MAX));
        self.prefetch_capacity_drop_count = self
            .prefetch_capacity_drop_count
            .saturating_add(u64::try_from(drop_count).unwrap_or(u64::MAX));
        let parked_payload_bytes = prefetched_payload_bytes(
            pending_insert.weights.resident_payload_byte_count(),
            pending_insert.expert_ids.len(),
            written_count,
        );
        self.prefetch_payload_bytes = self
            .prefetch_payload_bytes
            .saturating_add(parked_payload_bytes);
        if written_count > 0 {
            self.mark_prefetched_experts(pending_insert.layer_index, &missing_before_insert);
        }
        Ok(())
    }

    /// Inserts one queued miss page the ordinary way and returns how many
    /// experts were newly written. A decode-demand page goes through the
    /// candidate policy when that policy is selected.
    fn flush_ordinary_insert(
        &mut self,
        runtime: &MlxRuntime,
        pending_insert: &PendingSlotInsert,
    ) -> Result<usize, MlxRuntimeError> {
        if let Some(decode_step_epoch) =
            self.candidate_decode_step_epoch(pending_insert.is_decode_demand)
        {
            return self.insert_candidate_decode_experts(
                runtime,
                pending_insert.layer_index,
                &pending_insert.expert_ids,
                &pending_insert.weights,
                &[],
                pending_insert.warm_slot_count,
                decode_step_epoch,
            );
        }
        self.insert_streamed_experts(
            runtime,
            pending_insert.layer_index,
            &pending_insert.expert_ids,
            &pending_insert.weights,
            &[],
            pending_insert.warm_slot_count,
        )
    }

    /// The decode step epoch to insert under when the candidate decode policy
    /// governs this page, `None` when the baseline insert applies.
    fn candidate_decode_step_epoch(&self, is_decode_demand: bool) -> Option<u64> {
        (is_decode_demand
            && self.decode_astro_expert_cache_policy
                == ipc_protocol::WorkerDecodeAstroExpertCachePolicy::Candidate)
            .then_some(self.decode_step_epoch)
    }

    /// The experts of `expert_ids` the layer's table does not hold, in the
    /// order given. A layer without a table holds none.
    fn expert_ids_missing_from_table(
        &self,
        layer_index: usize,
        expert_ids: &[usize],
    ) -> Vec<usize> {
        expert_ids
            .iter()
            .copied()
            .filter(|expert_id| {
                !self
                    .tables_by_layer
                    .get(layer_index)
                    .and_then(|table| table.as_ref())
                    .is_some_and(|table| table.slot_by_expert_id.contains_key(expert_id))
            })
            .collect()
    }
}

/// Payload bytes a leftover-slot prefetch parked: the streamed page's bytes
/// averaged over its experts, times the experts actually written. A page that
/// names no experts is averaged over one, never dividing by zero.
fn prefetched_payload_bytes(
    page_payload_bytes: u64,
    page_expert_count: usize,
    written_count: usize,
) -> u64 {
    let expert_id_count = u64::try_from(page_expert_count).unwrap_or(1).max(1);
    page_payload_bytes
        .saturating_div(expert_id_count)
        .saturating_mul(u64::try_from(written_count).unwrap_or(u64::MAX))
}

#[cfg(test)]
#[path = "flush_tests.rs"]
mod flush_tests;
