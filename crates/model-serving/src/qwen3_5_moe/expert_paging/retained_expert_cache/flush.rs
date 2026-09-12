//! Post-evaluation drain of queued slot inserts.
//!
//! Inserts are queued during a forward and drained after its arrays evaluate,
//! so `slice_update` can donate the destination buffer instead of copying a live
//! gather buffer. Two kinds of insert share this drain: ordinary hot-expert
//! warming, and the leftover-only previous-token prefetch from issue #537. The
//! prefetch kind is scored separately because it is an experiment, not warming.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::RetainedExpertCache;
use crate::expert_paging::ExpertWeightPage;

impl RetainedExpertCache {
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
                let missing_before_insert: Vec<usize> = pending_insert
                    .expert_ids
                    .iter()
                    .copied()
                    .filter(|expert_id| {
                        !self
                            .tables_by_layer
                            .get(pending_insert.layer_index)
                            .and_then(|table| table.as_ref())
                            .is_some_and(|table| table.slot_by_expert_id.contains_key(expert_id))
                    })
                    .collect();
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
                let expert_id_count = u64::try_from(pending_insert.expert_ids.len())
                    .unwrap_or(1)
                    .max(1);
                let per_expert_payload_bytes = pending_insert
                    .weights
                    .resident_payload_byte_count()
                    .saturating_div(expert_id_count);
                self.prefetch_payload_bytes = self.prefetch_payload_bytes.saturating_add(
                    per_expert_payload_bytes
                        .saturating_mul(u64::try_from(written_count).unwrap_or(u64::MAX)),
                );
                if written_count > 0 {
                    self.mark_prefetched_experts(
                        pending_insert.layer_index,
                        &missing_before_insert,
                    );
                }
            } else {
                let written_count = self.insert_streamed_experts(
                    runtime,
                    pending_insert.layer_index,
                    &pending_insert.expert_ids,
                    &pending_insert.weights,
                    &[],
                    pending_insert.warm_slot_count,
                )?;
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
}
