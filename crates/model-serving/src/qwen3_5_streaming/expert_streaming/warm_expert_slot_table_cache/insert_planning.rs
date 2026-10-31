//! Slot-insert planning for the warm-table cache: the free functions the
//! routed-expert insert sequences. They take the table (or the cache, for
//! the immutable pre-borrow snapshot) and return plans, counts, and grown
//! payloads; every cache-wide counter write stays with the caller that owns
//! the mutable table borrow.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::decode_cache_policy::{
    DecodeExpertSlotState, MAXIMUM_CANDIDATE_SLOT_REPLACEMENTS_PER_DECODE_STEP,
    available_probationary_slot_count,
};
use super::demand_history::HISTORICAL_HOT_DEMAND_COUNT;
use super::slot_writes::{
    create_warm_table_weights, materialize_and_detach_table_weights, write_expert_into_slot,
};
use super::{ExpertSlotTable, WarmExpertSlotTableCache};
use crate::expert_paging::ExpertWeightPage;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;

/// Snapshots the eviction-planning inputs from the cache before the table
/// borrow: the layer's prefetched-but-unconsumed experts and the incoming
/// set's historically hot members.
pub(super) fn snapshot_insert_planning_inputs(
    cache: &WarmExpertSlotTableCache,
    layer_index: usize,
    expert_ids: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let layer_prefetched_unconsumed_expert_ids: Vec<usize> = cache
        .prefetched_unconsumed_experts
        .iter()
        .filter(|(prefetched_layer_index, _)| *prefetched_layer_index == layer_index)
        .map(|(_, prefetched_expert_id)| *prefetched_expert_id)
        .collect();
    let historically_hot_expert_ids: Vec<usize> = expert_ids
        .iter()
        .copied()
        .filter(|expert_id| {
            cache.historical_demand_count(layer_index, *expert_id) >= HISTORICAL_HOT_DEMAND_COUNT
        })
        .collect();
    (
        layer_prefetched_unconsumed_expert_ids,
        historically_hot_expert_ids,
    )
}

/// Plans the incoming expert rows that are not already contained in the
/// table, truncated to the probationary budget when a decode epoch governs
/// the insert.
pub(super) fn plan_new_expert_rows(
    table: &ExpertSlotTable,
    expert_ids: &[usize],
    candidate_decode_step_epoch: Option<u64>,
) -> Vec<(usize, usize)> {
    let mut new_expert_rows: Vec<(usize, usize)> = expert_ids
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, expert_id)| !table.slot_by_expert_id.contains_key(expert_id))
        .collect();
    if let Some(decode_step_epoch) = candidate_decode_step_epoch {
        let available_probationary_slot_count =
            available_probationary_slot_count(&table.decode_policy_by_slot, decode_step_epoch);
        new_expert_rows.truncate(available_probationary_slot_count);
    }
    new_expert_rows
}

/// Grows the table within the cache budget to cover the free-slot deficit.
/// The returned free-slot list is the caller's own list when growth was
/// refused, and a recomputed list when the table tensor grew; the grown
/// payload carries back to the caller so the cache-wide accounting moves
/// with the tensor even if slot planning below gives up.
#[allow(clippy::too_many_arguments)]
pub(super) fn grow_free_slots_within_budget(
    runtime: &MlxRuntime,
    table: &mut ExpertSlotTable,
    resident_payload_bytes: u64,
    cache_ceiling_bytes: u64,
    warm_slot_count: usize,
    free_slot_deficit: usize,
    free_slots: Vec<usize>,
) -> Result<(Vec<usize>, Option<u64>), MlxRuntimeError> {
    let growth_outcome = grow_table_within_budget(
        runtime,
        table,
        resident_payload_bytes,
        cache_ceiling_bytes,
        warm_slot_count,
        free_slot_deficit,
    )?;
    if let Some(grown_resident_payload_bytes) = growth_outcome {
        let grown_free_slots: Vec<usize> = (0..table.slot_count())
            .filter(|slot| table.expert_id_by_slot[*slot].is_none())
            .collect();
        return Ok((grown_free_slots, Some(grown_resident_payload_bytes)));
    }
    Ok((free_slots, None))
}

/// Plans the eviction candidates when free slots alone cannot seat the
/// incoming set: routed and protected experts are never victims,
/// prefetched-but-unconsumed experts are never victims, and a decode epoch
/// bounds and prioritizes replacements per step.
pub(super) fn plan_evictable_slots(
    table: &ExpertSlotTable,
    expert_ids: &[usize],
    protected_expert_ids: &[usize],
    layer_prefetched_unconsumed_expert_ids: &[usize],
    candidate_decode_step_epoch: Option<u64>,
    free_slot_count: usize,
    new_expert_count: usize,
) -> Vec<usize> {
    if free_slot_count >= new_expert_count {
        return Vec::new();
    }
    let mut protected_expert_ids = protected_expert_ids.to_vec();
    protected_expert_ids.sort_unstable();
    protected_expert_ids.dedup();
    let mut evictable_slots: Vec<usize> = (0..table.slot_count())
        .filter(|slot| {
            table.expert_id_by_slot[*slot].is_some_and(|retained_expert_id| {
                !expert_ids.contains(&retained_expert_id)
                    && protected_expert_ids
                        .binary_search(&retained_expert_id)
                        .is_err()
                    && !layer_prefetched_unconsumed_expert_ids.contains(&retained_expert_id)
                    && candidate_decode_step_epoch.is_none_or(|decode_step_epoch| {
                        table.decode_policy_by_slot[*slot].is_evictable(decode_step_epoch)
                    })
            })
        })
        .collect();
    if let Some(decode_step_epoch) = candidate_decode_step_epoch {
        // Bound replacements per layer per decode step; beyond the cap
        // the insert refuses and the streamed page stays
        // operation-local. Unbounded swap-for-admission measured 492
        // evictions in 200 tokens and a 4.4x decode source-byte blowup.
        evictable_slots.truncate(MAXIMUM_CANDIDATE_SLOT_REPLACEMENTS_PER_DECODE_STEP);
        evictable_slots.sort_unstable_by(|left_slot, right_slot| {
            table.decode_policy_by_slot[*left_slot].compare_eviction_priority(
                *left_slot,
                table.decode_policy_by_slot[*right_slot],
                *right_slot,
                decode_step_epoch,
                table.per_expert_payload_bytes,
                table.per_expert_payload_bytes,
            )
        });
    } else {
        evictable_slots.sort_unstable_by_key(|slot| {
            table
                .read_count_by_slot
                .get(*slot)
                .copied()
                .unwrap_or(u64::MAX)
        });
    }
    evictable_slots
}

/// Pure: fills the plan from free slots first, then eviction candidates, and
/// never overfills past the incoming expert count.
pub(super) fn assemble_planned_slots(
    free_slots: &[usize],
    evictable_slots: &[usize],
    new_expert_count: usize,
) -> Vec<usize> {
    let mut planned_slots: Vec<usize> = free_slots.iter().copied().take(new_expert_count).collect();
    if planned_slots.len() < new_expert_count {
        for victim_slot in evictable_slots {
            if planned_slots.len() == new_expert_count {
                break;
            }
            planned_slots.push(*victim_slot);
        }
    }
    planned_slots
}

/// Writes each incoming expert into its planned slot, tracking evictions and
/// per-slot decode policy. Returns the written count and the slot
/// replacement count for the caller's cache-wide counters.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_planned_experts(
    runtime: &MlxRuntime,
    layer_index: usize,
    table: &mut ExpertSlotTable,
    streamed_weights: &Qwen3_5PagedExpertWeights,
    new_expert_rows: Vec<(usize, usize)>,
    planned_slots: Vec<usize>,
    historically_hot_expert_ids: &[usize],
    candidate_decode_step_epoch: Option<u64>,
) -> Result<(usize, u64), MlxRuntimeError> {
    let mut planned_slot_iter = planned_slots.into_iter();
    let mut written_expert_count = 0_usize;
    let mut slot_replacement_count = 0_u64;
    for (expert_row, expert_id) in new_expert_rows {
        let slot = planned_slot_iter
            .next()
            .expect("planned slots match the counted new expert set");
        let evicted = table.expert_id_by_slot[slot].take();
        if let Some(evicted_id) = evicted {
            table.slot_by_expert_id.remove(&evicted_id);
            slot_replacement_count = slot_replacement_count.saturating_add(1);
            // The victim's slot is refilled by this same iteration, so the
            // occupied payload stays net-constant: decrementing here leaked
            // one expert of payload per evict-refill cycle and drifted the
            // table's accounting away from its slot map (issue #955).
        } else {
            table.occupied_slot_count += 1;
            // Filling a free slot: the zero padding is overwritten by real expert
            // data, so we count the per-expert payload.
            table.payload_bytes = table
                .payload_bytes
                .saturating_add(table.per_expert_payload_bytes);
        }
        let write_started_at = std::time::Instant::now();
        write_expert_into_slot(
            runtime,
            &mut table.weights,
            streamed_weights,
            expert_row,
            slot,
        )?;
        let write_elapsed = write_started_at.elapsed();
        if write_elapsed > std::time::Duration::from_millis(5) {
            tracing::info!(
                layer_index,
                expert_id,
                slot,
                write_elapsed_millis = write_elapsed.as_millis(),
                "slow write_expert_into_slot"
            );
        }
        table.expert_id_by_slot[slot] = Some(expert_id);
        table.slot_by_expert_id.insert(expert_id, slot);
        table.read_count_by_slot[slot] = 0;
        if let Some(decode_step_epoch) = candidate_decode_step_epoch {
            if historically_hot_expert_ids.contains(&expert_id) {
                table.decode_policy_by_slot[slot].mark_historically_hot(decode_step_epoch);
            } else {
                table.decode_policy_by_slot[slot].mark_probationary(decode_step_epoch);
            }
        } else {
            table.decode_policy_by_slot[slot] = DecodeExpertSlotState::default();
        }
        written_expert_count += 1;
    }
    Ok((written_expert_count, slot_replacement_count))
}

pub(super) fn grow_table_within_budget(
    runtime: &MlxRuntime,
    table: &mut ExpertSlotTable,
    resident_payload_bytes: u64,
    cache_ceiling_bytes: u64,
    warm_slot_count: usize,
    free_slot_deficit: usize,
) -> Result<Option<u64>, MlxRuntimeError> {
    if warm_slot_count == 0 || table.slot_count() >= warm_slot_count {
        return Ok(None);
    }
    let growth_target_slot_count = (table
        .slot_count()
        .saturating_mul(2)
        .max(table.slot_count() + free_slot_deficit))
    .min(warm_slot_count);
    // Budget gate before allocation: the per-expert estimate refuses a growth
    // the ceiling cannot hold without allocating the tensor first.
    let estimated_growth_payload_bytes = table
        .per_expert_payload_bytes
        .saturating_mul(u64::try_from(growth_target_slot_count - table.slot_count()).unwrap_or(0));
    if estimated_growth_payload_bytes > 0
        && resident_payload_bytes.saturating_add(estimated_growth_payload_bytes)
            > cache_ceiling_bytes
    {
        return Ok(None);
    }
    let previous_slot_count = table.slot_count();
    let previous_full_padded_payload_bytes = table.full_padded_payload_bytes;
    let mut grown_weights =
        create_warm_table_weights(runtime, &table.weights, growth_target_slot_count)?;
    for previous_slot in 0..previous_slot_count {
        if table.expert_id_by_slot[previous_slot].is_some() {
            write_expert_into_slot(
                runtime,
                &mut grown_weights,
                &table.weights,
                previous_slot,
                previous_slot,
            )?;
        }
    }
    // Until the copies evaluate, the retired tensor stays alive in active
    // memory while the cache claims only its replacement, and the lazy graph
    // would pin it for the grown table's lifetime. Growth is rare, so paying
    // one eager evaluation and a detach here keeps both the buffers and the
    // published decomposition exact (issue #955).
    materialize_and_detach_table_weights(runtime, &grown_weights)?;
    let grown_full_padded_payload_bytes = grown_weights.resident_payload_byte_count();
    let measured_growth_payload_bytes =
        grown_full_padded_payload_bytes.saturating_sub(previous_full_padded_payload_bytes);
    let projected_resident_payload_bytes =
        resident_payload_bytes.saturating_add(measured_growth_payload_bytes);
    if measured_growth_payload_bytes > estimated_growth_payload_bytes
        && projected_resident_payload_bytes > cache_ceiling_bytes
    {
        // The measured growth exceeded the pre-allocation estimate and no
        // longer fits. The table itself already grew, so keep the honest
        // accounting and let the ceiling's own reclamation shed on demand.
        tracing::warn!(
            measured_growth_payload_bytes,
            estimated_growth_payload_bytes,
            projected_resident_payload_bytes,
            cache_ceiling_bytes,
            "warm-table growth measured above the budget estimate"
        );
    }
    table.weights = grown_weights;
    table
        .expert_id_by_slot
        .resize(growth_target_slot_count, None);
    table.read_count_by_slot.resize(growth_target_slot_count, 0);
    table
        .decode_policy_by_slot
        .resize(growth_target_slot_count, DecodeExpertSlotState::default());
    table.full_padded_payload_bytes = grown_full_padded_payload_bytes;
    tracing::debug!(
        previous_slot_count,
        growth_target_slot_count,
        measured_growth_payload_bytes,
        "grew a decode warm table for demonstrated demand"
    );
    Ok(Some(projected_resident_payload_bytes))
}

#[cfg(test)]
#[path = "insert_planning_tests.rs"]
mod insert_planning_tests;
