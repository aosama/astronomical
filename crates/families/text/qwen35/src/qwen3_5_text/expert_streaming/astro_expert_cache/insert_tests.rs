//! Beside-source coverage for the pure pieces of the first candidate-policy
//! table: how admitted experts are seated one per slot, and which decode
//! policy state each starts in. The table writes run under the serial engine
//! lane against real arrays.

use super::{
    AstroExpertCache, DecodeExpertSlotState, HISTORICAL_HOT_DEMAND_COUNT, seat_experts_one_per_slot,
};
use crate::qwen3_5_text::expert_streaming::PersistedExpertUsage;

const DECODE_STEP_EPOCH: u64 = 10;
const EXPERT_CAPACITY: usize = 8;

/// A one-layer `astro_expert_cache` whose persisted decode record counts
/// `expert_id` `demand_count` times, as a previous session would have left it.
fn cache_with_historical_demand(expert_id: usize, demand_count: u64) -> AstroExpertCache {
    let mut astro_expert_cache = AstroExpertCache::new(1);
    let mut decode_demand_counts = vec![0_u64; EXPERT_CAPACITY];
    decode_demand_counts[expert_id] = demand_count;
    astro_expert_cache.seed_expert_demand_history(PersistedExpertUsage {
        prefill_demand_counts_by_layer: vec![vec![0; EXPERT_CAPACITY]],
        decode_demand_counts_by_layer: vec![decode_demand_counts],
    });
    astro_expert_cache
}

#[test]
fn should_seat_each_admitted_expert_in_its_own_slot_in_order() {
    let (expert_id_by_slot, slot_by_expert_id) = seat_experts_one_per_slot(&[6, 2, 5]);

    assert_eq!(expert_id_by_slot, vec![Some(6), Some(2), Some(5)]);
    assert_eq!(slot_by_expert_id.len(), 3);
    assert_eq!(slot_by_expert_id[&6], 0);
    assert_eq!(slot_by_expert_id[&2], 1);
    assert_eq!(slot_by_expert_id[&5], 2);
}

#[test]
fn should_seat_nothing_for_an_empty_admission() {
    let (expert_id_by_slot, slot_by_expert_id) = seat_experts_one_per_slot(&[]);

    assert!(expert_id_by_slot.is_empty());
    assert!(slot_by_expert_id.is_empty());
}

#[test]
fn should_keep_the_later_slot_for_an_expert_named_twice() {
    let (expert_id_by_slot, slot_by_expert_id) = seat_experts_one_per_slot(&[4, 7, 4]);

    assert_eq!(expert_id_by_slot, vec![Some(4), Some(7), Some(4)]);
    assert_eq!(slot_by_expert_id[&4], 2);
}

#[test]
fn should_promote_an_expert_whose_history_reaches_the_hot_count() {
    let astro_expert_cache = cache_with_historical_demand(3, HISTORICAL_HOT_DEMAND_COUNT);

    let decode_policy_by_slot =
        astro_expert_cache.first_admission_decode_policy_by_slot(0, &[3], DECODE_STEP_EPOCH);

    assert!(!decode_policy_by_slot[0].is_probationary(DECODE_STEP_EPOCH));
}

#[test]
fn should_put_an_expert_just_under_the_hot_count_on_probation() {
    let astro_expert_cache = cache_with_historical_demand(3, HISTORICAL_HOT_DEMAND_COUNT - 1);

    let decode_policy_by_slot =
        astro_expert_cache.first_admission_decode_policy_by_slot(0, &[3], DECODE_STEP_EPOCH);

    assert!(decode_policy_by_slot[0].is_probationary(DECODE_STEP_EPOCH));
}

#[test]
fn should_decide_each_slot_from_its_own_experts_history() {
    let astro_expert_cache = cache_with_historical_demand(5, HISTORICAL_HOT_DEMAND_COUNT * 2);

    let decode_policy_by_slot =
        astro_expert_cache.first_admission_decode_policy_by_slot(0, &[1, 5], DECODE_STEP_EPOCH);

    assert_eq!(decode_policy_by_slot.len(), 2);
    assert!(decode_policy_by_slot[0].is_probationary(DECODE_STEP_EPOCH));
    assert!(!decode_policy_by_slot[1].is_probationary(DECODE_STEP_EPOCH));
}

#[test]
fn should_start_every_expert_on_probation_without_any_history() {
    let astro_expert_cache = AstroExpertCache::new(1);

    let decode_policy_by_slot =
        astro_expert_cache.first_admission_decode_policy_by_slot(0, &[0, 1], DECODE_STEP_EPOCH);

    assert!(
        decode_policy_by_slot
            .iter()
            .all(|decode_policy_state: &DecodeExpertSlotState| {
                decode_policy_state.is_probationary(DECODE_STEP_EPOCH)
            })
    );
}

#[test]
fn should_return_no_policy_states_for_an_empty_admission() {
    let astro_expert_cache = AstroExpertCache::new(1);

    assert!(
        astro_expert_cache
            .first_admission_decode_policy_by_slot(0, &[], DECODE_STEP_EPOCH)
            .is_empty()
    );
}
