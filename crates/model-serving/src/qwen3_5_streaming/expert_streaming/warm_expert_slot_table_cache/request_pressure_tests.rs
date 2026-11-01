//! Unit contracts for the prefill memory-recovery freeze on the retained
//! expert cache (issue #1729). Each test reproduces one way the freeze used
//! to be undone within a request: a per-chunk admission budget installed as
//! a request-pressure cap, a residency-plan refresh, and a request-pressure
//! resume after a staging loan.

use super::*;

const NORMAL_CEILING_BYTES: u64 = 1_073_741_824;
const ADMITTED_FORWARD_BUDGET_BYTES: u64 = 805_306_368;

fn effective_ceiling_bytes(cache: &WarmExpertSlotTableCache) -> u64 {
    cache.statistics().maximum_resident_payload_byte_count
}

fn frozen_cache() -> WarmExpertSlotTableCache {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.update_maximum_resident_payload_bytes(NORMAL_CEILING_BYTES);
    cache.freeze_for_prefill_memory_recovery();
    cache
}

#[test]
fn should_hold_the_recovery_freeze_when_an_admission_budget_is_installed() {
    let mut cache = frozen_cache();
    cache.limit_for_request_pressure_to_maximum(ADMITTED_FORWARD_BUDGET_BYTES);
    assert_eq!(effective_ceiling_bytes(&cache), 0);
}

#[test]
fn should_hold_the_recovery_freeze_across_a_residency_plan_refresh() {
    let mut cache = frozen_cache();
    cache.update_maximum_resident_payload_bytes(NORMAL_CEILING_BYTES);
    assert_eq!(effective_ceiling_bytes(&cache), 0);
}

#[test]
fn should_hold_the_recovery_freeze_when_request_pressure_is_resumed() {
    let mut cache = frozen_cache();
    cache.limit_for_request_pressure_to_maximum(ADMITTED_FORWARD_BUDGET_BYTES);
    cache.resume_after_request_pressure();
    assert_eq!(effective_ceiling_bytes(&cache), 0);
}

#[test]
fn should_restore_the_normal_ceiling_when_the_recovery_freeze_is_lifted() {
    let mut cache = frozen_cache();
    assert!(cache.lift_prefill_memory_recovery_freeze());
    assert_eq!(effective_ceiling_bytes(&cache), NORMAL_CEILING_BYTES);
}

#[test]
fn should_report_no_recovery_freeze_to_lift_when_none_was_set() {
    let mut cache = WarmExpertSlotTableCache::new(1);
    cache.update_maximum_resident_payload_bytes(NORMAL_CEILING_BYTES);
    assert!(!cache.lift_prefill_memory_recovery_freeze());
    assert_eq!(effective_ceiling_bytes(&cache), NORMAL_CEILING_BYTES);
}
