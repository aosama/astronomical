//! Boundary contracts for decode expert-cache eviction.
//!
//! These tests pin the eviction edges: an exact ceiling boundary evicts
//! nothing, a fully protected over-ceiling cache starves visibly instead of
//! looping, equal demand-per-byte evicts in stable slot-then-id order, and
//! zero-payload experts never satisfy an eviction demand.

use crate::expert_cache::decode::{DecodeExpertCache, ResidentExpertWeight};

#[derive(Clone, Copy, Debug)]
struct BoundaryExpertWeight {
    payload_bytes: u64,
}

impl ResidentExpertWeight for BoundaryExpertWeight {
    fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

fn admit(
    decode_cache: &mut DecodeExpertCache<BoundaryExpertWeight>,
    slot: usize,
    expert_id: usize,
    payload_bytes: u64,
) {
    decode_cache.admit(slot, expert_id, BoundaryExpertWeight { payload_bytes });
}

#[test]
fn should_evict_nothing_when_the_payload_lands_exactly_on_the_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut decode_cache = DecodeExpertCache::new(1);
    admit(&mut decode_cache, 0, 0, 8);
    admit(&mut decode_cache, 0, 1, 8);

    decode_cache.set_ceiling(16);

    assert_eq!(decode_cache.total_payload_bytes(), 16);
    assert_eq!(decode_cache.eviction_count(), 0);
    assert_eq!(decode_cache.resident_ids(0).len(), 2);
}

#[test]
fn should_starve_visibly_when_every_over_ceiling_expert_is_protected() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut decode_cache = DecodeExpertCache::new(1);
    admit(&mut decode_cache, 0, 0, 8);
    admit(&mut decode_cache, 0, 1, 8);

    let vacated_slots = decode_cache.enforce_ceiling(&[(0, 0), (0, 1)]);

    assert!(
        vacated_slots.is_empty(),
        "protected experts must never be evicted"
    );
    assert_eq!(
        decode_cache.total_payload_bytes(),
        16,
        "an over-ceiling cache with no evictable experts stays over: the caller must see the starvation instead of a silent wipe"
    );
}

#[test]
fn should_break_demand_per_byte_ties_in_slot_then_expert_id_order() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut decode_cache = DecodeExpertCache::new(2);
    admit(&mut decode_cache, 0, 0, 8);
    admit(&mut decode_cache, 0, 1, 8);
    admit(&mut decode_cache, 0, 2, 8);
    admit(&mut decode_cache, 1, 0, 8);
    admit(&mut decode_cache, 1, 1, 8);
    for slot in [0, 1] {
        decode_cache.record_demand(slot, &[0, 1, 2]);
    }

    decode_cache.set_ceiling(24);

    // All five experts have identical demand-per-byte, so eviction is
    // (slot, expert id) ascending: slot 0 ids 0 and 1 go, the rest stay.
    assert_eq!(decode_cache.resident_ids(0), vec![2]);
    assert_eq!(decode_cache.resident_ids(1), vec![0, 1]);
    assert_eq!(decode_cache.eviction_count(), 2);
}

#[test]
fn should_not_count_zero_payload_experts_toward_an_eviction_demand() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut decode_cache = DecodeExpertCache::new(1);
    admit(&mut decode_cache, 0, 0, 0);
    admit(&mut decode_cache, 0, 1, 16);

    decode_cache.set_ceiling(0);

    assert_eq!(
        decode_cache.resident_ids(0),
        vec![0],
        "a zero-payload expert frees nothing and must never be chosen as an eviction candidate"
    );
    assert_eq!(decode_cache.total_payload_bytes(), 0);
    assert_eq!(decode_cache.eviction_count(), 1);
}

#[test]
fn should_drop_every_unprotected_expert_at_a_zero_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut decode_cache = DecodeExpertCache::new(2);
    admit(&mut decode_cache, 0, 0, 8);
    admit(&mut decode_cache, 1, 3, 8);
    admit(&mut decode_cache, 1, 4, 8);

    decode_cache.set_ceiling(0);

    assert!(decode_cache.resident_ids(0).is_empty());
    assert!(decode_cache.resident_ids(1).is_empty());
    assert_eq!(decode_cache.total_payload_bytes(), 0);
    assert_eq!(decode_cache.eviction_count(), 3);
}
