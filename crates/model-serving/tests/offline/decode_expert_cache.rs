//! Decode expert cache policy: coverage, admission, and eviction order.
//!
//! The irreducible unit is one expert. Bytes and pages are accounting and I/O
//! batching, never ownership. A repeated token route must be a complete hit
//! against the resident experts, and eviction removes the coldest experts.

use astronomical_model_serving::{DecodeExpertCache, ResidentExpertWeight};

#[derive(Clone, Copy, Debug)]
struct FakeExpertWeight {
    payload_bytes: u64,
}

impl ResidentExpertWeight for FakeExpertWeight {
    fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

fn fake_weight(payload_bytes: u64) -> FakeExpertWeight {
    FakeExpertWeight { payload_bytes }
}

fn admit_experts(
    cache: &mut DecodeExpertCache<FakeExpertWeight>,
    slot: usize,
    expert_ids: &[usize],
) {
    for expert_id in expert_ids {
        cache.admit(slot, *expert_id, fake_weight(8));
    }
}

#[test]
fn should_report_no_missing_experts_when_the_route_is_already_resident() {
    let mut decode_cache = DecodeExpertCache::new(4);
    admit_experts(&mut decode_cache, 2, &[2, 5, 7]);
    assert!(decode_cache.contains_every(2, &[2, 5, 7]));
    let missing_expert_ids = decode_cache.missing(2, &[2, 5]);
    assert!(
        missing_expert_ids.is_empty(),
        "a repeated token route must not reload experts already in the decode cache"
    );
}

#[test]
fn should_evict_the_coldest_experts_instead_of_dropping_the_whole_layer() {
    let mut decode_cache = DecodeExpertCache::new(1);
    admit_experts(&mut decode_cache, 0, &[0, 1, 2, 3, 4, 5]);
    decode_cache.record_demand(0, &[0, 1, 2, 3, 4, 5]);
    decode_cache.record_demand(0, &[0, 1, 2, 3]);
    decode_cache.set_ceiling(40);
    let remaining_expert_ids = decode_cache.resident_ids(0);
    assert!(
        remaining_expert_ids.len() < 6,
        "a tight ceiling must evict some experts"
    );
    assert!(
        !remaining_expert_ids.is_empty(),
        "eviction must not wipe the whole layer when some experts still fit"
    );
    assert!(
        remaining_expert_ids.contains(&0) && remaining_expert_ids.contains(&1),
        "hotter experts must survive colder ones"
    );
}

#[test]
fn should_keep_the_current_route_when_enforcing_the_decode_ceiling() {
    let mut decode_cache = DecodeExpertCache::new(1);
    admit_experts(&mut decode_cache, 0, &[0, 1, 2, 3]);
    decode_cache.record_demand(0, &[0, 1]);
    let vacated_slots = decode_cache.enforce_ceiling(&[(0, 2), (0, 3)]);
    let remaining_expert_ids = decode_cache.resident_ids(0);
    assert_eq!(vacated_slots, vec![0]);
    assert!(
        remaining_expert_ids.contains(&2) && remaining_expert_ids.contains(&3),
        "the current decode route must survive ceiling enforcement"
    );
    assert!(
        !remaining_expert_ids.contains(&0) && !remaining_expert_ids.contains(&1),
        "unprotected colder experts must still be eligible for eviction"
    );
}
