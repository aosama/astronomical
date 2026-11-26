//! Beside-source coverage for the pure decisions of the post-evaluation
//! flush: pricing a leftover-slot prefetch, finding the experts a table
//! lacks, and choosing the insert policy. The writes themselves run under the
//! serial engine lane against real arrays.

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;

use super::{AstroExpertCache, prefetched_payload_bytes};

#[test]
fn should_price_a_prefetch_at_the_page_average_times_the_experts_written() {
    // 1,000 bytes over four experts is 250 each; two written is 500.
    assert_eq!(prefetched_payload_bytes(1_000, 4, 2), 500);
}

#[test]
fn should_price_a_prefetch_that_wrote_nothing_at_zero_bytes() {
    assert_eq!(prefetched_payload_bytes(1_000, 4, 0), 0);
}

#[test]
fn should_average_over_one_expert_when_the_page_names_none() {
    assert_eq!(prefetched_payload_bytes(1_000, 0, 1), 1_000);
}

#[test]
fn should_saturate_instead_of_overflowing_the_prefetch_price() {
    assert_eq!(prefetched_payload_bytes(u64::MAX, 1, usize::MAX), u64::MAX);
}

#[test]
fn should_report_every_expert_missing_when_the_layer_has_no_table() {
    let astro_expert_cache = AstroExpertCache::new(2);

    let missing_expert_ids = astro_expert_cache.expert_ids_missing_from_table(1, &[9, 3, 5]);

    assert_eq!(missing_expert_ids, vec![9, 3, 5]);
}

#[test]
fn should_report_every_expert_missing_for_a_layer_index_past_the_layer_count() {
    let astro_expert_cache = AstroExpertCache::new(1);

    let missing_expert_ids = astro_expert_cache.expert_ids_missing_from_table(40, &[2, 4]);

    assert_eq!(missing_expert_ids, vec![2, 4]);
}

#[test]
fn should_report_nothing_missing_for_an_empty_expert_list() {
    let astro_expert_cache = AstroExpertCache::new(1);

    assert!(
        astro_expert_cache
            .expert_ids_missing_from_table(0, &[])
            .is_empty()
    );
}

#[test]
fn should_insert_under_the_candidate_policy_only_for_decode_demand() {
    let mut astro_expert_cache = AstroExpertCache::new_with_decode_astro_expert_cache_policy(
        1,
        WorkerDecodeAstroExpertCachePolicy::Candidate,
    );
    astro_expert_cache.decode_step_epoch = 7;

    assert_eq!(
        astro_expert_cache.candidate_decode_step_epoch(true),
        Some(7)
    );
    assert_eq!(astro_expert_cache.candidate_decode_step_epoch(false), None);
}

#[test]
fn should_insert_the_baseline_way_under_the_baseline_policy() {
    let mut astro_expert_cache = AstroExpertCache::new_with_decode_astro_expert_cache_policy(
        1,
        WorkerDecodeAstroExpertCachePolicy::Baseline,
    );
    astro_expert_cache.decode_step_epoch = 7;

    assert_eq!(astro_expert_cache.candidate_decode_step_epoch(true), None);
}
