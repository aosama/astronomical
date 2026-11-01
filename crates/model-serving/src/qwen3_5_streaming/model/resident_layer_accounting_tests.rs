//! Unit contracts for how a packed full hit from a seated complete layer is
//! accounted: one decode token is credited at once, a multi-token chunk is
//! queued exactly once so its expert demand is never counted twice.

use super::{ResidentPackedFullHitAccounting, resident_packed_full_hit_accounting};

#[test]
fn should_credit_a_single_decode_token_immediately() {
    assert_eq!(
        resident_packed_full_hit_accounting(1),
        ResidentPackedFullHitAccounting::CreditDecodeTokenNow
    );
}

#[test]
fn should_defer_a_multi_token_chunk_through_one_queue() {
    assert_eq!(
        resident_packed_full_hit_accounting(2_048),
        ResidentPackedFullHitAccounting::DeferChunkRouteOnce
    );
}

#[test]
fn should_defer_the_smallest_multi_token_chunk() {
    assert_eq!(
        resident_packed_full_hit_accounting(2),
        ResidentPackedFullHitAccounting::DeferChunkRouteOnce
    );
}

#[test]
fn should_never_credit_an_empty_route_as_a_decode_token() {
    assert_eq!(
        resident_packed_full_hit_accounting(0),
        ResidentPackedFullHitAccounting::DeferChunkRouteOnce
    );
}
