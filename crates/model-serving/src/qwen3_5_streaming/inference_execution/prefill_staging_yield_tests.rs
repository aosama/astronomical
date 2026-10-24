use super::staging_yield_bytes;

const STAGING_RESERVE_BYTES: u64 = 213_909_504;
const WARM_CACHE_BYTES: u64 = 15_001_583_616;

#[test]
fn should_borrow_nothing_when_admitted_headroom_covers_the_staging_reserve() {
    let unreserved_headroom_bytes = 4_000_000_000;

    assert_eq!(
        staging_yield_bytes(
            STAGING_RESERVE_BYTES,
            unreserved_headroom_bytes,
            WARM_CACHE_BYTES
        ),
        0
    );
}

#[test]
fn should_borrow_nothing_when_headroom_equals_the_staging_reserve_exactly() {
    assert_eq!(
        staging_yield_bytes(
            STAGING_RESERVE_BYTES,
            STAGING_RESERVE_BYTES,
            WARM_CACHE_BYTES
        ),
        0
    );
}

#[test]
fn should_borrow_only_the_shortfall_when_headroom_covers_part_of_the_reserve() {
    let unreserved_headroom_bytes = STAGING_RESERVE_BYTES - 1_000;

    assert_eq!(
        staging_yield_bytes(
            STAGING_RESERVE_BYTES,
            unreserved_headroom_bytes,
            WARM_CACHE_BYTES
        ),
        1_000
    );
}

#[test]
fn should_borrow_the_whole_reserve_when_admission_left_no_headroom() {
    assert_eq!(
        staging_yield_bytes(STAGING_RESERVE_BYTES, 0, WARM_CACHE_BYTES),
        STAGING_RESERVE_BYTES
    );
}

#[test]
fn should_never_borrow_more_than_half_the_warm_cache() {
    let small_warm_cache_bytes = 100_000_000;

    assert_eq!(
        staging_yield_bytes(STAGING_RESERVE_BYTES, 0, small_warm_cache_bytes),
        50_000_000
    );
}

#[test]
fn should_borrow_nothing_from_an_empty_warm_cache() {
    assert_eq!(staging_yield_bytes(STAGING_RESERVE_BYTES, 0, 0), 0);
}
