use crate::serving_ceiling::{
    effective_mlx_memory_ceiling_after_wired_claim, effective_mlx_memory_ceiling_bytes,
};

#[test]
fn should_limit_the_effective_ceiling_by_the_configured_and_machine_bounds() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    for configured_mlx_memory_ceiling_bytes in [
        23_000_000_000,
        28_000_000_000,
        32_000_000_000,
        35_000_000_000,
    ] {
        assert_eq!(
            effective_mlx_memory_ceiling_bytes(
                Some(configured_mlx_memory_ceiling_bytes),
                42_949_672_960
            ),
            configured_mlx_memory_ceiling_bytes
        );
    }
    for configured_mlx_memory_ceiling_bytes in [43_000_000_000, 51_000_000_000] {
        assert_eq!(
            effective_mlx_memory_ceiling_bytes(
                Some(configured_mlx_memory_ceiling_bytes),
                42_949_672_960
            ),
            42_949_672_960
        );
    }
}

#[test]
fn should_use_the_machine_ceiling_without_a_configured_override() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        effective_mlx_memory_ceiling_bytes(None, 42_949_672_960),
        42_949_672_960
    );
    assert_eq!(effective_mlx_memory_ceiling_bytes(None, 0), 0);
}

#[test]
fn should_clamp_the_ceiling_to_a_smaller_wired_budget() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // The 48 GB incident shape (issue #1297): Metal recommends
    // 40,000,028,672 wired bytes against a 42,949,672,960-byte machine
    // ceiling; the engine must plan against the recommendation, not the
    // ceiling, or long prefills grow into wiring the driver refuses.
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(42_949_672_960, Some(40_000_028_672)),
        40_000_028_672
    );
}

#[test]
fn should_keep_the_ceiling_when_the_wired_budget_is_absent_or_larger() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // A budget at or above the ceiling means the wiring can cover every
    // plan the engine can compose, and no budget at all means pageable
    // serving where the ceiling is the only bound.
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(42_949_672_960, None),
        42_949_672_960
    );
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(38_654_705_664, Some(40_000_028_672)),
        38_654_705_664
    );
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(42_949_672_960, Some(42_949_672_960)),
        42_949_672_960
    );
}
