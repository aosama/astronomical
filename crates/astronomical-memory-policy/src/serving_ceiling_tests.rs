use crate::serving_ceiling::{
    effective_mlx_memory_ceiling_after_wired_claim, effective_mlx_memory_ceiling_bytes,
    floor_raise_limit_bytes, floor_raised_ceiling_bytes, plannable_wired_residency_bytes,
};

/// The claim the 48 GB incident machine received on 2026-10-24.
const INCIDENT_WIRED_RESIDENCY_BUDGET_BYTES: u64 = 40_200_896_512;
/// 2.5% of the incident claim, rounded down to whole thousandths.
const INCIDENT_METAL_WORKING_SET_MARGIN_BYTES: u64 = 1_005_022_400;

#[test]
fn should_limit_the_effective_ceiling_by_the_configured_and_machine_bounds() {
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
    assert_eq!(
        effective_mlx_memory_ceiling_bytes(None, 42_949_672_960),
        42_949_672_960
    );
    assert_eq!(effective_mlx_memory_ceiling_bytes(None, 0), 0);
}

#[test]
fn should_clamp_the_ceiling_to_the_plannable_part_of_a_smaller_wired_budget() {
    // The 48 GB incident shape (issue #1297): Metal recommends
    // 40,000,028,672 wired bytes against a 42,949,672,960-byte machine
    // ceiling; the engine must plan below the recommendation, not at the
    // ceiling, or long prefills grow into wiring the driver refuses.
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(42_949_672_960, Some(40_000_028_672)),
        40_000_028_672 - 1_000_000_700
    );
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(
            42_949_672_960,
            Some(INCIDENT_WIRED_RESIDENCY_BUDGET_BYTES)
        ),
        INCIDENT_WIRED_RESIDENCY_BUDGET_BYTES - INCIDENT_METAL_WORKING_SET_MARGIN_BYTES
    );
}

#[test]
fn should_keep_a_two_and_a_half_percent_metal_working_set_margin_below_the_wired_claim() {
    // The 2026-10-24 122B decode died at 40,051,177,002 MLX bytes; the
    // plannable bound must sit below that measured failure point.
    let plannable_bytes = plannable_wired_residency_bytes(INCIDENT_WIRED_RESIDENCY_BUDGET_BYTES);
    assert_eq!(
        plannable_bytes,
        INCIDENT_WIRED_RESIDENCY_BUDGET_BYTES - INCIDENT_METAL_WORKING_SET_MARGIN_BYTES
    );
    assert!(plannable_bytes < 40_051_177_002);
}

#[test]
fn should_resolve_a_zero_or_tiny_wired_claim_without_underflow() {
    assert_eq!(plannable_wired_residency_bytes(0), 0);
    // Below one thousand bytes the per-thousand margin rounds to zero.
    assert_eq!(plannable_wired_residency_bytes(999), 999);
    assert_eq!(plannable_wired_residency_bytes(1_000), 975);
    assert_eq!(
        plannable_wired_residency_bytes(u64::MAX),
        u64::MAX - u64::MAX / 1_000 * 25
    );
}

#[test]
fn should_keep_the_ceiling_when_the_wired_budget_is_absent_or_larger() {
    // A plannable budget at or above the ceiling means the wiring can cover
    // every plan the engine can compose, and no budget at all means pageable
    // serving where the ceiling is the only bound.
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(42_949_672_960, None),
        42_949_672_960
    );
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(38_654_705_664, Some(40_000_028_672)),
        38_654_705_664
    );
    // Exactly at the plannable bound the configured ceiling is kept as is,
    // and one byte above it is clamped down to the bound.
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(39_000_027_972, Some(40_000_028_672)),
        39_000_027_972
    );
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(39_000_027_973, Some(40_000_028_672)),
        39_000_027_972
    );
    // A claim equal to the ceiling still keeps the Metal margin free.
    assert_eq!(
        effective_mlx_memory_ceiling_after_wired_claim(42_949_672_960, Some(42_949_672_960)),
        plannable_wired_residency_bytes(42_949_672_960)
    );
}

const BYTES_PER_MARKETED_GIGABYTE: u64 = 1_073_741_824;

#[test]
fn should_treat_no_configured_maximum_or_one_at_the_machine_as_no_user_limit() {
    let machine_ceiling_bytes = 16 * BYTES_PER_MARKETED_GIGABYTE;
    assert_eq!(
        floor_raise_limit_bytes(None, machine_ceiling_bytes),
        u64::MAX
    );
    assert_eq!(
        floor_raise_limit_bytes(Some(machine_ceiling_bytes), machine_ceiling_bytes),
        u64::MAX
    );
    assert_eq!(
        floor_raise_limit_bytes(Some(machine_ceiling_bytes - 1), machine_ceiling_bytes),
        machine_ceiling_bytes - 1,
        "one byte under the machine is a limit the user chose"
    );
}

#[test]
fn should_serve_a_model_at_its_floor_when_the_user_set_no_limit() {
    // An 8 GB Mac whose plannable wired residency is about 5.4 GB, loading a
    // dense chat model whose floor is 6.6 GB (#1902).
    assert_eq!(
        floor_raised_ceiling_bytes(5_400_000_000, 6_600_000_000, u64::MAX),
        Some(6_600_000_000)
    );
}

#[test]
fn should_raise_only_up_to_a_user_limit_above_the_wired_clamp() {
    // A 16 GB Mac: the wired clamp leaves about 10.4 GB, the user chose 12 GB.
    // A 11 GB floor fits the user's limit and is raised; a 14 GB floor would
    // pass it and is not.
    let wired_clamped_ceiling_bytes = 10_400_000_000;
    let user_limit_bytes = 12 * BYTES_PER_MARKETED_GIGABYTE;
    assert_eq!(
        floor_raised_ceiling_bytes(
            wired_clamped_ceiling_bytes,
            11 * BYTES_PER_MARKETED_GIGABYTE,
            user_limit_bytes
        ),
        Some(11 * BYTES_PER_MARKETED_GIGABYTE)
    );
    assert_eq!(
        floor_raised_ceiling_bytes(
            wired_clamped_ceiling_bytes,
            14 * BYTES_PER_MARKETED_GIGABYTE,
            user_limit_bytes
        ),
        None
    );
    assert_eq!(
        floor_raised_ceiling_bytes(
            wired_clamped_ceiling_bytes,
            user_limit_bytes,
            user_limit_bytes
        ),
        Some(user_limit_bytes),
        "a floor exactly at the user's limit is within it"
    );
}

#[test]
fn should_keep_a_user_limit_below_the_model_floor() {
    // A 48 GB Mac whose user chose 4 GB: the model's 6.6 GB floor stays refused.
    let user_limit_bytes = 4 * BYTES_PER_MARKETED_GIGABYTE;
    assert_eq!(
        floor_raised_ceiling_bytes(user_limit_bytes, 6_600_000_000, user_limit_bytes),
        None
    );
}

#[test]
fn should_leave_the_ceiling_alone_when_the_floor_fits_or_no_raise_is_allowed() {
    let ceiling_bytes = 8 * BYTES_PER_MARKETED_GIGABYTE;
    for model_floor_bytes in [1, ceiling_bytes - 1, ceiling_bytes] {
        assert_eq!(
            floor_raised_ceiling_bytes(ceiling_bytes, model_floor_bytes, u64::MAX),
            None,
            "floor {model_floor_bytes} fits under the ceiling"
        );
    }
    assert_eq!(
        floor_raised_ceiling_bytes(ceiling_bytes, ceiling_bytes + 1, 0),
        None,
        "a zero raise limit (not yet known) never raises"
    );
}

#[test]
fn should_raise_on_each_machine_size_only_when_its_own_memory_binds() {
    for sticker_gigabytes in [8_u64, 16, 24, 48] {
        let machine_ceiling_bytes = sticker_gigabytes * BYTES_PER_MARKETED_GIGABYTE;
        let model_floor_bytes = 12 * BYTES_PER_MARKETED_GIGABYTE;
        let expected_ceiling =
            (model_floor_bytes > machine_ceiling_bytes).then_some(model_floor_bytes);
        assert_eq!(
            floor_raised_ceiling_bytes(
                machine_ceiling_bytes,
                model_floor_bytes,
                floor_raise_limit_bytes(None, machine_ceiling_bytes)
            ),
            expected_ceiling,
            "{sticker_gigabytes} GB Mac with a 12 GB floor"
        );
    }
}
