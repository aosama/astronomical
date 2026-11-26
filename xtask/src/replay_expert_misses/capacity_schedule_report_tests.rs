use memory_policy::astro_expert_cache_model::CapacityMoment;

use super::{WiredOutsideMlxReading, reserve_by_moment};

const GIBIBYTE: u64 = 1 << 30;

/// A moment at a 34 GiB limit under a 100 GiB wire line: 1 GiB of slack
/// leaves MLX 99 GiB less what is wired outside it.
fn moment_with_outside(wired_outside_mlx_gibibytes: u64) -> CapacityMoment {
    CapacityMoment {
        reply_token_index: 0,
        kept_expert_capacity_bytes: 20 * GIBIBYTE,
        system_wired_reserve_bytes: 0,
        wired_outside_mlx_bytes: wired_outside_mlx_gibibytes * GIBIBYTE,
        limit_bytes: 34 * GIBIBYTE,
        system_wire_limit_bytes: 100 * GIBIBYTE,
        released_expert_bytes: 0,
    }
}

/// Steady at 63 GiB outside MLX (no reserve), one reading at 67 GiB (2 GiB
/// reserve as read), then steady again.
fn moments_with_one_spike() -> Vec<CapacityMoment> {
    vec![
        moment_with_outside(63),
        moment_with_outside(63),
        moment_with_outside(67),
        moment_with_outside(63),
    ]
}

#[test]
fn should_reserve_for_a_spike_as_read() {
    let reserves = reserve_by_moment(&moments_with_one_spike(), WiredOutsideMlxReading::AsRead);
    assert_eq!(reserves, vec![0, 0, 2 * GIBIBYTE, 0]);
}

#[test]
fn should_reserve_nothing_for_a_spike_one_reading_shows() {
    let reserves = reserve_by_moment(
        &moments_with_one_spike(),
        WiredOutsideMlxReading::TwoReadingsInARow,
    );
    assert_eq!(reserves, vec![0, 0, 0, 0]);
}

#[test]
fn should_count_a_repeated_moment_of_one_reading_as_that_one_reading() {
    // The spike recorded at two moments in a row is still one reading.
    let mut moments = moments_with_one_spike();
    moments.insert(2, moment_with_outside(67));
    let reserves = reserve_by_moment(&moments, WiredOutsideMlxReading::TwoReadingsInARow);
    assert_eq!(reserves, vec![0, 0, 0, 0, 0]);
}

#[test]
fn should_reserve_for_a_rise_once_enough_readings_show_it() {
    let moments = [
        moment_with_outside(63),
        moment_with_outside(67),
        moment_with_outside(68),
        moment_with_outside(69),
        moment_with_outside(70),
    ];
    let two = reserve_by_moment(&moments, WiredOutsideMlxReading::TwoReadingsInARow);
    let four = reserve_by_moment(&moments, WiredOutsideMlxReading::FourReadingsInARow);
    assert_eq!(two[2], 2 * GIBIBYTE, "the smaller of 67 and 68 GiB counts");
    assert_eq!(
        four[4],
        2 * GIBIBYTE,
        "the smallest of the last four counts"
    );
    assert_eq!(four[3], 0, "63 GiB is still among the last four readings");
}

#[test]
fn should_reserve_nothing_at_all_as_the_bound() {
    let reserves = reserve_by_moment(&moments_with_one_spike(), WiredOutsideMlxReading::NoReserve);
    assert_eq!(reserves, vec![0; 4]);
}
