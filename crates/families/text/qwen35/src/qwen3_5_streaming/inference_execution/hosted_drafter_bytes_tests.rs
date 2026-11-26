use super::{MemoryAroundALoad, bytes_a_load_added};

const GIGABYTE: u64 = 1 << 30;

fn memory(active_bytes: u64, kept_weight_bytes: u64) -> MemoryAroundALoad {
    MemoryAroundALoad {
        active_bytes,
        kept_weight_bytes,
    }
}

#[test]
fn should_count_the_whole_rise_when_no_kept_weight_was_given_back() {
    // Room was already there: 18 GB limit, the kept layers untouched.
    let added = bytes_a_load_added(
        memory(15 * GIGABYTE, 13 * GIGABYTE),
        memory(16 * GIGABYTE, 13 * GIGABYTE),
    );

    assert_eq!(added, GIGABYTE);
}

#[test]
fn should_count_a_drafter_whose_room_was_made_by_giving_kept_layers_back() {
    // A 12 GB limit: active memory ends where it began, because a gigabyte
    // of kept layers left as the drafter's gigabyte arrived.
    let added = bytes_a_load_added(
        memory(12 * GIGABYTE, 11 * GIGABYTE),
        memory(12 * GIGABYTE, 10 * GIGABYTE),
    );

    assert_eq!(added, GIGABYTE);
}

#[test]
fn should_count_nothing_when_memory_outside_the_kept_weights_did_not_grow() {
    // Negative: a load that failed and freed what it had read.
    let nothing_added = bytes_a_load_added(
        memory(12 * GIGABYTE, 10 * GIGABYTE),
        memory(11 * GIGABYTE, 10 * GIGABYTE),
    );

    assert_eq!(nothing_added, 0);
}

#[test]
fn should_count_nothing_when_memory_could_not_be_read() {
    // Edge: an unreadable snapshot is zero on both sides, and kept weights
    // above a zero reading must not underflow.
    assert_eq!(bytes_a_load_added(memory(0, 0), memory(0, 0)), 0);
    assert_eq!(
        bytes_a_load_added(memory(0, 10 * GIGABYTE), memory(0, 9 * GIGABYTE)),
        0
    );
}
