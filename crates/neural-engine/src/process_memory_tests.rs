use super::{MemoryGrowth, MemorySnapshot, mebibytes, physical_memory_bytes};

fn snapshot(process_physical_footprint_bytes: u64, system_wired_bytes: u64) -> MemorySnapshot {
    MemorySnapshot {
        process_physical_footprint_bytes,
        system_wired_bytes,
    }
}

#[test]
fn should_read_a_non_zero_footprint_for_this_process() {
    let current = MemorySnapshot::capture();

    assert!(current.process_physical_footprint_bytes > 0);
    assert!(current.system_wired_bytes > 0);
}

#[test]
fn should_read_more_installed_memory_than_this_process_uses() {
    let installed = physical_memory_bytes();

    assert!(installed > MemorySnapshot::capture().process_physical_footprint_bytes);
}

#[test]
fn should_see_the_footprint_grow_after_touching_a_large_allocation() {
    let before = MemorySnapshot::capture();
    let touched_pages = std::hint::black_box(vec![1_u8; 64 * 1024 * 1024]);

    let growth = MemoryGrowth::between(before, MemorySnapshot::capture());

    assert!(growth.process_physical_footprint_bytes >= 32 * 1024 * 1024);
    drop(touched_pages);
}

#[test]
fn should_report_growth_as_after_minus_before() {
    let growth = MemoryGrowth::between(snapshot(100, 1_000), snapshot(350, 900));

    assert_eq!(growth.process_physical_footprint_bytes, 250);
    assert_eq!(growth.system_wired_bytes, -100);
}

#[test]
fn should_clamp_a_difference_beyond_the_signed_range() {
    let growth = MemoryGrowth::between(snapshot(0, u64::MAX), snapshot(u64::MAX, 0));

    assert_eq!(growth.process_physical_footprint_bytes, i64::MAX);
    assert_eq!(growth.system_wired_bytes, i64::MIN);
}

#[test]
fn should_convert_bytes_to_mebibytes() {
    assert_eq!(mebibytes(3 * 1024 * 1024), 3.0);
    assert_eq!(mebibytes(-512 * 1024), -0.5);
    assert_eq!(mebibytes(0), 0.0);
}
