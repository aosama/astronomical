use super::{StorageSlice, plan_record_reads};

fn slice(file_index: usize, file_offset: u64, byte_count: usize) -> StorageSlice {
    StorageSlice {
        file_index,
        file_offset,
        byte_count,
    }
}

const QUARTER_MEGABYTE: usize = 256 * 1024;

#[test]
fn should_join_slices_that_follow_one_another_into_one_read() {
    let slices = [slice(0, 0, 100), slice(0, 100, 50), slice(0, 150, 10)];
    assert_eq!(plan_record_reads(&slices), vec![vec![0, 1, 2]]);
}

#[test]
fn should_keep_slices_with_a_gap_between_them_as_separate_reads() {
    let slices = [slice(0, 0, 100), slice(0, 101, 50)];
    assert_eq!(plan_record_reads(&slices), vec![vec![0], vec![1]]);
}

#[test]
fn should_never_join_slices_of_different_files() {
    // The second file's slice starts exactly where the first file's ends.
    let slices = [slice(0, 0, 100), slice(1, 100, 50)];
    assert_eq!(plan_record_reads(&slices), vec![vec![0], vec![1]]);
}

#[test]
fn should_order_a_read_by_file_offset_whatever_order_the_slices_came_in() {
    let slices = [slice(0, 150, 10), slice(0, 0, 100), slice(0, 100, 50)];
    assert_eq!(plan_record_reads(&slices), vec![vec![1, 2, 0]]);
}

#[test]
fn should_read_two_experts_records_as_two_reads_when_another_record_lies_between() {
    // Expert A's three slices, a skipped record, then expert B's three.
    let slices = [
        slice(0, 0, 10),
        slice(0, 10, 10),
        slice(0, 20, 10),
        slice(0, 60, 10),
        slice(0, 70, 10),
        slice(0, 80, 10),
    ];
    assert_eq!(
        plan_record_reads(&slices),
        vec![vec![0, 1, 2], vec![3, 4, 5]]
    );
}

#[test]
fn should_cut_a_long_run_where_its_second_half_begins() {
    // Six quarter-megabyte slices back to back: 1.5 MB, cut after the third.
    let mut slices = Vec::new();
    for slice_position in 0..6 {
        slices.push(slice(
            0,
            (slice_position * QUARTER_MEGABYTE) as u64,
            QUARTER_MEGABYTE,
        ));
    }
    assert_eq!(
        plan_record_reads(&slices),
        vec![vec![0, 1, 2], vec![3, 4, 5]]
    );
}

#[test]
fn should_leave_a_run_shorter_than_a_megabyte_whole() {
    let slices = [
        slice(0, 0, QUARTER_MEGABYTE),
        slice(0, QUARTER_MEGABYTE as u64, QUARTER_MEGABYTE),
    ];
    assert_eq!(plan_record_reads(&slices), vec![vec![0, 1]]);
}

#[test]
fn should_leave_one_long_slice_whole_because_a_destination_cannot_be_cut() {
    let slices = [slice(0, 0, 8 * QUARTER_MEGABYTE)];
    assert_eq!(plan_record_reads(&slices), vec![vec![0]]);
}

#[test]
fn should_cut_at_a_slice_boundary_when_one_slice_passes_the_middle() {
    // A small slice then a large one: the middle falls inside the large
    // slice, so the cut comes before it.
    let slices = [
        slice(0, 0, QUARTER_MEGABYTE),
        slice(0, QUARTER_MEGABYTE as u64, 6 * QUARTER_MEGABYTE),
        slice(0, 7 * QUARTER_MEGABYTE as u64, QUARTER_MEGABYTE),
    ];
    assert_eq!(plan_record_reads(&slices), vec![vec![0, 1], vec![2]]);
}

#[test]
fn should_plan_no_read_for_no_slices_and_skip_empty_ones() {
    assert!(plan_record_reads(&[]).is_empty());
    let slices = [slice(0, 0, 0), slice(0, 0, 10)];
    assert_eq!(plan_record_reads(&slices), vec![vec![1]]);
}
