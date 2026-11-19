use super::dispatch_plan::{RowBatchedDispatchPlan, RowBatchedDispatchRefusal, input_slice_count};

#[test]
fn should_plan_one_threadgroup_column_per_tile_and_slice_for_sixteen_rows() {
    let plan =
        RowBatchedDispatchPlan::plan(16, 17408, 5120).expect("the 27B gate projection plans");
    assert_eq!(plan.padded_row_count, 16);
    assert_eq!(plan.row_blocks_per_threadgroup, 1);
    assert_eq!(plan.grid, [544 * 32 * plan.input_slice_count, 1, 1]);
    assert_eq!(plan.threadgroup, [32 * plan.input_slice_count, 1, 1]);
    assert!(!plan.last_block_passes_padding);
}

#[test]
fn should_give_the_narrow_long_down_projection_the_most_input_slices() {
    assert_eq!(input_slice_count(5120, 17408), 8);
    assert!(input_slice_count(17408, 5120) < input_slice_count(5120, 17408));
}

#[test]
fn should_keep_the_slice_count_independent_of_the_row_count() {
    let one_row = RowBatchedDispatchPlan::plan(1, 5120, 6144).expect("one row plans");
    let sixteen_rows = RowBatchedDispatchPlan::plan(16, 5120, 6144).expect("sixteen rows plan");
    assert_eq!(one_row.input_slice_count, sixteen_rows.input_slice_count);
}

#[test]
fn should_pad_rows_to_whole_blocks_and_split_past_thirty_two() {
    let seventeen_rows = RowBatchedDispatchPlan::plan(17, 1024, 5120).expect("17 rows plan");
    assert_eq!(seventeen_rows.padded_row_count, 32);
    assert_eq!(seventeen_rows.row_blocks_per_threadgroup, 2);
    assert_eq!(seventeen_rows.group_sum_grid, [80, 32, 1]);
}

#[test]
fn should_accept_the_boundary_row_counts() {
    assert!(RowBatchedDispatchPlan::plan(1, 1024, 5120).is_ok());
    assert!(RowBatchedDispatchPlan::plan(32, 1024, 5120).is_ok());
}

#[test]
fn should_refuse_row_counts_outside_the_kernel_range() {
    assert_eq!(
        RowBatchedDispatchPlan::plan(0, 1024, 5120),
        Err(RowBatchedDispatchRefusal::RowCount { row_count: 0 })
    );
    assert_eq!(
        RowBatchedDispatchPlan::plan(33, 1024, 5120),
        Err(RowBatchedDispatchRefusal::RowCount { row_count: 33 })
    );
}

#[test]
fn should_refuse_widths_the_kernel_cannot_tile() {
    assert_eq!(
        RowBatchedDispatchPlan::plan(4, 48, 5120),
        Err(RowBatchedDispatchRefusal::OutputWidth { output_width: 48 })
    );
    assert_eq!(
        RowBatchedDispatchPlan::plan(4, 1024, 5000),
        Err(RowBatchedDispatchRefusal::InputWidth { input_width: 5000 })
    );
}
