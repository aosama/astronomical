use super::dispatch_plan::{RowBatchedDispatchRefusal, input_slice_count};
use super::register_ordered_plan::{
    REGISTER_ORDERED_ROWS_PER_DISPATCH, RegisterOrderedDispatchPlan,
    register_order_conversion_grid, register_ordered_slice_count,
};

#[test]
fn should_put_each_sixteen_row_block_on_the_grids_second_axis() {
    let plan = RegisterOrderedDispatchPlan::plan(40, 17408, 5120, 2).expect("40 rows plan");
    assert_eq!(plan.padded_row_count, 48);
    assert_eq!(plan.grid, [544 * 32 * 2, 3, 1]);
    assert_eq!(plan.threadgroup, [64, 1, 1]);
    assert_eq!(plan.group_sum_grid, [80, 48, 1]);
}

#[test]
fn should_take_the_measured_slices_for_listed_shapes_and_the_rule_for_others() {
    assert_eq!(register_ordered_slice_count(5120, 17408), 4);
    assert_eq!(register_ordered_slice_count(10240, 5120), 2);
    assert_eq!(
        register_ordered_slice_count(17408, 5120),
        input_slice_count(17408, 5120)
    );
}

#[test]
fn should_accept_one_row_and_a_full_dispatch() {
    assert!(RegisterOrderedDispatchPlan::plan(1, 1024, 5120, 4).is_ok());
    assert!(
        RegisterOrderedDispatchPlan::plan(REGISTER_ORDERED_ROWS_PER_DISPATCH, 1024, 5120, 4)
            .is_ok()
    );
}

#[test]
fn should_refuse_rows_past_one_dispatch_and_untileable_widths() {
    assert_eq!(
        RegisterOrderedDispatchPlan::plan(129, 1024, 5120, 4),
        Err(RowBatchedDispatchRefusal::RowCount { row_count: 129 })
    );
    assert_eq!(
        RegisterOrderedDispatchPlan::plan(8, 48, 5120, 4),
        Err(RowBatchedDispatchRefusal::OutputWidth { output_width: 48 })
    );
}

#[test]
fn should_convert_one_block_per_tile_and_group() {
    assert_eq!(
        register_order_conversion_grid(1024, 5120),
        ([32 * 80 * 256, 1, 1], [256, 1, 1])
    );
}
