use super::dispatch_plan::{
    SortedExpertDispatchPlan, SortedExpertDispatchRefusal, SortedExpertLayout, row_tile_bound,
};

const WIDE: SortedExpertLayout = SortedExpertLayout {
    simdgroups_across_columns: 4,
    stages_rows: true,
    rows_per_pass: 96,
    splits_experts_into_tiles: false,
};
const NARROW: SortedExpertLayout = SortedExpertLayout {
    simdgroups_across_columns: 2,
    stages_rows: false,
    rows_per_pass: 64,
    splits_experts_into_tiles: false,
};
const TILED: SortedExpertLayout = SortedExpertLayout {
    splits_experts_into_tiles: true,
    ..WIDE
};

// --- positive ---

#[test]
fn should_plan_one_threadgroup_per_expert_and_column_block() {
    let plan =
        SortedExpertDispatchPlan::plan(16_384, 512, 2_048, 256, WIDE).expect("the gate is taken");
    assert_eq!(plan.grid, [4 * 256, 256, 1]);
    assert_eq!(plan.threadgroup, [256, 1, 1]);
    assert_eq!(plan.first_row_grid, [256, 1, 1]);
    assert_eq!(plan.first_row_threadgroup, [64, 1, 1]);
}

#[test]
fn should_plan_the_down_projection_with_four_times_the_column_blocks() {
    let plan =
        SortedExpertDispatchPlan::plan(16_384, 2_048, 512, 256, WIDE).expect("the down is taken");
    assert_eq!(plan.grid[0], 16 * 256);
}

#[test]
fn should_plan_a_narrow_tile_with_half_the_threads_and_twice_the_column_blocks() {
    let plan =
        SortedExpertDispatchPlan::plan(16_384, 512, 2_048, 256, NARROW).expect("the gate is taken");
    assert_eq!(plan.grid[0], 8 * 128);
    assert_eq!(plan.threadgroup, [128, 1, 1]);
}

// --- edge / boundary ---

#[test]
fn should_plan_a_single_expert_as_one_row_of_threadgroups() {
    let plan = SortedExpertDispatchPlan::plan(1, 128, 64, 1, WIDE).expect("one row is taken");
    assert_eq!(plan.grid, [256, 1, 1]);
    assert_eq!(plan.first_row_threadgroup, [1, 1, 1]);
}

#[test]
fn should_search_few_experts_in_one_small_threadgroup() {
    let plan =
        SortedExpertDispatchPlan::plan(32, 128, 64, 4, WIDE).expect("a tiny multiply is taken");
    assert_eq!(plan.first_row_threadgroup, [4, 1, 1]);
}

// --- negative ---

#[test]
fn should_refuse_output_widths_off_the_128_column_tile() {
    assert_eq!(
        SortedExpertDispatchPlan::plan(16_384, 576, 2_048, 256, WIDE),
        Err(SortedExpertDispatchRefusal::OutputWidth {
            output_width: 576,
            tile_width: 128
        })
    );
}

#[test]
fn should_refuse_input_widths_off_the_quantization_group() {
    assert_eq!(
        SortedExpertDispatchPlan::plan(16_384, 512, 2_000, 256, WIDE),
        Err(SortedExpertDispatchRefusal::InputWidth { input_width: 2_000 })
    );
}

#[test]
fn should_refuse_an_empty_multiply() {
    assert!(matches!(
        SortedExpertDispatchPlan::plan(0, 512, 2_048, 256, WIDE),
        Err(SortedExpertDispatchRefusal::EmptyMultiply { .. })
    ));
}

#[test]
fn should_refuse_a_layout_the_kernel_has_no_tile_for() {
    let three_across = SortedExpertLayout {
        simdgroups_across_columns: 3,
        ..WIDE
    };
    let thirty_two_rows = SortedExpertLayout {
        rows_per_pass: 32,
        ..WIDE
    };
    assert_eq!(
        SortedExpertDispatchPlan::plan(16_384, 512, 2_048, 256, three_across),
        Err(SortedExpertDispatchRefusal::Layout {
            simdgroups_across_columns: 3,
            rows_per_pass: 96
        })
    );
    assert!(SortedExpertDispatchPlan::plan(16_384, 512, 2_048, 256, thirty_two_rows).is_err());
}

#[test]
fn should_plan_one_threadgroup_per_possible_row_tile_when_experts_split() {
    let plan =
        SortedExpertDispatchPlan::plan(16_384, 512, 2_048, 256, TILED).expect("the gate is taken");
    assert_eq!(plan.grid[1], 16_384 / 96 + 1 + 255);
}

#[test]
fn should_never_bound_more_row_tiles_than_rows() {
    assert_eq!(row_tile_bound(8, 256, 96), 8);
    assert_eq!(row_tile_bound(96, 1, 96), 1);
    assert_eq!(row_tile_bound(97, 1, 96), 2);
}
