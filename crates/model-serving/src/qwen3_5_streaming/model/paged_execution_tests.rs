//! Direct-MLX contracts for the decode-token row shaping that routes a
//! jangtq2 projection to the fused gathered projection kernel: one shared
//! row or one row per assignment is a decode token, anything wider is a
//! chunk that keeps the materializing path.

use super::paged_execution::{decode_token_activation_rows, expert_assignment_order};
use crate::sparse_experts::ExpertAssignmentOrder;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::MlxArray;
use serial_test::serial;

const INPUT_WIDTH: usize = 64;
const EXPERTS_PER_TOKEN: usize = 8;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the decode-token row test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn activations(runtime: &MlxRuntime, shape: &[i32]) -> MlxArray {
    let element_count = shape.iter().map(|&dimension| dimension as usize).product();
    runtime
        .array_from_f32(&vec![0.5; element_count], shape)
        .expect("the activation fixture should upload")
}

fn route(assignment_count: usize) -> MlxArray {
    let expert_indices: Vec<i32> = (0..assignment_count as i32).collect();
    MlxArray::from_i32(&expert_indices, &[1, 1, assignment_count as i32])
        .expect("the route fixture should upload")
}

#[test]
#[serial]
fn should_flatten_one_shared_decode_token_row_for_every_assignment() {
    let runtime = test_runtime();
    let rows = decode_token_activation_rows(
        &runtime,
        &activations(&runtime, &[1, 1, 1, INPUT_WIDTH as i32]),
        &route(EXPERTS_PER_TOKEN),
        EXPERTS_PER_TOKEN,
    )
    .expect("a decode token should shape")
    .expect("a single shared row is a decode token");
    assert_eq!(rows.shape(), [1, INPUT_WIDTH as i32]);
}

#[test]
#[serial]
fn should_flatten_one_row_per_assignment_on_the_sorted_decode_path() {
    let runtime = test_runtime();
    let rows = decode_token_activation_rows(
        &runtime,
        &activations(&runtime, &[EXPERTS_PER_TOKEN as i32, 1, INPUT_WIDTH as i32]),
        &route(EXPERTS_PER_TOKEN),
        EXPERTS_PER_TOKEN,
    )
    .expect("a sorted decode token should shape")
    .expect("one row per assignment is a decode token");
    assert_eq!(rows.shape(), [EXPERTS_PER_TOKEN as i32, INPUT_WIDTH as i32]);
}

#[test]
#[serial]
fn should_keep_a_multi_token_chunk_on_the_materializing_path() {
    let runtime = test_runtime();
    // Two tokens route sixteen assignments: more than one token routes, so
    // the chunk belongs to gather_mm even though its row count matches.
    let chunk_rows = decode_token_activation_rows(
        &runtime,
        &activations(
            &runtime,
            &[2 * EXPERTS_PER_TOKEN as i32, 1, INPUT_WIDTH as i32],
        ),
        &route(2 * EXPERTS_PER_TOKEN),
        EXPERTS_PER_TOKEN,
    )
    .expect("a chunk should shape");
    assert!(chunk_rows.is_none());
    // Two shared rows against eight assignments match neither layout.
    let mismatched_rows = decode_token_activation_rows(
        &runtime,
        &activations(&runtime, &[2, 1, 1, INPUT_WIDTH as i32]),
        &route(EXPERTS_PER_TOKEN),
        EXPERTS_PER_TOKEN,
    )
    .expect("a mismatched layout should shape");
    assert!(mismatched_rows.is_none());
}

#[test]
#[serial]
fn should_reject_empty_routes_and_partial_routes_stay_decode_tokens() {
    let runtime = test_runtime();
    let empty_route = MlxArray::from_i32(&[], &[1, 1, 0]).expect("an empty route should upload");
    assert!(
        decode_token_activation_rows(
            &runtime,
            &activations(&runtime, &[1, 1, 1, INPUT_WIDTH as i32]),
            &empty_route,
            EXPERTS_PER_TOKEN,
        )
        .expect("an empty route should shape")
        .is_none()
    );
    // A mixed route serves part of one token's experts from one page: fewer
    // assignments than experts per token is still one decode token.
    let partial_rows = decode_token_activation_rows(
        &runtime,
        &activations(&runtime, &[1, 1, 1, INPUT_WIDTH as i32]),
        &route(3),
        EXPERTS_PER_TOKEN,
    )
    .expect("a partial route should shape")
    .expect("a partial route of one token is a decode token");
    assert_eq!(partial_rows.shape(), [1, INPUT_WIDTH as i32]);
}

#[test]
fn should_map_a_sorted_assignment_assertion_to_the_sorted_expert_order() {
    assert_eq!(
        expert_assignment_order(true),
        ExpertAssignmentOrder::SortedByExpert
    );
}

#[test]
fn should_map_an_unsorted_assignment_assertion_to_the_original_router_order() {
    assert_eq!(
        expert_assignment_order(false),
        ExpertAssignmentOrder::Original
    );
}

#[test]
fn should_promise_sorted_indices_only_for_the_sorted_expert_order() {
    // MLX exploits the sorted promise for contiguous expert reads, so the
    // mapping must never hand `SortedByExpert` to router-order IDs: the two
    // bool states are the whole domain and each maps to exactly one contract.
    assert!(ExpertAssignmentOrder::SortedByExpert.uses_sorted_indices());
    assert!(!ExpertAssignmentOrder::Original.uses_sorted_indices());
}
