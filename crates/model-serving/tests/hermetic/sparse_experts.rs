use astronomical_model_serving::{
    SparseExpertError, gathered_indices_use_sorted_contract, invert_assignment_order,
};

#[test]
fn should_invert_a_complete_assignment_permutation() {
    // sorted_order[sorted_slot] = original_slot for the characterization example
    // selected_indices [5,0,4,1,3,2] argsort -> [1,3,5,4,2,0]
    let sorted_order = [1_u32, 3, 5, 4, 2, 0];
    let inverse_order =
        invert_assignment_order(&sorted_order).expect("a complete permutation should invert");
    assert_eq!(inverse_order, vec![5, 0, 4, 1, 3, 2]);
}

#[test]
fn should_invert_an_empty_assignment_set() {
    let inverse_order =
        invert_assignment_order(&[]).expect("an empty permutation should invert to empty");
    assert!(inverse_order.is_empty());
}

#[test]
fn should_reject_a_duplicate_or_out_of_range_permutation() {
    assert_eq!(
        invert_assignment_order(&[0, 0]),
        Err(SparseExpertError::InvalidAssignmentGeometry {
            description: "assignment permutation contains a duplicate slot",
        })
    );
    assert_eq!(
        invert_assignment_order(&[0, 2]),
        Err(SparseExpertError::InvalidAssignmentGeometry {
            description: "assignment permutation contains an out-of-range slot",
        })
    );
}

#[test]
fn should_allow_sorted_gather_only_when_the_caller_sorted_assignments() {
    assert!(gathered_indices_use_sorted_contract(true));
    assert!(!gathered_indices_use_sorted_contract(false));
}

#[test]
fn should_treat_xs_and_s_top_k_widths_as_named_rows() {
    let named_rows = [
        ("xs_routed", 8_u32, 256_u32, 512_u32),
        ("s_routed", 10, 256, 1_024),
    ];
    for (row_name, experts_per_token, expert_count, routed_width) in named_rows {
        assert!(
            experts_per_token < expert_count && routed_width > 0,
            "{row_name} is named evidence, not a default"
        );
    }
}
