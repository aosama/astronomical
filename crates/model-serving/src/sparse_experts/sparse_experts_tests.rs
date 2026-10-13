use astronomical_test_harness::offline_test;

use crate::SparseExpertError;

offline_test!(should_invert_a_complete_assignment_permutation, {
    // sorted_order[sorted_slot] = original_slot for the characterization example
    // selected_indices [5,0,4,1,3,2] argsort -> [1,3,5,4,2,0]
    let sorted_order = [1_u32, 3, 5, 4, 2, 0];
    let inverse_order = crate::invert_assignment_order(&sorted_order)
        .expect("a complete permutation should invert");
    assert_eq!(inverse_order, vec![5, 0, 4, 1, 3, 2]);
});

offline_test!(should_invert_an_empty_assignment_set, {
    let inverse_order =
        crate::invert_assignment_order(&[]).expect("an empty permutation should invert to empty");
    assert!(inverse_order.is_empty());
});

offline_test!(should_reject_a_duplicate_or_out_of_range_permutation, {
    assert_eq!(
        crate::invert_assignment_order(&[0, 0]),
        Err(SparseExpertError::InvalidAssignmentGeometry {
            description: "assignment permutation contains a duplicate slot",
        })
    );
    assert_eq!(
        crate::invert_assignment_order(&[0, 2]),
        Err(SparseExpertError::InvalidAssignmentGeometry {
            description: "assignment permutation contains an out-of-range slot",
        })
    );
});
