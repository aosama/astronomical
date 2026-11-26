use super::most_tree_rows_an_exact_attention_call_takes;

#[test]
fn should_take_five_rows_at_six_query_heads_a_key_value_head_and_four_at_eight() {
    // The dense 27B: 24 query heads over 4 key-value heads.
    assert_eq!(most_tree_rows_an_exact_attention_call_takes(24, 4), 5);
    // The 35B-A3B: 16 query heads over 2 key-value heads.
    assert_eq!(most_tree_rows_an_exact_attention_call_takes(16, 2), 4);
}

#[test]
fn should_stop_at_the_vector_kernels_eight_rows_however_few_heads_share_a_key_value_head() {
    assert_eq!(most_tree_rows_an_exact_attention_call_takes(8, 8), 8);
    assert_eq!(most_tree_rows_an_exact_attention_call_takes(8, 4), 8);
    assert_eq!(most_tree_rows_an_exact_attention_call_takes(20, 4), 6);
}

#[test]
fn should_never_take_fewer_than_one_row_for_head_counts_no_kernel_serves() {
    assert_eq!(most_tree_rows_an_exact_attention_call_takes(64, 1), 1);
    assert_eq!(most_tree_rows_an_exact_attention_call_takes(0, 0), 8);
    assert_eq!(most_tree_rows_an_exact_attention_call_takes(16, 0), 2);
}
