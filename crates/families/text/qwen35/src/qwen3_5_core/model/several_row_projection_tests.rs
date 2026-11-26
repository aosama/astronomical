use super::row_batched_kernel_serves;

#[test]
fn should_serve_a_dense_draft_trees_rows_with_the_row_batched_kernel() {
    assert!(row_batched_kernel_serves(16));
}

#[test]
fn should_serve_from_eight_rows_to_thirty_two_and_no_further() {
    assert!(row_batched_kernel_serves(8));
    assert!(row_batched_kernel_serves(32));
    assert!(!row_batched_kernel_serves(7));
    assert!(!row_batched_kernel_serves(33));
}

#[test]
fn should_leave_a_reply_token_a_small_tree_and_a_prompt_chunk_to_mlx() {
    for row_count in [0, 1, 4, 2_048, 4_096] {
        assert!(
            !row_batched_kernel_serves(row_count),
            "{row_count} rows stay on MLX's kernel"
        );
    }
}
