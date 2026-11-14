use astronomical_model_serving::persistent_context_restore_workspace_bytes;

#[test]
fn should_size_the_restore_workspace_by_one_block_for_incremental_restore() {
    // The restore loop absorbs one block at a time into a preallocated
    // destination, so the temporary workspace is a single source block, not
    // the whole restored prefix (71_680 tokens would be 1_468_006_400
    // bytes). The destination is already part of the stored state.
    let context_memory_reservation_bytes_per_token = 20_480;
    let restore_block_token_count = 128;

    let restore_workspace_bytes = persistent_context_restore_workspace_bytes(
        context_memory_reservation_bytes_per_token,
        restore_block_token_count,
    )
    .expect("the one-block restore workspace should fit usize");

    assert_eq!(restore_workspace_bytes, 2_621_440);
}
