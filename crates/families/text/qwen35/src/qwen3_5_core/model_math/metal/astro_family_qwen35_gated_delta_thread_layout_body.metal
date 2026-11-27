// The thread layout the gated-delta tree and replay kernels share: 16 value
// rows a threadgroup, each spread over 8 lanes, with the value head on the
// grid's y axis. Written once, as the opening of both kernels, so the two
// cannot drift apart (issue #2340); the bit-for-bit tree and replay tests
// prove the assembled kernels unchanged.

    constexpr int value_row_block_size = 16;
    auto thread_index = thread_position_in_threadgroup.x;
    auto value_head_index = threadgroup_position_in_grid.y;
    auto key_head_index = value_head_index / (Hv / Hk);
    auto value_row = threadgroup_position_in_grid.x * value_row_block_size + thread_index / 8;
    auto key_segment_index = thread_index % 8;
