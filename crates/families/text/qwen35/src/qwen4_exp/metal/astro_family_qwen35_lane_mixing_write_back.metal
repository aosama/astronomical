// The write-back of a Flash-Next lane mixer in one launch:
// lanes + block_output * lane_weights, each lane of a row receiving its
// weight's share of the block's output, rounded to bfloat16 where the
// composed steps round.
//
// @input lanes T
// @input block_output T
// @input lane_weights T may-be-small
// @output written_lanes T
// @template T typename
// @template H int
// @template L int
// @compile-with T=bfloat16_t H=2048 L=4
// @compile-with T=bfloat16_t H=4 L=1
// @contract direct_mlx::fused_lane_mixing
// @probe none, the model compares it with the composed steps when it loads (enable_fused_lane_mixing)

    uint feature = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint lane = feature / uint(H);
    uint hidden = feature - lane * uint(H);
    const T injected = block_output[row * uint(H) + hidden]
        * lane_weights[row * uint(L) + lane];
    uint element = row * uint(L) * uint(H) + feature;
    written_lanes[element] = lanes[element] + injected;
