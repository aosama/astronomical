// The gate of a Flash-Next lane mixer from x / lanes in one launch: KIND 0
// is the lane read's mix, silu(x / lanes); KIND 1 the write-back weight,
// 2 * sigmoid(x / lanes). scaled is x * (1 / lanes) as the composed scalar
// multiply leaves it, and the two lines after it are MLX's sigmoid
// (Sigmoid::operator() in metal/kernels/unary_ops.h) verbatim, because any
// other form of it rounds differently in bfloat16. INVERSE_LANE_COUNT is
// 1 / lanes, a power of two the launcher states exactly as a float literal.
//
// @input gate_input T
// @output gate_output T
// @template T typename
// @template W int
// @template KIND int
// @define INVERSE_LANE_COUNT
// @compile-with T=bfloat16_t W=2048 KIND=0 INVERSE_LANE_COUNT=2.500000000e-01f
// @compile-with T=bfloat16_t W=4 KIND=1 INVERSE_LANE_COUNT=5.000000000e-01f
// @contract direct_mlx::fused_lane_mixing
// @probe none, the model compares it with the composed steps when it loads (enable_fused_lane_mixing)

    uint element = thread_position_in_grid.y * uint(W) + thread_position_in_grid.x;
    const T scaled = gate_input[element] * T(INVERSE_LANE_COUNT);
    auto y = 1 / (1 + metal::precise::exp(metal::abs(scaled)));
    const T sigmoid_value = (scaled < 0) ? y : 1 - y;
    if (KIND == 0) {
        gate_output[element] = scaled * sigmoid_value;
    } else {
        gate_output[element] = sigmoid_value * T(2.0f);
    }
