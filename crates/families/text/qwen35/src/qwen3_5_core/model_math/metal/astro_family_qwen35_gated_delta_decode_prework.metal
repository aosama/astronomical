// Fused Qwen 3.5 gated-delta decode prework, adapted from an upstream
// open-source Qwen 3.5 gated-delta kernel (see third-party/THIRD_PARTY_NOTICES
// for the provenance and license). One launch replaces the composed
// decode-side chain between the projections and the recurrence: the rolling
// convolution window, the depthwise conv1d, SiLU, the q/k/v split, the two
// ones-weight RMS normalizations, the two scalar scales, and the next
// rolling-state slice. One threadgroup of 32 lanes serves one (token row,
// logical head), the logical heads being the query heads, the key heads, then
// the value heads; each lane owns four contiguous channels.
//
// Numerics mirror the composed production path exactly, kernel by kernel: the
// conv accumulates four taps in float32 and rounds once like MLX's
// implicit-GEMM depthwise conv; the in-kernel SiLU uses MLX's own unary
// sigmoid formula (exp-of-abs) with MLX's `metal::precise::exp`: the fast
// `metal::exp` gives another result far from zero, which once left this
// kernel a bf16 step away from the composed path on strongly negative values
// and so a one-token step away from a draft tree's rows (issue #2599); the
// RMS reduction follows `rms_single_row` from MLX's Metal kernels (contiguous
// four-element lanes, one simd-group sum, and
// `rsqrt(sum_of_squares / head_dimension + epsilon)`) with this model's
// configured epsilon, followed by the ones-weight rounding cast and the
// separate scalar-scale rounding. The epsilon placement follows this
// repository's `rms_norm_without_weight` reference, not the upstream source's.
// The direct-MLX numerics contract proves bit-for-bit parity before any
// dispatch engages.
//
// RMS_NORM_EPSILON is the model's RMS epsilon, baked into the source as a
// define because Metal kernel template arguments carry no floats.
//
// @input mixed_queries_keys_values T
// @input convolution_state T
// @input convolution_weight T
// @input query_scale T scalar
// @input key_scale T scalar
// @output queries T
// @output keys T
// @output values T
// @output next_convolution_state T
// @template T typename
// @template HK int
// @template HV int
// @template DK int
// @template DV int
// @template NKEEP int
// @template C int
// @template S int
// @define RMS_NORM_EPSILON
// @compile-with T=bfloat16_t HK=16 HV=48 DK=128 DV=128 NKEEP=3 C=10240 S=1 RMS_NORM_EPSILON=9.999999975e-7f
// @compile-with T=bfloat16_t HK=1 HV=2 DK=128 DV=128 NKEEP=3 C=512 S=4 RMS_NORM_EPSILON=9.999999975e-7f
// @contract verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/gdn_decode_prework_numerics.rs
// @probe gdn_decode_prework

    uint lane = thread_position_in_threadgroup.x;
    uint batch_index = threadgroup_position_in_grid.y / uint(S);
    uint row = threadgroup_position_in_grid.y % uint(S);
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint k_head_base = uint(HK);
    constexpr uint v_head_base = 2 * uint(HK);
    bool is_q = logical_head < q_heads;
    bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
    uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));
    T activated[4];
    float sum_of_squares = 0.0f;
    for (uint element_index = 0; element_index < 4; ++element_index) {
        uint channel = channel_base + lane * 4 + element_index;
        float tap_accumulator = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            uint input_row = row + tap;
            const T window_value = input_row < uint(NKEEP)
                ? convolution_state[(batch_index * uint(NKEEP) + input_row) * uint(C) + channel]
                : mixed_queries_keys_values[(batch_index * uint(S) + input_row - uint(NKEEP)) * uint(C) + channel];
            tap_accumulator += float(window_value) * float(convolution_weight[channel * 4 + tap]);
        }
        const T convolved = T(tap_accumulator);
        T sigmoid_branch = T(1) / (T(1) + metal::precise::exp(metal::abs(convolved)));
        const T silu_value = convolved * ((convolved < T(0)) ? sigmoid_branch : T(1) - sigmoid_branch);
        activated[element_index] = silu_value;
        float activated_value = float(silu_value);
        sum_of_squares += activated_value * activated_value;
    }
    if (is_q || is_k) {
        sum_of_squares = simd_sum(sum_of_squares);
        float inverse_mean = metal::precise::rsqrt(sum_of_squares / float(DK) + RMS_NORM_EPSILON);
        const T scale = is_q ? query_scale : key_scale;
        uint out_base = ((batch_index * uint(S) + row) * uint(HK) + head) * uint(DK) + lane * 4;
        for (uint element_index = 0; element_index < 4; ++element_index) {
            const T normalized = T(1) * T(float(activated[element_index]) * inverse_mean);
            const T scaled = scale * normalized;
            if (is_q) {
                queries[out_base + element_index] = scaled;
            } else {
                keys[out_base + element_index] = scaled;
            }
        }
    } else {
        uint out_base = ((batch_index * uint(S) + row) * uint(HV) + head) * uint(DV) + lane * 4;
        for (uint element_index = 0; element_index < 4; ++element_index) {
            values[out_base + element_index] = activated[element_index];
        }
    }
    if (S < NKEEP && row == 0) {
        for (uint retained_row = 0; retained_row < uint(NKEEP) - uint(S); ++retained_row) {
            uint destination = (batch_index * uint(NKEEP) + retained_row) * uint(C)
                               + channel_base + lane * 4;
            uint source = (batch_index * uint(NKEEP) + retained_row + uint(S)) * uint(C)
                          + channel_base + lane * 4;
            for (uint element_index = 0; element_index < 4; ++element_index) {
                next_convolution_state[destination + element_index] = convolution_state[source + element_index];
            }
        }
    }
    if (row + uint(NKEEP) >= uint(S)) {
        uint state_row = row + uint(NKEEP) - uint(S);
        uint raw_base = (batch_index * uint(S) + row) * uint(C) + channel_base + lane * 4;
        uint state_base = (batch_index * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
        for (uint element_index = 0; element_index < 4; ++element_index) {
            next_convolution_state[state_base + element_index] = mixed_queries_keys_values[raw_base + element_index];
        }
    }
