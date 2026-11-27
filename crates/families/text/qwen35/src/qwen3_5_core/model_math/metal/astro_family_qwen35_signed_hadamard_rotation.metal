// The forward signed Hadamard rotation of a Prism pack in one launch: MLX's
// hadamard_n<T, N, 16, 4> schedule with the signed float32 load
// float(activation) * sign in place of its plain read, and the
// dtype-rounding store T(value * SIGNED_ROTATION_SCALE) in place of its
// float32 write. One threadgroup of N / 16 threads owns one Hadamard block of
// one row; the block's position within its activation row selects the slice
// of the sign vector it multiplies by.
//
// The butterfly is MLX's own hadamard_n kernel (radix-16 passes followed by
// the residual radix, working in float32 with the same thread-to-butterfly
// assignment), reproduced rather than paraphrased so every float32 addition
// happens in the same order; see third-party/THIRD_PARTY_NOTICES for the MLX
// attribution. The radix 16 below is the launcher's MAX_RADIX, which sizes
// the threadgroups with it; the read width 4 is MLX's device read width.
//
// SIGNED_ROTATION_SCALE is the normalization 1 / sqrt(block size), baked in
// as a float literal because Metal template arguments carry no floats; the
// literal round-trips the exact float32 the composed route hands MLX's
// kernel.
//
// @header astro_family_qwen35_signed_hadamard_rotation.h
// @input activations T
// @input signs float
// @output rotated T
// @template T typename
// @template LOG_BLOCK int
// @template BLOCKS_PER_ROW int
// @define SIGNED_ROTATION_SCALE
// @compile-with T=bfloat16_t LOG_BLOCK=7 BLOCKS_PER_ROW=40 SIGNED_ROTATION_SCALE=8.838834614e-02f
// @compile-with T=float16_t LOG_BLOCK=13 BLOCKS_PER_ROW=1 SIGNED_ROTATION_SCALE=1.104854345e-02f
// @compile-with T=float LOG_BLOCK=4 BLOCKS_PER_ROW=1 SIGNED_ROTATION_SCALE=2.500000000e-01f
// @contract none yet; its capability probe is the only bit-for-bit check against the composed route
// @probe prism_hadamard_signed_rotation

    constexpr int N = 1 << LOG_BLOCK;
    constexpr short max_radix = 16;
    constexpr short read_width = 4;
    constexpr short num_threads = N / max_radix;
    constexpr short logN = LOG_BLOCK;
    constexpr short logR = 4;
    constexpr short num_steps = logN / logR;
    constexpr short logFinal = logN % logR;
    constexpr short final_radix = 1 << logFinal;

    const uint block_row = threadgroup_position_in_grid.y;
    const int block_offset = int(block_row) * N;
    const int sign_offset = int(block_row % uint(BLOCKS_PER_ROW)) * N;
    const short thread_index = short(thread_position_in_threadgroup.x);

    threadgroup float block_values[N];

#pragma clang loop unroll(full)
    for (short read_step = 0; read_step < max_radix / read_width; read_step++) {
        short element_index = read_step * read_width * num_threads + thread_index * read_width;
#pragma clang loop unroll(full)
        for (short lane = 0; lane < read_width; lane++) {
            block_values[element_index + lane] =
                float(activations[block_offset + element_index + lane]) * signs[sign_offset + element_index + lane];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float butterfly_values[max_radix];
    short h = 1;

#pragma clang loop unroll(full)
    for (short pass = 0; pass < num_steps; pass++) {
        short k = thread_index & (h - 1);
        short j = ((thread_index - k) << logR) + k;
#pragma clang loop unroll(full)
        for (short radix_index = 0; radix_index < max_radix; radix_index++) {
            butterfly_values[radix_index] = block_values[j + h * radix_index];
        }
        astronomical_hadamard_radix<max_radix>(butterfly_values);
#pragma clang loop unroll(full)
        for (short radix_index = 0; radix_index < max_radix; radix_index++) {
            block_values[j + h * radix_index] = butterfly_values[radix_index];
        }
        h <<= logR;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (final_radix > 1) {
#pragma clang loop unroll(full)
        for (int butterfly = 0; butterfly < max_radix / final_radix; butterfly++) {
            short butterfly_index = thread_index + butterfly * num_threads;
            short k = butterfly_index & (h - 1);
            short j = ((butterfly_index - k) << logFinal) + k;
#pragma clang loop unroll(full)
            for (short radix_index = 0; radix_index < final_radix; radix_index++) {
                butterfly_values[radix_index] = block_values[j + h * radix_index];
            }
            astronomical_hadamard_radix<final_radix>(butterfly_values);
#pragma clang loop unroll(full)
            for (short radix_index = 0; radix_index < final_radix; radix_index++) {
                block_values[j + h * radix_index] = butterfly_values[radix_index];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (short read_step = 0; read_step < max_radix / read_width; read_step++) {
        short element_index = read_step * read_width * num_threads + thread_index * read_width;
#pragma clang loop unroll(full)
        for (short lane = 0; lane < read_width; lane++) {
            rotated[block_offset + element_index + lane] = T(block_values[element_index + lane] * SIGNED_ROTATION_SCALE);
        }
    }
