// The per-group load one lane of the register-ordered projection holds, and
// its fetch: the codes, two scale-and-bias quads and two row sums of a group,
// loaded a group ahead of the product that uses them.

// One lane's codes, its two scale-and-bias quads and its two row sums for one group.
struct RegisterGroupLoad { uint4 low_codes, high_codes, low_pairs, high_pairs; float low_sum, high_sum; };

inline RegisterGroupLoad fetch_register_group(
    const device uint4* lane_codes, const device uint4* pair_quads, const device float* row_sums,
    int group, int output_width, int column, int row, int row_stride) {
  return {lane_codes[(size_t)group * 64], lane_codes[(size_t)group * 64 + 32],
          pair_quads[((size_t)group * output_width + column) / 4],
          pair_quads[((size_t)group * output_width + column + 16) / 4],
          row_sums[group * row_stride + row], row_sums[group * row_stride + row + 8]};
}
