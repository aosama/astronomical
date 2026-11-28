// Helpers of the tiled sorted-expert 4-bit and 6-bit matmuls (the gate/up
// kernel with its SwiGLU epilogue and the plain kernel the down projection
// takes). They
// follow astro_primitive_sorted_expert_matmul.h, whose fragments and
// matrix-unit operation they reuse with its 4-bit dequantization; 6-bit
// weights dequantize here as MLX's own 6-bit `dequantize` does. Every
// product keeps MLX's bits.
//
// The schedule follows oMLX's tensor-unit sorted gather_qmm
// (omlx/patches/m5_gather_qmm_nax.py, Apache-2.0, Copyright the oMLX
// contributors), which on an M5 Pro ran one Ornith 35B-A3B prompt layer's
// routed experts in 6.28 ms against 7.42 ms for MLX's own sorted kernels
// (2,048 tokens, measured 2026-11-27):
// - a tile list built once per multiply (astro_primitive_sorted_expert_tile_list)
//   instead of every threadgroup searching the experts for its tile;
// - 64-column output tiles, each simdgroup owning a 32 x 32 block;
// - double-buffered weight steps: the packed words of the next 64-deep step
//   are read before this step's matrix-unit operations and dequantized into
//   the other buffer after them, so a step has one barrier, not two;
// - activation rows read straight from device memory through precomputed
//   per-lane row offsets, never staged;
// - optionally the row tile on the grid's x axis in groups of GX tiles, so a
//   row tile's column blocks run next to each other and share its rows in the
//   GPU's caches.

namespace astro_sorted_expert_tiled {
using namespace metal;
using namespace astro_sorted_expert;

// Output columns a threadgroup computes, in two 32-column simdgroups.
constant constexpr int kTileColumns = 64;
// K depth of one weight step: one 64-value quantization group a weight row.
constant constexpr int kStepDepth = 64;
// A staged weight row: 16 bytes of padding, as MLX pads its tiles.
constant constexpr int kPaddedStep = kStepDepth + 8;
// Threads that dequantize a step's 64 x 64 weight tile, 32 values each.
constant constexpr int kLoaderThreads = 128;

// The row tile and output column block of this threadgroup, false past the
// last tile. GX == 0: grid (column block, tile). GX > 0: tile t on grid x
// t % GX and (t / GX, column block) on y.
template <int GX>
inline bool tile_of(const device uint* tiles, uint tile_count, uint3 threadgroup_position, int column_blocks,
                    thread uint4& tile, thread int& column_block) {
  uint tile_index;
  uint block;
  if (GX > 0) {
    tile_index = (threadgroup_position.y / uint(column_blocks)) * uint(GX) + threadgroup_position.x;
    block = threadgroup_position.y % uint(column_blocks);
  } else {
    tile_index = threadgroup_position.y;
    block = threadgroup_position.x;
  }
  if (tile_index >= tile_count) {
    return false;
  }
  tile = *((const device uint4*)tiles + tile_index);
  column_block = int(block);
  return true;
}

// Gate/up pairing: in a 64-row weight tile, 16-row blocks alternate the gate
// rows and the up rows of the same 16 output columns, so a simdgroup's two
// 16-column fragments accumulate gate and up of the same columns in the same
// lanes. Tile row r reads the up stack when this is true...
inline bool paired_row_reads_up(int tile_row) {
  return ((tile_row >> 4) & 1) != 0;
}

// ...and this row of the tile's 32 output columns, in whichever stack.
inline int paired_operand_row(int tile_row) {
  return ((tile_row >> 5) << 4) + (tile_row & 15);
}

// The 16 packed bytes (32 weights of one group) a loader thread holds,
// dequantized exactly as dequantize_16_bytes does from device memory.
inline void dequantize_held_bytes(vec<uchar, 16> packed_bytes, bfloat scale, bfloat bias, threadgroup bfloat* out) {
  const float s = float(scale), b = float(bias), s_hi = s / 16.0f;
  _Pragma("clang loop unroll(full)")
  for (short quarter = 0; quarter < 4; quarter++) {
    vec<bfloat, 8> weights;
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 4; i++) {
      const uchar q = packed_bytes[4 * quarter + i];
      weights[2 * i] = bfloat(s * (q & 0x0f) + b);
      weights[2 * i + 1] = bfloat(s_hi * (q & 0xf0) + b);
    }
    *(threadgroup vec<bfloat, 8>*)(out + 8 * quarter) = weights;
  }
}

// The packed half group (32 weights) a loader thread holds between reading
// it and dequantizing it, for one weight bit width. Each specialization
// dequantizes exactly as MLX's `dequantize` in metal/kernels/quantized.h does
// for that width, so a staged weight is MLX's to the bit.
template <int BITS>
struct HeldHalfGroup;

// 4 bits: 16 bytes, one 128-bit load.
template <>
struct HeldHalfGroup<4> {
  vec<uchar, 16> packed_bytes;

  inline void fetch(const device uchar* half_group) thread {
    packed_bytes = *(const device vec<uchar, 16>*)half_group;
  }

  inline void dequantize_into(bfloat scale, bfloat bias, threadgroup bfloat* out) const thread {
    dequantize_held_bytes(packed_bytes, scale, bias, out);
  }
};

// 6 bits: 24 bytes, four weights in every three bytes as MLX packs them. A
// half group starts 24 bytes into its group, so its loads are 8 bytes wide.
template <>
struct HeldHalfGroup<6> {
  vec<uchar, 8> packed_words[3];

  inline void fetch(const device uchar* half_group) thread {
    _Pragma("clang loop unroll(full)")
    for (short word = 0; word < 3; word++) {
      packed_words[word] = *(const device vec<uchar, 8>*)(half_group + 8 * word);
    }
  }

  inline uchar byte_at(short byte_index) const thread {
    return packed_words[byte_index >> 3][byte_index & 7];
  }

  // MLX's 6-bit `dequantize`, value for value: (q * s + b) in float, then bf16.
  inline void dequantize_into(bfloat scale, bfloat bias, threadgroup bfloat* out) const thread {
    const float s = float(scale), b = float(bias);
    _Pragma("clang loop unroll(full)")
    for (short eighth = 0; eighth < 4; eighth++) {
      vec<bfloat, 8> weights;
      _Pragma("clang loop unroll(full)")
      for (short pack = 0; pack < 2; pack++) {
        const short first_byte = 6 * eighth + 3 * pack;
        const uchar w0 = byte_at(first_byte), w1 = byte_at(first_byte + 1), w2 = byte_at(first_byte + 2);
        weights[4 * pack] = bfloat((w0 & 0x3f) * s + b);
        weights[4 * pack + 1] = bfloat((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * s + b);
        weights[4 * pack + 2] = bfloat((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * s + b);
        weights[4 * pack + 3] = bfloat(((w2 >> 2) & 0x3f) * s + b);
      }
      *(threadgroup vec<bfloat, 8>*)(out + 8 * eighth) = weights;
    }
  }
};

// Bytes of one 64-value group of a weight row at BITS bits.
template <int BITS>
inline constexpr int group_bytes() {
  return 8 * BITS;
}

// One loader thread's share of every weight step: half (thread % 2) of tile
// row thread / 2, from the stack and row the caller chose.
template <int BITS>
struct WeightStepLoader {
  const device uchar* packed;
  const device bfloat* scales;
  const device bfloat* biases;
  // Where this thread's 32 weights sit in a staged step.
  int slot_offset;
  HeldHalfGroup<BITS> held;
  bfloat held_scale;
  bfloat held_bias;

  inline void fetch(int step) thread {
    held.fetch(packed + step * group_bytes<BITS>());
    held_scale = scales[step];
    held_bias = biases[step];
  }

  inline void store(threadgroup bfloat* weight_buffer) thread {
    held.dequantize_into(held_scale, held_bias, weight_buffer + slot_offset);
  }
};

// MLX's compiled silu(gate) * up on two bf16 projections, op for op: the
// sigmoid is MLX's Sigmoid::operator() (metal/kernels/unary_ops.h) evaluated
// on bfloat16, then two bfloat16 multiplies, as the compiled graph
// gate * sigmoid(gate) * up runs them. Any other formulation, even an equal
// one, rounds differently in bfloat16.
inline bfloat mlx_compiled_swiglu(bfloat gate, bfloat up) {
  auto sigmoid_of_negative_magnitude = 1 / (1 + metal::precise::exp(metal::abs(gate)));
  bfloat sigmoid_gate = (gate < 0) ? sigmoid_of_negative_magnitude : 1 - sigmoid_of_negative_magnitude;
  bfloat activated_gate = gate * sigmoid_gate;
  return activated_gate * up;
}

}  // namespace astro_sorted_expert_tiled
