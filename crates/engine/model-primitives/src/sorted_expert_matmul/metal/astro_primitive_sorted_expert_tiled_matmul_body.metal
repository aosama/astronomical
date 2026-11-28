// Body shared by the tiled sorted-expert 4-bit matmuls, which declare its
// inputs and template integers in their signature blocks and define:
// - SOURCE_ROW(sorted_row): the row of X a sorted row is read from;
// - PAIRED: 1 when the 64-row weight tile pairs gate and up rows of the same
//   32 output columns and the epilogue writes MLX's compiled SwiGLU of them,
//   0 for a plain multiply;
// - WEIGHT_PACKED / WEIGHT_SCALES / WEIGHT_BIASES(reads_up): the expert stack
//   a weight row is read from.
//
// Every product is MLX's: weights dequantized with MLX's bits
// (dequantize_held_bytes), the 16x32x16 matrix-unit operations of
// astro_primitive_sorted_expert_matmul.h in the same K order, accumulators
// rounded to bf16 as MLX stores them. Rows past a tile's own are clamped to
// its last row and never stored, so they change no stored value.
//
// Requirements the source does not check: the M5 matrix units, K and N
// multiples of 64 (32 for a paired N), and inputs on 16-byte boundaries.

  using namespace astro_sorted_expert;
  using namespace astro_sorted_expert_tiled;
  constexpr int TM = 2;  // 16-row fragments of a simdgroup's 32 rows
  constexpr int COLUMNS_PER_TILE = PAIRED ? kTileColumns / 2 : kTileColumns;
  constexpr int COLUMN_BLOCKS = N / COLUMNS_PER_TILE;
  constexpr int STEPS = K / kStepDepth;
  constexpr int STEP_ELEMENTS = kTileColumns * kPaddedStep;
  threadgroup bfloat weight_steps[2 * STEP_ELEMENTS];

  uint4 tile;
  int column_block;
  if (!tile_of<GX>(tiles, tile_count[0], threadgroup_position_in_grid, COLUMN_BLOCKS, tile, column_block)) {
    return;
  }
  const int tile_first_row = int(tile.x);
  const int expert = int(tile.y);
  const int tile_rows = int(tile.z);
  const uint lane = thread_index_in_simdgroup;
  const uint simdgroup = simdgroup_index_in_threadgroup;
  const int thread_in_group = int(simdgroup) * 32 + int(lane);
  const int simdgroup_first_row = 32 * int(simdgroup / 2);
  const int simdgroup_column = 32 * int(simdgroup % 2);
  const int live_rows = clamp(tile_rows - simdgroup_first_row, 0, 32);
  const int live_fragments = (live_rows + 15) / 16;
  const short2 home = frag_home(ushort(lane));

  // Loader thread t dequantizes half (t % 2) of weight tile row t / 2.
  const bool loads = thread_in_group < kLoaderThreads;
  WeightStepLoader loader;
  if (loads) {
    const int tile_row = thread_in_group / 2;
    const bool reads_up = PAIRED && paired_row_reads_up(tile_row);
    const int operand_row = PAIRED ? paired_operand_row(tile_row) : tile_row;
    const long weight_row = long(expert) * N + long(column_block) * COLUMNS_PER_TILE + operand_row;
    loader.packed = (const device uchar*)WEIGHT_PACKED(reads_up) + weight_row * (K / 2) + 16 * (thread_in_group % 2);
    loader.scales = (const device bfloat*)WEIGHT_SCALES(reads_up) + weight_row * (K / 64);
    loader.biases = (const device bfloat*)WEIGHT_BIASES(reads_up) + weight_row * (K / 64);
    loader.slot_offset = tile_row * kPaddedStep + 32 * (thread_in_group % 2);
  }

  // Element offsets of this lane's two rows of each 16-row fragment, found
  // once for the whole K loop.
  long row_offset[TM][2];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < TM; m++) {
    _Pragma("clang loop unroll(full)")
    for (short row_half = 0; row_half < 2; row_half++) {
      const int local_row = min(16 * m + 8 * row_half + int(home.y), max(live_rows, 1) - 1);
      row_offset[m][row_half] = SOURCE_ROW(tile_first_row + simdgroup_first_row + local_row) * K + home.x;
    }
  }

  frag<float> acc[TM][2];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < TM; m++) {
    acc[m][0] = frag<float>(0);
    acc[m][1] = frag<float>(0);
  }
  if (loads) {
    loader.fetch(0);
    loader.store(weight_steps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device bfloat* rows = (const device bfloat*)X;
  for (int step = 0; step < STEPS; step++) {
    const bool more = step + 1 < STEPS;
    if (more && loads) {
      loader.fetch(step + 1);
    }
    const threadgroup bfloat* step_weights = weight_steps + (step & 1) * STEP_ELEMENTS;
    _Pragma("clang loop unroll(full)")
    for (short kk = 0; kk < kStepDepth; kk += 32) {
      frag<bfloat> b[2][2];
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 2; i++) {
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 2; j++) {
          frag_get_threadgroup(b[j][i], step_weights, kPaddedStep, simdgroup_column + 16 * i, kk + 16 * j, home);
        }
      }
      // A 16-row fragment with no live row is skipped: its rows are never
      // stored, and every stored row depends on its own fragment only.
      _Pragma("clang loop unroll(full)")
      for (short m = 0; m < TM; m++) {
        if (m < live_fragments) {
          frag<bfloat> a[2];
          _Pragma("clang loop unroll(full)")
          for (short j = 0; j < 2; j++) {
            const long column = long(step) * kStepDepth + kk + 16 * j;
            a[j] = frag<bfloat>(*(const device vec<bfloat, 4>*)(rows + row_offset[m][0] + column),
                                *(const device vec<bfloat, 4>*)(rows + row_offset[m][1] + column));
          }
          _Pragma("clang loop unroll(full)")
          for (short j = 0; j < 2; j++) {
            mma_16x32<false, true>(acc[m][0], acc[m][1], a[j], b[j][0], b[j][1]);
          }
        }
      }
    }
    if (more && loads) {
      loader.store(weight_steps + ((step + 1) & 1) * STEP_ELEMENTS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (live_rows > 0) {
    const long first_output_row = long(tile_first_row + simdgroup_first_row);
    if (PAIRED) {
      // acc[m][0] holds gate and acc[m][1] up of the same 16 output columns.
      device bfloat* out = (device bfloat*)Y + first_output_row * N + long(column_block) * COLUMNS_PER_TILE +
                           simdgroup_column / 2;
      _Pragma("clang loop unroll(full)")
      for (short m = 0; m < TM; m++) {
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; e++) {
          const int row = 16 * m + home.y + (e >> 2) * 8;
          if (row < live_rows) {
            out[long(row) * N + home.x + (e & 3)] =
                mlx_compiled_swiglu(bfloat(acc[m][0][e]), bfloat(acc[m][1][e]));
          }
        }
      }
    } else {
      device bfloat* out =
          (device bfloat*)Y + first_output_row * N + long(column_block) * COLUMNS_PER_TILE + simdgroup_column;
      _Pragma("clang loop unroll(full)")
      for (short m = 0; m < TM; m++) {
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 2; j++) {
          frag_put_rows_below(acc[m][j], out, N, 16 * m, 16 * j, home, live_rows);
        }
      }
    }
  }
