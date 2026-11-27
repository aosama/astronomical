// Body of the sorted-expert 4-bit matmul kernel. MLX's custom-kernel API
// wraps it in a kernel function whose parameters are the inputs and outputs
// declared in sorted_expert_quantized_matmul.rs (X, Wq, S, B, first_rows,
// [order,] row_count -> Y) and the template integers N, K, E, BM, WN,
// STAGE_ROWS, TILE_SCHEDULE [, TOPK]; one of the two preamble files ahead of
// it defines SOURCE_ROW. The inputs, outputs and template integers are
// documented in kernel_source.rs.

  using namespace astro_sorted_expert;
  constexpr int BN = 32 * WN;
  constexpr int THREADS = 64 * WN;
  constexpr int SM = BM / 2;   // rows of a simdgroup
  constexpr int TM = SM / 16;  // its 16-row fragments
  constexpr int PAD = 64 + 8;  // 16 bytes of padding a row, as MLX pads its tiles
  constexpr int STAGED_CHUNKS = BM * 8 / THREADS;  // 8-value chunks a thread stages
  threadgroup bfloat weight_tile[BN * PAD];
  threadgroup bfloat row_tile[STAGE_ROWS ? BM * PAD : 8];

  const uint lane = thread_index_in_simdgroup;
  const uint simdgroup = simdgroup_index_in_threadgroup;
  const int thread_in_group = int(simdgroup) * 32 + int(lane);
  const int M = row_count[0];
  int expert, first_row, end_row;
  if (TILE_SCHEDULE) {
    int tile_rows;
    if (!expert_tile<BM>(first_rows, E, M, int(threadgroup_position_in_grid.y), lane, expert, first_row,
                         tile_rows)) {
      return;
    }
    end_row = first_row + tile_rows;
  } else {
    expert = int(threadgroup_position_in_grid.y);
    first_row = first_rows[expert];
    end_row = expert + 1 < E ? first_rows[expert + 1] : M;
  }
  const int col = int(threadgroup_position_in_grid.x) * BN;
  const int tm = SM * int(simdgroup / WN), tn = 32 * int(simdgroup % WN);
  const short2 home = frag_home(ushort(lane));

  // Weight staging: thread t dequantizes half (t % 2) of weight row t / 2.
  const long weight_row = long(expert) * N + col + thread_in_group / 2;
  const device uchar* expert_packed = (const device uchar*)Wq + weight_row * (K / 2) + 16 * (thread_in_group % 2);
  const device bfloat* expert_scale = (const device bfloat*)S + weight_row * (K / 64);
  const device bfloat* expert_bias = (const device bfloat*)B + weight_row * (K / 64);
  threadgroup bfloat* weight_slot = weight_tile + (thread_in_group / 2) * PAD + 32 * (thread_in_group % 2);

  for (int row = first_row; row < end_row; row += BM) {
    const int rows = min(BM, end_row - row);
    const int live = clamp(rows - tm, 0, SM);
    const device bfloat* simdgroup_rows = (const device bfloat*)X + long(row + tm) * K;
    // Whether all SM of the simdgroup's rows exist in X (MLX's own test):
    // then rows are read without a bound; only the matrix's end needs one.
    const bool rows_inside_matrix = row + tm + SM <= M;
    const device uchar* packed = expert_packed;
    const device bfloat* scale = expert_scale;
    const device bfloat* bias = expert_bias;
    frag<float> acc[TM][2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < TM; i++) {
      acc[i][0] = frag<float>(0);
      acc[i][1] = frag<float>(0);
    }

    for (int k = 0; k < K; k += 64) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      dequantize_16_bytes(packed, *scale, *bias, weight_slot);
      if (STAGE_ROWS) {
        // Chunk c is 8 values of pass row c / 8; rows past the expert's stage
        // zeros, which add nothing and are never stored.
        _Pragma("clang loop unroll(full)")
        for (short chunk_pass = 0; chunk_pass < STAGED_CHUNKS; chunk_pass++) {
          const int chunk = thread_in_group + THREADS * chunk_pass;
          const int staged_row = chunk / 8, staged_column = 8 * (chunk % 8);
          *(threadgroup vec<bfloat, 8>*)(row_tile + staged_row * PAD + staged_column) = staged_row < rows
              ? *(const device vec<bfloat, 8>*)((const device bfloat*)X + SOURCE_ROW(row + staged_row) * K + k + staged_column)
              : vec<bfloat, 8>(0);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(disable)
      for (int kk = 0; kk < 64; kk += 32) {
        if (live > 0) {
          frag<bfloat> b[2][2];
          _Pragma("clang loop unroll(full)")
          for (short i = 0; i < 2; i++) {
            _Pragma("clang loop unroll(full)")
            for (short j = 0; j < 2; j++) {
              frag_get_threadgroup(b[j][i], weight_tile, PAD, tn + 16 * i, kk + 16 * j, home);
            }
          }
          // A 16-row fragment with no live row is skipped: its rows are never
          // stored, and every stored row depends on its own fragment only.
          _Pragma("clang loop unroll(full)")
          for (short m = 0; m < TM; m++) {
            if (live > 16 * m) {
              frag<bfloat> a[2];
              _Pragma("clang loop unroll(full)")
              for (short j = 0; j < 2; j++) {
                if (STAGE_ROWS) {
                  frag_get_threadgroup(a[j], row_tile, PAD, tm + 16 * m, kk + 16 * j, home);
                } else if (rows_inside_matrix) {
                  frag_get_device(a[j], simdgroup_rows, K, 16 * m, k + kk + 16 * j, home);
                } else {
                  frag_get_live_rows(a[j], simdgroup_rows, K, 16 * m, k + kk + 16 * j, home, live);
                }
              }
              _Pragma("clang loop unroll(full)")
              for (short j = 0; j < 2; j++) {
                mma_16x32<false, true>(acc[m][0], acc[m][1], a[j], b[j][0], b[j][1]);
              }
            }
          }
        }
      }
      packed += 32;
      scale++;
      bias++;
    }

    device bfloat* out = (device bfloat*)Y + long(row + tm) * N + col + tn;
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < TM; i++) {
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 2; j++) {
        frag_put_rows_below(acc[i][j], out, N, 16 * i, 16 * j, home, live);
      }
    }
  }
