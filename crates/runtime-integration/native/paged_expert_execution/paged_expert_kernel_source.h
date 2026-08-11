#pragma once

namespace astronomical::paged_expert_execution {

// Runtime-compiled Metal source for affine expert pages. The entry layout must
// stay byte-identical to the 80-byte host page-table entry.
inline constexpr const char* kPagedGatherQuantizedMatrixKernelSource = R"METAL(
template <typename ScaleT, typename BiasT>
struct AstronomicalExpertPageEntry {
  const device uint32_t* gate_weight;
  const device ScaleT* gate_scales;
  const device BiasT* gate_biases;
  const device uint32_t* up_weight;
  const device ScaleT* up_scales;
  const device BiasT* up_biases;
  const device uint32_t* down_weight;
  const device ScaleT* down_scales;
  const device BiasT* down_biases;
  uint presence;
  uint generation;
};

template <typename ScaleT, typename BiasT, int projection_index>
METAL_FUNC void astronomical_select_expert_projection(
    const constant AstronomicalExpertPageEntry<ScaleT, BiasT>& expert_page,
    thread const device uint32_t*& packed_weight,
    thread const device ScaleT*& scales,
    thread const device BiasT*& biases) {
  if constexpr (projection_index == 0) {
    packed_weight = expert_page.gate_weight;
    scales = expert_page.gate_scales;
    biases = expert_page.gate_biases;
  } else if constexpr (projection_index == 1) {
    packed_weight = expert_page.up_weight;
    scales = expert_page.up_scales;
    biases = expert_page.up_biases;
  } else {
    packed_weight = expert_page.down_weight;
    scales = expert_page.down_scales;
    biases = expert_page.down_biases;
  }
}

template <
    typename ComputeT,
    typename ScaleT,
    typename BiasT,
    int group_size,
    int bits,
    int projection_index,
    bool fast>
[[kernel]] void astronomical_paged_gather_qmv(
    const constant AstronomicalExpertPageEntry<ScaleT, BiasT>* page_table [[buffer(0)]],
    const device ComputeT* x [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device ComputeT* y [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& assignments_per_activation_row [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  const uint32_t expert_id = indices[tid.z];
  const constant AstronomicalExpertPageEntry<ScaleT, BiasT>& expert_page =
      page_table[expert_id];
  const device uint32_t* packed_weight;
  const device ScaleT* scales;
  const device BiasT* biases;
  astronomical_select_expert_projection<ScaleT, BiasT, projection_index>(
      expert_page, packed_weight, scales, biases);

  x += size_t(tid.z / assignments_per_activation_row) * K;
  y += size_t(tid.z) * N;
  const uint3 projection_tid = uint3(tid.x, tid.y, 0);
  if constexpr (
      metal::is_same_v<ComputeT, ScaleT> &&
      metal::is_same_v<ComputeT, BiasT>) {
    if constexpr (fast) {
      qmv_fast_impl<ComputeT, group_size, bits>(
          packed_weight,
          scales,
          biases,
          x,
          y,
          K,
          N,
          projection_tid,
          simd_group_id,
          simd_lane_id);
    } else {
      qmv_impl<ComputeT, group_size, bits>(
          packed_weight,
          scales,
          biases,
          x,
          y,
          K,
          N,
          projection_tid,
          simd_group_id,
          simd_lane_id);
    }
  } else {
    if constexpr (fast) {
      astronomical_mixed_dtype_qmv_fast_impl<
          ComputeT, ScaleT, BiasT, group_size, bits>(
          packed_weight,
          scales,
          biases,
          x,
          y,
          K,
          N,
          projection_tid,
          simd_group_id,
          simd_lane_id);
    } else {
      astronomical_mixed_dtype_qmv_impl<
          ComputeT, ScaleT, BiasT, group_size, bits>(
          packed_weight,
          scales,
          biases,
          x,
          y,
          K,
          N,
          projection_tid,
          simd_group_id,
          simd_lane_id);
    }
  }
}

template <
    typename ComputeT,
    typename ScaleT,
    typename BiasT,
    int group_size,
    int bits,
    int projection_index,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
[[kernel]] void astronomical_paged_gather_qmm_rhs(
    const constant AstronomicalExpertPageEntry<ScaleT, BiasT>* page_table [[buffer(0)]],
    const device ComputeT* x [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device ComputeT* y [[buffer(3)]],
    const constant int& M [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(ComputeT));
  constexpr int BN_padded = (BN + 16 / sizeof(ComputeT));

  using mma_t = mlx::steel::BlockMMA<
      ComputeT, ComputeT, BM, BN, BK, WM, WN, false, transpose, BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<
          ComputeT, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = AstronomicalQuantizedBlockLoader<
      ComputeT, ScaleT, BiasT, transpose ? BN : BK, transpose ? BK : BN,
      transpose ? BK_padded : BN_padded, transpose,
      WM * WN * SIMD_SIZE, group_size, bits>;

  threadgroup ComputeT Xs[BM * BK_padded];
  threadgroup ComputeT Ws[transpose ? BN * BK_padded : BK * BN_padded];

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  x += y_row_long * K;
  y += y_row_long * N + y_col_long;

  uint32_t expert_id;
  short assignment_start;
  uint32_t next_expert_id = indices[y_row];
  short next_assignment_start = 0;
  int assignment_position = 0;
  // Sorted assignments form contiguous runs by expert. Recompute the tile for
  // each run and store only rows belonging to that expert, avoiding a stacked
  // expert tensor while retaining MLX's quantized matrix tile arithmetic.
  while (assignment_position < tgp_bm) {
    assignment_position++;
    assignment_start = next_assignment_start;
    expert_id = next_expert_id;
    next_assignment_start = tgp_bm;
    for (; assignment_position < tgp_bm; assignment_position++) {
      if (indices[y_row + assignment_position] != expert_id) {
        next_assignment_start = assignment_position;
        next_expert_id = indices[y_row + assignment_position];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    const constant AstronomicalExpertPageEntry<ScaleT, BiasT>& expert_page =
        page_table[expert_id];
    const device uint32_t* packed_weight;
    const device ScaleT* scales;
    const device BiasT* biases;
    astronomical_select_expert_projection<ScaleT, BiasT, projection_index>(
        expert_page, packed_weight, scales, biases);

    const device uint8_t* packed_weight_bytes =
        reinterpret_cast<const device uint8_t*>(packed_weight);
    packed_weight_bytes +=
        transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
    scales += transpose ? y_col_long * K_g : y_col / group_size;
    biases += transpose ? y_col_long * K_g : y_col / group_size;

    thread mma_t matrix_operation(simd_group_id, simd_lane_id);
    thread loader_x_t activation_loader(
        x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t weight_loader(
        packed_weight_bytes, scales, biases, transpose ? K : N, Ws,
        simd_group_id, simd_lane_id);

    if (align_M && align_N) {
      gemm_loop_aligned(
          Xs, Ws, matrix_operation, activation_loader, weight_loader, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(
            Xs, Ws, matrix_operation, activation_loader, weight_loader,
            tile_x, tile_w);
      }
      if (next_assignment_start - assignment_start == BM) {
        matrix_operation.store_result(y, N);
      } else {
        matrix_operation.store_result_slice(
            y, N, short2(0, assignment_start),
            short2(BN, next_assignment_start));
      }
    } else if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
      gemm_loop_aligned(
          Xs, Ws, matrix_operation, activation_loader, weight_loader, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(
            Xs, Ws, matrix_operation, activation_loader, weight_loader,
            tile_x, tile_w);
      }
      matrix_operation.store_result_slice(
          y, N, short2(0, assignment_start),
          short2(tgp_bn, next_assignment_start));
    } else if (align_N || tgp_bn == BN) {
      gemm_loop_unaligned<false, true, transpose>(
          Xs, Ws, matrix_operation, activation_loader, weight_loader, K_it,
          tgp_bm, tgp_bn, BK);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(
            Xs, Ws, matrix_operation, activation_loader, weight_loader,
            tile_x, tile_w);
      }
      matrix_operation.store_result_slice(
          y, N, short2(0, assignment_start),
          short2(BN, next_assignment_start));
    } else if (align_M || tgp_bm == BM) {
      gemm_loop_unaligned<true, false, transpose>(
          Xs, Ws, matrix_operation, activation_loader, weight_loader, K_it,
          tgp_bm, tgp_bn, BK);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(
            Xs, Ws, matrix_operation, activation_loader, weight_loader,
            tile_x, tile_w);
      }
      matrix_operation.store_result_slice(
          y, N, short2(0, assignment_start),
          short2(tgp_bn, next_assignment_start));
    } else {
      gemm_loop_unaligned<false, false, transpose>(
          Xs, Ws, matrix_operation, activation_loader, weight_loader, K_it,
          tgp_bm, tgp_bn, BK);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(
            Xs, Ws, matrix_operation, activation_loader, weight_loader,
            tile_x, tile_w);
      }
      matrix_operation.store_result_slice(
          y, N, short2(0, assignment_start),
          short2(tgp_bn, next_assignment_start));
    }
  }
}
)METAL";

}  // namespace astronomical::paged_expert_execution
