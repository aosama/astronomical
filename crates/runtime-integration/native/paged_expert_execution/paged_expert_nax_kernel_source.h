#pragma once

namespace astronomical::paged_expert_execution {

// Metal 4 matrix implementation for large sorted affine routes. This preserves
// the same page-table and quantization contract as the generic matrix fallback.
inline constexpr const char* kPagedGatherQuantizedMatrixNaxKernelSource = R"METAL(
template <typename T>
struct AstronomicalNaxExpertPageEntry {
  const device uint32_t* gate_weight;
  const device T* gate_scales;
  const device T* gate_biases;
  const device uint32_t* up_weight;
  const device T* up_scales;
  const device T* up_biases;
  const device uint32_t* down_weight;
  const device T* down_scales;
  const device T* down_biases;
  uint presence;
  uint generation;
};

template <typename T, int projection_index>
METAL_FUNC void astronomical_select_nax_expert_projection(
    const constant AstronomicalNaxExpertPageEntry<T>& expert_page,
    thread const device uint32_t*& packed_weight,
    thread const device T*& scales,
    thread const device T*& biases) {
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
    typename T,
    int group_size,
    int bits,
    int projection_index,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
[[kernel]] void astronomical_paged_gather_qmm_rhs_nax(
    const constant AstronomicalNaxExpertPageEntry<T>* page_table [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& M [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  x += y_row_long * K;
  y += y_row_long * N + y_col_long;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm =
      align_M ? SM : min(SM, short(max(0, (M - (y_row + tm)))));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);
  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;
  using AccumType = float;

  uint32_t expert_id;
  short assignment_start;
  uint32_t next_expert_id = indices[y_row];
  short next_assignment_start = 0;
  int assignment_position = 0;
  // A matrix tile can straddle several sorted expert runs. Load each expert's
  // weights separately and write only that run's row slice into the shared
  // output tile coordinates.
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

    const constant AstronomicalNaxExpertPageEntry<T>& expert_page =
        page_table[expert_id];
    const device uint32_t* packed_weight;
    const device T* scales;
    const device T* biases;
    astronomical_select_nax_expert_projection<T, projection_index>(
        expert_page, packed_weight, scales, biases);
    const device uint8_t* packed_weight_bytes =
        reinterpret_cast<const device uint8_t*>(packed_weight);
    packed_weight_bytes +=
        transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
    scales += transpose ? y_col_long * K_g : y_col / group_size;
    biases += transpose ? y_col_long * K_g : y_col / group_size;

    NAXTile<AccumType, TM, TN> output_tile;
    output_tile.clear();
    const device T* activation_tile = x + tm * K;
    thread loader_w_t weight_loader(
        packed_weight_bytes,
        scales,
        biases,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            weight_loader.load_unsafe();
          } else {
            weight_loader.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int inner_k = 0; inner_k < BK; inner_k += SK) {
            NAXTile<T, TM, TK> activation_matrix_tile;
            NAXTile<T, BR, BC> weight_matrix_tile;
            volatile int compiler_barrier;
            if constexpr (kAlignedM.value) {
              activation_matrix_tile.load(activation_tile + inner_k, K);
            } else {
              activation_matrix_tile.load_safe(
                  activation_tile + inner_k, K, short2(SK, sgp_sm));
            }
            if constexpr (transpose) {
              weight_matrix_tile.template load<T, BK_padded, 1>(
                  Ws + tn * BK_padded + inner_k);
            } else {
              weight_matrix_tile.template load<T, BN_padded, 1>(
                  Ws + tn + inner_k * BN_padded);
            }
            tile_matmad_nax(
                output_tile,
                activation_matrix_tile,
                metal::bool_constant<false>{},
                weight_matrix_tile,
                metal::bool_constant<transpose>{});
            (void)compiler_barrier;
          }
          activation_tile += BK;
          weight_loader.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          weight_loader.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          STEEL_PRAGMA_NO_UNROLL
          for (int inner_k = 0; inner_k < BK; inner_k += SK) {
            NAXTile<T, TM, TK> activation_matrix_tile;
            NAXTile<T, BR, BC> weight_matrix_tile;
            volatile int compiler_barrier;
            const short valid_inner_k =
                min(int(SK), max(0, (BK - inner_k)));
            activation_matrix_tile.load_safe(
                activation_tile + inner_k,
                K,
                short2(valid_inner_k, sgp_sm));
            if constexpr (transpose) {
              weight_matrix_tile.template load<T, BK_padded, 1>(
                  Ws + tn * BK_padded + inner_k);
            } else {
              weight_matrix_tile.template load<T, BN_padded, 1>(
                  Ws + tn + inner_k * BN_padded);
            }
            tile_matmad_nax(
                output_tile,
                activation_matrix_tile,
                metal::bool_constant<false>{},
                weight_matrix_tile,
                metal::bool_constant<transpose>{});
            (void)compiler_barrier;
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        const short minimum_assignment_row =
            min(int(sgp_sm), max(0, assignment_start - tm));
        const short maximum_assignment_row =
            min(int(sgp_sm), max(0, next_assignment_start - tm));
        if constexpr (kAlignedN.value) {
          if (minimum_assignment_row == 0 && maximum_assignment_row == SM) {
            output_tile.store(y + tm * N + tn, N);
          } else {
            output_tile.store_slice(
                y + tm * N + tn,
                N,
                short2(0, minimum_assignment_row),
                short2(SN, maximum_assignment_row));
          }
        } else {
          output_tile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, minimum_assignment_row),
              short2(sgp_sn, maximum_assignment_row));
        }
      });
    });
  }
}
)METAL";

}  // namespace astronomical::paged_expert_execution
