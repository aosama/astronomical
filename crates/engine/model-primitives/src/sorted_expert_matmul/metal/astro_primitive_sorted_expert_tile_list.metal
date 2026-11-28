// The row tiles of a sorted-expert multiply: rows sorted by expert ids [M]
// are cut into tiles of at most BM rows of one expert, written expert by
// expert as (first row, expert, rows, 0) to tiles [max_tiles * 4], with
// their count in tile_count [1]. bounds [2] holds M and max_tiles. One
// threadgroup does it all: every expert's run of rows is found in parallel
// over the rows, then a scan of the per-expert tile counts gives each expert
// its first tile. Ids that are not sorted (which callers never pass) cannot
// write past max_tiles.
//
// @input ids uint32_t|int32_t may-be-small
// @input bounds int32_t always-small
// @output tiles uint32_t
// @output tile_count uint32_t
// @template BM int
// @template E int
// @compile-with BM=96 E=256
// @compile-with BM=64 E=4
// @contract direct_mlx::sorted_expert_tiled_matmul
// @probe sorted_expert_tiled_matmul

  threadgroup uint run_start[E];
  threadgroup uint run_end[E];
  threadgroup uint simdgroup_tile_totals[32];
  const int thread_id = int(thread_index_in_threadgroup);
  const int thread_count = int(threads_per_threadgroup.x);
  const uint simdgroup = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const int M = bounds[0];
  const uint max_tiles = uint(bounds[1]);

  for (int expert = thread_id; expert < E; expert += thread_count) {
    run_start[expert] = 0;
    run_end[expert] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int row = thread_id; row < M; row += thread_count) {
    const uint expert = uint(ids[row]);
    if (expert < uint(E)) {
      if (row == 0 || uint(ids[row - 1]) != expert) {
        run_start[expert] = uint(row);
      }
      if (row == M - 1 || uint(ids[row + 1]) != expert) {
        run_end[expert] = uint(row + 1);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint simdgroup_count = uint(thread_count + 31) / 32;
  uint tiles_before = 0;
  for (int first_expert = 0; first_expert < E; first_expert += thread_count) {
    const int expert = first_expert + thread_id;
    uint start = 0;
    uint row_total = 0;
    if (expert < E) {
      start = run_start[expert];
      row_total = run_end[expert] > start ? run_end[expert] - start : 0;
    }
    const uint expert_tiles = (row_total + uint(BM) - 1) / uint(BM);
    const uint before_in_simdgroup = simd_prefix_exclusive_sum(expert_tiles);
    const uint simdgroup_total = simd_sum(expert_tiles);
    if (lane == 0) {
      simdgroup_tile_totals[simdgroup] = simdgroup_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before_this_simdgroup = 0;
    uint pass_total = 0;
    for (uint other = 0; other < simdgroup_count; other++) {
      const uint other_total = simdgroup_tile_totals[other];
      before_this_simdgroup += other < simdgroup ? other_total : 0;
      pass_total += other_total;
    }
    const uint first_tile = tiles_before + before_this_simdgroup + before_in_simdgroup;
    for (uint tile = 0; tile < expert_tiles && first_tile + tile < max_tiles; tile++) {
      const uint tile_row = start + tile * uint(BM);
      *((device uint4*)tiles + first_tile + tile) =
          uint4(tile_row, uint(expert), min(uint(BM), start + row_total - tile_row), 0);
    }
    tiles_before += pass_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_id == 0) {
    tile_count[0] = min(tiles_before, max_tiles);
  }
