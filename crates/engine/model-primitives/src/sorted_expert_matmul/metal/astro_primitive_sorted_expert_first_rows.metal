// Each expert's first row among rows sorted by expert: one thread per
// expert binary-searches the sorted ids [M] for the first row whose expert is
// not below its own, so first_rows [E] int32 bounds every expert's rows.
//
// @input ids uint32_t|int32_t may-be-small
// @input row_count int32_t always-small
// @output first_rows int32_t
// @template E int
// @compile-with E=256
// @compile-with E=4
// @contract direct_mlx::sorted_expert_matmul
// @probe sorted_expert_matmul

  const uint expert = thread_position_in_grid.x;
  if (expert >= uint(E)) return;
  int first = 0, count = row_count[0];
  while (count > 0) {
    const int step = count >> 1;
    if (uint(ids[first + step]) < expert) {
      first += step + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  first_rows[expert] = first;
