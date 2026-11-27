// Body of the expert search kernel: each thread finds one expert's first row
// in the sorted ids by binary search. MLX's custom-kernel API wraps it in a
// kernel function with the inputs ids, row_count, the output first_rows and
// the template integer E.

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
