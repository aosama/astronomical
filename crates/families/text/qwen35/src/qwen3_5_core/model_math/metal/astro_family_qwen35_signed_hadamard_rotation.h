// The thread-local radix butterfly of MLX's hadamard.h (radix_func). The
// butterfly index names (h for the half-span, j and k for the butterfly pair
// positions) are kept as MLX writes them so the two sources can be audited
// side by side; the arithmetic order is what parity relies on.

template <short R>
METAL_FUNC void astronomical_hadamard_radix(thread float* butterfly_values) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  // Full unrolling (as MLX's STEEL_PRAGMA_UNROLL does) keeps every index
  // constant so the butterfly array stays in registers; a rolled loop
  // indexes thread memory dynamically and runs several times slower.
#pragma clang loop unroll(full)
  for (short stage = 0; stage < logR; stage++) {
#pragma clang loop unroll(full)
    for (short pair_index = 0; pair_index < R / 2; pair_index++) {
      short k = pair_index & (h - 1);
      short j = ((pair_index - k) << 1) + k;
      float upper = butterfly_values[j];
      float lower = butterfly_values[j + h];
      butterfly_values[j] = upper + lower;
      butterfly_values[j + h] = upper - lower;
    }
    h <<= 1;
  }
}
