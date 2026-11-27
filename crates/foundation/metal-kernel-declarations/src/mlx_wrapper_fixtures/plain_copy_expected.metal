[[kernel]] void custom_kernel_copy(
  const device float* X [[buffer(0)]],
  device float* Y [[buffer(1)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  Y[thread_position_in_grid.x] = X[thread_position_in_grid.x];
}
