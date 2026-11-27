[[kernel]] void custom_kernel_copy(
  const device float* X [[buffer(0)]],
  const constant int* X_shape [[buffer(1)]],
  const constant int& X_ndim [[buffer(2)]],
  device float* Y [[buffer(3)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
Y[0] = X[0] * X_shape[0] * X_ndim; // thread_position_in_grid
}
