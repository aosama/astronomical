[[kernel]] void custom_kernel_copy(
  const constant int32_t* count [[buffer(0)]],
  const constant float& scale [[buffer(1)]],
  device float* Y [[buffer(2)]]) {
Y[0] = scale * count[0];
}
