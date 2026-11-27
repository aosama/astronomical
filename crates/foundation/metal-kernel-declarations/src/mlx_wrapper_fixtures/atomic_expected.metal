[[kernel]] void custom_kernel_copy(
  device atomic<float>* Y [[buffer(0)]]) {
Y[0] = 0;
}
