template <int N, bool STAGE, typename T>
[[kernel]] void custom_kernel_copy(
  const device bfloat16_t* X [[buffer(0)]],
  device bfloat16_t* Y [[buffer(1)]]) {
Y[0] = X[0];
}

template [[host_name("custom_kernel_copy")]] [[kernel]] decltype(custom_kernel_copy<-4, 1, bfloat16_t>) custom_kernel_copy<-4, 1, bfloat16_t>;
