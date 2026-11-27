// Stages one 32-key tile of a key or value slab in threadgroup memory: the
// context's own keys, or, past the context, source_row's ancestors in depth
// order; keys that row does not see are zero.

// Pointer types are template parameters: MLX may place a small input in the
// constant address space and a large one in device memory.
template <typename SlabPointer, typename LengthsPointer, typename ChainsPointer>
inline void stage_tile(threadgroup bfloat* staged_tile, SlabPointer slab,
    size_t head_offset, int tile_start, int context, bool shared_tile, int source_row,
    LengthsPointer chain_lengths, ChainsPointer row_chains, int maximum_chain,
    int thread_index, int thread_count) {
  constexpr int head_size = 256;
  constexpr int tile_keys = 32;
  for (int element = thread_index; element < tile_keys * head_size / 8; element += thread_count) {
    const int tile_key = element / (head_size / 8);
    const int column = (element % (head_size / 8)) * 8;
    const int logical_key = tile_start + tile_key;
    int physical_key = -1;
    if (logical_key < context) {
      physical_key = logical_key;
    } else if (!shared_tile && logical_key < context + chain_lengths[source_row]) {
      physical_key = context + row_chains[source_row * maximum_chain + (logical_key - context)];
    }
    vec<bfloat, 8> values = vec<bfloat, 8>(bfloat(0.0f));
    if (physical_key >= 0) {
      values = *(const device vec<bfloat, 8>*)(slab + head_offset + (size_t)physical_key * head_size + column);
    }
    *(threadgroup vec<bfloat, 8>*)(staged_tile + tile_key * head_size + column) = values;
  }
}
