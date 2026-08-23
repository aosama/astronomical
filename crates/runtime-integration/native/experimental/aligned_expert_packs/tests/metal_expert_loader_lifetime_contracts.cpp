// Keeps repetition and allocation-lifetime contracts separate from the native
// journey entry point so each source retains one cohesive test responsibility.

#include "metal_expert_loader_test_support.h"

#include <filesystem>
#include <fstream>
#include <vector>

namespace {

constexpr size_t kLifetimePackByteCount = 64 * 1024;

void write_lifetime_pack(const std::filesystem::path& source_pack_path) {
  std::ofstream source_pack_file(
      source_pack_path, std::ios::binary | std::ios::trunc);
  std::vector<uint8_t> source_bytes(kLifetimePackByteCount, 0x39);
  source_pack_file.write(
      reinterpret_cast<const char*>(source_bytes.data()),
      static_cast<std::streamsize>(source_bytes.size()));
  require_condition(
      source_pack_file.good(), "could not write the lifetime-contract pack");
}

size_t active_memory_bytes() {
  size_t active_memory_byte_count = 0;
  require_mlx_success(
      mlx_get_active_memory(&active_memory_byte_count),
      "read native MLX active memory");
  return active_memory_byte_count;
}

}  // namespace

void should_release_native_transaction_allocations_between_repetitions() {
  std::cout
      << "[native-metal-expert-loader] status=progress test=repeated_transaction_release phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path =
      temporary_directory.path() / "repeated.expert-pack";
  write_lifetime_pack(source_pack_path);
  NativeStreamOwner gpu_stream;
  const std::vector<int> output_shape = {
      static_cast<int>(kLifetimePackByteCount / sizeof(uint32_t))};
  const std::vector<astronomical_metal_expert_loader_output_tensor>
      output_tensors = {{
          output_shape.data(),
          static_cast<int>(output_shape.size()),
          MLX_UINT32,
      }};
  const std::vector<astronomical_metal_expert_loader_load_range> load_ranges =
      {{0, 0, 0, kLifetimePackByteCount}};
  size_t baseline_active_memory_bytes = 0;
  for (size_t repetition_index = 0; repetition_index < 4;
       ++repetition_index) {
    CompleteTransactionOwner transaction(
        source_pack_path,
        output_tensors,
        load_ranges,
        gpu_stream.get(),
        "repeated_transaction_release");
    transaction.consume_all_outputs_on_gpu();
    transaction.wait_for_io_completion();
    transaction.release();
    const auto active_memory_after_release_bytes = active_memory_bytes();
    if (repetition_index == 0) {
      baseline_active_memory_bytes = active_memory_after_release_bytes;
    } else {
      require_condition(
          active_memory_after_release_bytes == baseline_active_memory_bytes,
          "repeated native transactions retained active MLX allocations");
    }
    std::cout
        << "[native-metal-expert-loader] status=progress test=repeated_transaction_release completed_repetitions="
        << repetition_index + 1 << "/4" << std::endl;
  }
  std::cout
      << "[native-metal-expert-loader] status=success test=repeated_transaction_release"
      << std::endl;
}

void should_repeat_large_eight_range_native_transactions() {
  std::cout
      << "[native-metal-expert-loader] status=progress test=repeated_large_eight_range phase=setup"
      << std::endl;
  constexpr size_t range_byte_count = 3'342'336;
  constexpr size_t range_count = 8;
  constexpr size_t source_byte_count = range_byte_count * range_count;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path =
      temporary_directory.path() / "large-eight-range.expert-pack";
  std::vector<uint8_t> source_bytes(source_byte_count);
  for (size_t byte_index = 0; byte_index < source_bytes.size(); ++byte_index) {
    source_bytes[byte_index] = static_cast<uint8_t>((byte_index * 17U) % 251U);
  }
  std::ofstream source_pack_file(
      source_pack_path, std::ios::binary | std::ios::trunc);
  source_pack_file.write(
      reinterpret_cast<const char*>(source_bytes.data()),
      static_cast<std::streamsize>(source_bytes.size()));
  source_pack_file.close();
  require_condition(
      source_pack_file.good(), "could not write the large native test pack");
  NativeStreamOwner gpu_stream;
  const std::vector<int> output_shape = {
      static_cast<int>(source_byte_count / sizeof(uint32_t))};
  const std::vector<astronomical_metal_expert_loader_output_tensor>
      output_tensors = {{
          output_shape.data(),
          static_cast<int>(output_shape.size()),
          MLX_UINT32,
      }};
  std::vector<astronomical_metal_expert_loader_load_range> load_ranges;
  for (size_t range_index = 0; range_index < range_count; ++range_index) {
    load_ranges.push_back({
        0,
        range_index * range_byte_count,
        static_cast<uint64_t>(range_index * range_byte_count),
        range_byte_count,
    });
  }
  for (size_t repetition_index = 0; repetition_index < 4;
       ++repetition_index) {
    CompleteTransactionOwner transaction(
        source_pack_path,
        output_tensors,
        load_ranges,
        gpu_stream.get(),
        "repeated_large_eight_range");
    transaction.consume_all_outputs_on_gpu();
    transaction.wait_for_io_completion();
    transaction.release();
    std::cout
        << "[native-metal-expert-loader] status=progress test=repeated_large_eight_range completed_repetitions="
        << repetition_index + 1 << "/4" << std::endl;
  }
  std::cout
      << "[native-metal-expert-loader] status=success test=repeated_large_eight_range"
      << std::endl;
}
