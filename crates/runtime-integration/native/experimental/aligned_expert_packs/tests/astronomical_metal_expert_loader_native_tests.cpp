#include "metal_expert_loader_test_support.h"

#include <array>
#include <chrono>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mlx/c/error.h"

#include <unistd.h>

namespace {

constexpr size_t kTestTimeoutSeconds = 120;
constexpr size_t kSyntheticPackByteCount = 64 * 1024;

struct NativeErrorCapture {
  std::string latest_error_message;
};

void capture_mlx_error(const char* error_message, void* raw_error_capture) {
  auto* error_capture = static_cast<NativeErrorCapture*>(raw_error_capture);
  error_capture->latest_error_message =
      error_message == nullptr ? "native MLX reported an empty error" : error_message;
  std::cerr << "[native-metal-expert-loader] status=error mlx_message="
            << error_capture->latest_error_message << std::endl;
}

void retain_stack_owned_error_capture(void*) {}

void fail_after_timeout(int) {
  const char timeout_message[] =
      "[native-metal-expert-loader] status=error reason=timeout_seconds_120\n";
  write(STDERR_FILENO, timeout_message, sizeof(timeout_message) - 1);
  _Exit(124);
}

void write_synthetic_pack(const std::filesystem::path& pack_path) {
  std::ofstream pack_file(pack_path, std::ios::binary | std::ios::trunc);
  require_condition(pack_file.good(), "could not open the synthetic native test pack");
  for (size_t byte_index = 0; byte_index < kSyntheticPackByteCount; ++byte_index) {
    const auto source_byte = static_cast<char>((byte_index * 31U) % 251U);
    pack_file.write(&source_byte, 1);
  }
  require_condition(pack_file.good(), "could not write the synthetic native test pack");
}

std::vector<uint8_t> read_file_bytes(const std::filesystem::path& file_path) {
  std::ifstream input_file(file_path, std::ios::binary | std::ios::ate);
  require_condition(input_file.good(), "could not open a native test file");
  const auto byte_count = input_file.tellg();
  require_condition(byte_count >= 0, "could not measure a native test file");
  input_file.seekg(0);
  std::vector<uint8_t> file_bytes(static_cast<size_t>(byte_count));
  input_file.read(
      reinterpret_cast<char*>(file_bytes.data()),
      static_cast<std::streamsize>(file_bytes.size()));
  require_condition(input_file.good(), "could not read a native test file");
  return file_bytes;
}

void require_output_bytes(
    const mlx_array output_array,
    const uint8_t* expected_bytes,
    size_t expected_byte_count,
    const std::string& description) {
  require_condition(
      mlx_array_nbytes(output_array) == expected_byte_count,
      description + " has an unexpected byte count");
  const auto* observed_bytes = mlx_array_data_uint8(output_array);
  require_condition(observed_bytes != nullptr, description + " has no evaluated byte data");
  require_condition(
      std::memcmp(observed_bytes, expected_bytes, expected_byte_count) == 0,
      description + " differs from its source bytes");
}

void should_load_contiguous_bytes_before_native_gpu_consumption() {
  std::cout
      << "[native-metal-expert-loader] status=progress test=contiguous_gpu_consumption phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path = temporary_directory.path() / "contiguous.expert-pack";
  write_synthetic_pack(source_pack_path);
  const auto source_bytes = read_file_bytes(source_pack_path);
  NativeStreamOwner gpu_stream;
  const std::vector<int> output_shape = {
      static_cast<int>(kSyntheticPackByteCount / sizeof(uint32_t))};
  const std::vector<astronomical_metal_expert_loader_output_tensor> output_tensors = {{
      output_shape.data(),
      static_cast<int>(output_shape.size()),
      MLX_UINT32,
  }};
  const std::vector<astronomical_metal_expert_loader_load_range> load_ranges = {{
      0,
      0,
      0,
      kSyntheticPackByteCount,
  }};
  CompleteTransactionOwner transaction(
      source_pack_path,
      output_tensors,
      load_ranges,
      gpu_stream.get(),
      "contiguous_gpu_consumption");
  transaction.consume_all_outputs_on_gpu();
  transaction.wait_for_io_completion();
  require_output_bytes(
      transaction.output_arrays()[0],
      source_bytes.data(),
      source_bytes.size(),
      "contiguous native output");
  require_condition(
      transaction.io_metrics().command_count == 1,
      "contiguous native transaction reported an unexpected command count");
  require_condition(
      transaction.io_metrics().requested_byte_count == kSyntheticPackByteCount,
      "contiguous native transaction reported an unexpected byte count");
  transaction.release();
  std::cout
      << "[native-metal-expert-loader] status=success test=contiguous_gpu_consumption"
      << std::endl;
}

void should_assemble_shuffled_ranges_into_multiple_native_outputs() {
  std::cout
      << "[native-metal-expert-loader] status=progress test=shuffled_multi_output phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path = temporary_directory.path() / "shuffled.expert-pack";
  write_synthetic_pack(source_pack_path);
  const auto source_bytes = read_file_bytes(source_pack_path);
  NativeStreamOwner gpu_stream;
  constexpr size_t range_byte_count = kSyntheticPackByteCount / 4;
  const std::array<std::vector<int>, 2> output_shapes = {{
      {static_cast<int>((range_byte_count * 2) / sizeof(uint32_t))},
      {static_cast<int>(range_byte_count / sizeof(uint32_t))},
  }};
  const std::vector<astronomical_metal_expert_loader_output_tensor> output_tensors = {
      {output_shapes[0].data(), static_cast<int>(output_shapes[0].size()), MLX_UINT32},
      {output_shapes[1].data(), static_cast<int>(output_shapes[1].size()), MLX_UINT32},
  };
  const std::vector<astronomical_metal_expert_loader_load_range> load_ranges = {{
      1,
      0,
      static_cast<uint64_t>(range_byte_count * 3),
      range_byte_count,
  }, {
      0,
      range_byte_count,
      static_cast<uint64_t>(range_byte_count * 2),
      range_byte_count,
  }, {
      0,
      0,
      0,
      range_byte_count,
  }};
  CompleteTransactionOwner transaction(
      source_pack_path,
      output_tensors,
      load_ranges,
      gpu_stream.get(),
      "shuffled_multi_output");
  transaction.consume_all_outputs_on_gpu();
  transaction.wait_for_io_completion();
  std::vector<uint8_t> expected_first_output;
  expected_first_output.insert(
      expected_first_output.end(),
      source_bytes.begin(),
      source_bytes.begin() + static_cast<std::ptrdiff_t>(range_byte_count));
  expected_first_output.insert(
      expected_first_output.end(),
      source_bytes.begin() + static_cast<std::ptrdiff_t>(range_byte_count * 2),
      source_bytes.begin() + static_cast<std::ptrdiff_t>(range_byte_count * 3));
  require_output_bytes(
      transaction.output_arrays()[0],
      expected_first_output.data(),
      expected_first_output.size(),
      "first shuffled native output");
  require_output_bytes(
      transaction.output_arrays()[1],
      source_bytes.data() + range_byte_count * 3,
      range_byte_count,
      "second shuffled native output");
  require_condition(
      transaction.io_metrics().command_count == load_ranges.size(),
      "shuffled native transaction reported an unexpected command count");
  transaction.release();
  std::cout
      << "[native-metal-expert-loader] status=success test=shuffled_multi_output"
      << std::endl;
}

void should_reject_source_overflow_and_empty_ranges(
    NativeErrorCapture& native_error_capture) {
  std::cout
      << "[native-metal-expert-loader] status=progress test=invalid_native_ranges phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path = temporary_directory.path() / "invalid-ranges.expert-pack";
  write_synthetic_pack(source_pack_path);
  NativeStreamOwner gpu_stream;
  const std::array<int, 1> output_shape = {
      static_cast<int>(kSyntheticPackByteCount / sizeof(uint32_t))};
  const std::array<astronomical_metal_expert_loader_output_tensor, 1> output_tensors = {{
      {output_shape.data(), static_cast<int>(output_shape.size()), MLX_UINT32},
  }};
  const std::array<astronomical_metal_expert_loader_load_range, 1> source_overflow_range = {{
      {0, 0, 1, kSyntheticPackByteCount},
  }};
  std::vector<mlx_array> output_arrays(1, mlx_array_new());
  astronomical_metal_expert_loader_handle* load_handle = nullptr;
  native_error_capture.latest_error_message.clear();
  const auto source_overflow_status = astronomical_metal_expert_loader_start(
      source_pack_path.c_str(),
      output_tensors.data(),
      output_tensors.size(),
      source_overflow_range.data(),
      source_overflow_range.size(),
      gpu_stream.get(),
      output_arrays.data(),
      &load_handle,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
  astronomical_metal_expert_loader_free(load_handle);
  free_output_arrays(output_arrays);
  require_condition(
      source_overflow_status != 0 && !native_error_capture.latest_error_message.empty(),
      "native loader accepted a source range beyond the file");
  require_condition(
      native_error_capture.latest_error_message.find("range is invalid") !=
          std::string::npos,
      "native loader discarded the source-range failure cause");

  native_error_capture.latest_error_message.clear();
  const auto empty_range_status = astronomical_metal_expert_loader_start(
      source_pack_path.c_str(),
      output_tensors.data(),
      output_tensors.size(),
      nullptr,
      0,
      gpu_stream.get(),
      output_arrays.data(),
      &load_handle,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
  astronomical_metal_expert_loader_free(load_handle);
  free_output_arrays(output_arrays);
  require_condition(
      empty_range_status != 0 && !native_error_capture.latest_error_message.empty(),
      "native loader accepted an empty range list");

  const std::array<astronomical_metal_expert_loader_load_range, 1>
      partially_initialized_output_range = {{
          {0, 0, 0, kSyntheticPackByteCount / 2},
      }};
  native_error_capture.latest_error_message.clear();
  const auto partially_initialized_output_status =
      astronomical_metal_expert_loader_start(
          source_pack_path.c_str(),
          output_tensors.data(),
          output_tensors.size(),
          partially_initialized_output_range.data(),
          partially_initialized_output_range.size(),
          gpu_stream.get(),
          output_arrays.data(),
          &load_handle,
          nullptr,
          nullptr,
          nullptr,
          nullptr);
  astronomical_metal_expert_loader_free(load_handle);
  free_output_arrays(output_arrays);
  require_condition(
      partially_initialized_output_status != 0 &&
          !native_error_capture.latest_error_message.empty(),
      "native loader accepted an output tensor containing uninitialized bytes");
  require_condition(
      native_error_capture.latest_error_message.find(
          "exactly cover every output tensor") != std::string::npos,
      "native loader discarded the incomplete-output failure cause");
  std::cout
      << "[native-metal-expert-loader] status=success test=invalid_native_ranges"
      << std::endl;
}

void should_record_callback_completion_before_delayed_metric_read() {
  std::cout
      << "[native-metal-expert-loader] status=progress test=callback_completion_timing phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path = temporary_directory.path() / "timing.expert-pack";
  write_synthetic_pack(source_pack_path);
  NativeStreamOwner gpu_stream;
  const std::vector<int> output_shape = {
      static_cast<int>(kSyntheticPackByteCount / sizeof(uint32_t))};
  const std::vector<astronomical_metal_expert_loader_output_tensor> output_tensors = {{
      output_shape.data(),
      static_cast<int>(output_shape.size()),
      MLX_UINT32,
  }};
  const std::vector<astronomical_metal_expert_loader_load_range> load_ranges = {{
      0,
      0,
      0,
      kSyntheticPackByteCount,
  }};
  CompleteTransactionOwner transaction(
      source_pack_path,
      output_tensors,
      load_ranges,
      gpu_stream.get(),
      "callback_completion_timing");
  transaction.consume_all_outputs_on_gpu();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  const auto elapsed_before_metric_read_nanoseconds = transaction.elapsed_nanoseconds();
  transaction.wait_for_io_completion();
  constexpr uint64_t callback_timing_tolerance_nanoseconds = 5'000'000;
  require_condition(
      transaction.io_metrics().queue_elapsed_nanoseconds +
              callback_timing_tolerance_nanoseconds <
          elapsed_before_metric_read_nanoseconds,
      "delayed metric collection inflated the Metal I/O completion duration");
  transaction.release();
  std::cout
      << "[native-metal-expert-loader] status=success test=callback_completion_timing"
      << std::endl;
}

void should_reject_overlapping_native_output_ranges(
    NativeErrorCapture& native_error_capture) {
  std::cout
      << "[native-metal-expert-loader] status=progress test=overlapping_output_ranges phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path = temporary_directory.path() / "synthetic.expert-pack";
  write_synthetic_pack(source_pack_path);
  NativeStreamOwner gpu_stream;
  const std::array<int, 1> output_shape = {
      static_cast<int>(kSyntheticPackByteCount / sizeof(uint32_t))};
  const std::array<astronomical_metal_expert_loader_output_tensor, 1> output_tensors = {{
      output_shape.data(),
      static_cast<int>(output_shape.size()),
      MLX_UINT32,
  }};
  const std::array<astronomical_metal_expert_loader_load_range, 2> overlapping_load_ranges = {{
      {0, 0, 0, kSyntheticPackByteCount / 2},
      {0, kSyntheticPackByteCount / 4, kSyntheticPackByteCount / 2,
       kSyntheticPackByteCount / 2},
  }};
  std::vector<mlx_array> output_arrays(1, mlx_array_new());
  astronomical_metal_expert_loader_handle* load_handle = nullptr;
  native_error_capture.latest_error_message.clear();

  std::cout
      << "[native-metal-expert-loader] status=progress test=overlapping_output_ranges phase=submit"
      << std::endl;
  const auto submission_status = astronomical_metal_expert_loader_start(
      source_pack_path.c_str(),
      output_tensors.data(),
      output_tensors.size(),
      overlapping_load_ranges.data(),
      overlapping_load_ranges.size(),
      gpu_stream.get(),
      output_arrays.data(),
      &load_handle,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
  astronomical_metal_expert_loader_free(load_handle);
  free_output_arrays(output_arrays);

  require_condition(
      submission_status != 0,
      "native loader accepted overlapping output ranges");
  require_condition(
      !native_error_capture.latest_error_message.empty(),
      "native loader rejected overlapping output ranges without reporting an error");
  std::cout
      << "[native-metal-expert-loader] status=success test=overlapping_output_ranges"
      << std::endl;
}

void run_contracts(NativeErrorCapture& native_error_capture) {
  should_reject_overlapping_native_output_ranges(native_error_capture);
  should_wait_until_completion_observer_publication_finishes();
  should_fail_downstream_wait_and_recover_after_asynchronous_io_failure();
  should_keep_failed_event_error_alive_when_handle_is_freed_on_another_thread();
  should_load_contiguous_bytes_before_native_gpu_consumption();
  should_assemble_shuffled_ranges_into_multiple_native_outputs();
  should_reject_source_overflow_and_empty_ranges(native_error_capture);
  should_record_callback_completion_before_delayed_metric_read();
  should_release_native_transaction_allocations_between_repetitions();
  should_repeat_large_eight_range_native_transactions();
}

}  // namespace

int main(int argument_count, char* argument_values[]) {
  std::signal(SIGALRM, fail_after_timeout);
  alarm(kTestTimeoutSeconds);
  try {
    NativeErrorCapture native_error_capture;
    mlx_set_error_handler(
        capture_mlx_error,
        &native_error_capture,
        retain_stack_owned_error_capture);
    if (argument_count == 2 && std::string(argument_values[1]) == "contracts") {
      std::cout
          << "[native-metal-expert-loader] status=start mode=contracts timeout_seconds="
          << kTestTimeoutSeconds << std::endl;
      run_contracts(native_error_capture);
      std::cout << "[native-metal-expert-loader] status=success mode=contracts" << std::endl;
    } else {
      throw std::runtime_error(
          "usage: astronomical_metal_expert_loader_native_tests contracts");
    }
    return 0;
  } catch (const std::exception& failure) {
    std::cerr << "[native-metal-expert-loader] status=error message=" << failure.what()
              << std::endl;
    return 1;
  }
}
