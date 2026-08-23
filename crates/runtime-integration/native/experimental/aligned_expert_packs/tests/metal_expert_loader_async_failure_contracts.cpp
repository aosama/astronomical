// Exercises the asynchronous hardware-failure path that valid file ranges cannot
// trigger deterministically on every supported Apple silicon machine.

#include "metal_expert_loader_test_support.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mlx/c/error.h"

namespace {

constexpr size_t kFailurePackByteCount = 64 * 1024;

struct CompletionCapture {
  uint64_t elapsed_nanoseconds{0};
  size_t callback_count{0};
  size_t failed_load_count{0};
};

void record_completion(
    void* callback_context,
    uint64_t elapsed_nanoseconds,
    int load_succeeded) {
  auto& completion_capture = *static_cast<CompletionCapture*>(callback_context);
  completion_capture.elapsed_nanoseconds = elapsed_nanoseconds;
  ++completion_capture.callback_count;
  if (load_succeeded == 0) {
    ++completion_capture.failed_load_count;
  }
}

void retain_stack_owned_completion_capture(void*) {}

void write_failure_pack(const std::filesystem::path& source_pack_path) {
  std::ofstream source_pack_file(
      source_pack_path, std::ios::binary | std::ios::trunc);
  std::vector<uint8_t> source_bytes(kFailurePackByteCount, 0x5A);
  source_pack_file.write(
      reinterpret_cast<const char*>(source_bytes.data()),
      static_cast<std::streamsize>(source_bytes.size()));
  require_condition(
      source_pack_file.good(), "could not write the asynchronous-failure pack");
}

}  // namespace

void should_fail_downstream_wait_and_recover_after_asynchronous_io_failure() {
  std::cout
      << "[native-metal-expert-loader] status=progress test=async_failure_recovery phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path =
      temporary_directory.path() / "asynchronous-failure.expert-pack";
  write_failure_pack(source_pack_path);
  NativeStreamOwner gpu_stream;
  const std::array<int, 1> output_shape = {
      static_cast<int>(kFailurePackByteCount / sizeof(uint32_t))};
  const std::array<astronomical_metal_expert_loader_output_tensor, 1>
      output_tensors = {{{
          output_shape.data(),
          static_cast<int>(output_shape.size()),
          MLX_UINT32,
      }}};
  const std::array<astronomical_metal_expert_loader_load_range, 1> load_ranges =
      {{{0, 0, 0, kFailurePackByteCount}}};
  std::vector<mlx_array> output_arrays(1, mlx_array_new());
  astronomical_metal_expert_loader_handle* load_handle = nullptr;
  astronomical_metal_expert_loader_metrics submission_metrics{};
  CompletionCapture completion_capture;

  require_mlx_success(
      astronomical_metal_expert_loader_start_with_async_failure(
          source_pack_path.c_str(),
          output_tensors.data(),
          output_tensors.size(),
          load_ranges.data(),
          load_ranges.size(),
          gpu_stream.get(),
          output_arrays.data(),
          &load_handle,
          &submission_metrics,
          &completion_capture,
          record_completion,
          retain_stack_owned_completion_capture),
      "submit deterministic asynchronous Metal I/O failure");
  require_condition(load_handle != nullptr, "failed load returned no lifetime owner");

  mlx_array downstream_output = mlx_array_new();
  require_mlx_success(
      mlx_add(
          &downstream_output,
          output_arrays[0],
          output_arrays[0],
          gpu_stream.get()),
      "build downstream asynchronous-failure consumer");
  mlx_vector_array downstream_outputs =
      mlx_vector_array_new_data(&downstream_output, 1);
  const auto downstream_submission_status = mlx_async_eval(downstream_outputs);
  const auto downstream_synchronization_status =
      downstream_submission_status == 0 ? mlx_synchronize(gpu_stream.get()) : 0;
  require_condition(
      downstream_submission_status != 0 || downstream_synchronization_status != 0,
      "downstream MLX synchronization consumed output from a failed Metal I/O load");

  astronomical_metal_expert_loader_metrics completion_metrics{};
  require_condition(
      astronomical_metal_expert_loader_wait(load_handle, &completion_metrics) != 0,
      "explicit completion wait accepted a failed Metal I/O load");
  require_condition(
      completion_metrics.final_status != 3,
      "failed Metal I/O load reported complete status");
  require_condition(
      completion_capture.callback_count == 1 &&
          completion_capture.failed_load_count == 1,
      "failed Metal I/O load did not record one failed completion");
  require_condition(
      completion_capture.elapsed_nanoseconds > 0 &&
          submission_metrics.command_count == 1,
      "failed Metal I/O load discarded its timing or command metrics");

  require_mlx_success(
      mlx_vector_array_free(downstream_outputs),
      "free failed downstream output vector");
  mlx_array_free(downstream_output);
  astronomical_metal_expert_loader_free(load_handle);
  free_output_arrays(output_arrays);

  const std::vector<astronomical_metal_expert_loader_output_tensor>
      recovery_output_tensors(output_tensors.begin(), output_tensors.end());
  const std::vector<astronomical_metal_expert_loader_load_range>
      recovery_load_ranges(load_ranges.begin(), load_ranges.end());
  CompleteTransactionOwner recovery_transaction(
      source_pack_path,
      recovery_output_tensors,
      recovery_load_ranges,
      gpu_stream.get(),
      "async_failure_recovery");
  recovery_transaction.consume_all_outputs_on_gpu();
  recovery_transaction.wait_for_io_completion();
  recovery_transaction.release();
  std::cout
      << "[native-metal-expert-loader] status=success test=async_failure_recovery"
      << std::endl;
}

void should_keep_failed_event_error_alive_when_handle_is_freed_on_another_thread() {
  std::cout
      << "[native-metal-expert-loader] status=progress test=cross_thread_failed_free phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path =
      temporary_directory.path() / "cross-thread-failure.expert-pack";
  write_failure_pack(source_pack_path);
  NativeStreamOwner gpu_stream;
  const std::array<int, 1> output_shape = {
      static_cast<int>(kFailurePackByteCount / sizeof(uint32_t))};
  const std::array<astronomical_metal_expert_loader_output_tensor, 1>
      output_tensors = {{{
          output_shape.data(),
          static_cast<int>(output_shape.size()),
          MLX_UINT32,
      }}};
  const std::array<astronomical_metal_expert_loader_load_range, 1> load_ranges =
      {{{0, 0, 0, kFailurePackByteCount}}};
  std::vector<mlx_array> output_arrays(1, mlx_array_new());
  astronomical_metal_expert_loader_handle* load_handle = nullptr;
  astronomical_metal_expert_loader_metrics submission_metrics{};
  CompletionCapture completion_capture;
  require_mlx_success(
      astronomical_metal_expert_loader_start_with_async_failure(
          source_pack_path.c_str(),
          output_tensors.data(),
          output_tensors.size(),
          load_ranges.data(),
          load_ranges.size(),
          gpu_stream.get(),
          output_arrays.data(),
          &load_handle,
          &submission_metrics,
          &completion_capture,
          record_completion,
          retain_stack_owned_completion_capture),
      "submit cross-thread asynchronous Metal I/O failure");

  std::thread freeing_thread([load_handle]() {
    astronomical_metal_expert_loader_free(load_handle);
  });
  freeing_thread.join();
  require_condition(
      mlx_synchronize(gpu_stream.get()) != 0,
      "cross-thread handle release discarded the queued MLX event failure");
  require_condition(
      completion_capture.callback_count == 1 &&
          completion_capture.failed_load_count == 1,
      "cross-thread handle release returned before failed metrics publication");
  free_output_arrays(output_arrays);

  const std::vector<astronomical_metal_expert_loader_output_tensor>
      recovery_output_tensors(output_tensors.begin(), output_tensors.end());
  const std::vector<astronomical_metal_expert_loader_load_range>
      recovery_load_ranges(load_ranges.begin(), load_ranges.end());
  CompleteTransactionOwner recovery_transaction(
      source_pack_path,
      recovery_output_tensors,
      recovery_load_ranges,
      gpu_stream.get(),
      "cross_thread_failed_free_recovery");
  recovery_transaction.consume_all_outputs_on_gpu();
  recovery_transaction.wait_for_io_completion();
  recovery_transaction.release();
  std::cout
      << "[native-metal-expert-loader] status=success test=cross_thread_failed_free"
      << std::endl;
}
