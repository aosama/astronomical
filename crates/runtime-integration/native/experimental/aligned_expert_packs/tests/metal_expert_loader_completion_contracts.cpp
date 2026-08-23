// Proves that explicit completion observes callback-published attribution rather
// than only the earlier native command-buffer status transition.

#include "metal_expert_loader_test_support.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr size_t kCompletionPackByteCount = 64 * 1024;

struct BlockingCompletionCapture {
  std::mutex mutex;
  std::condition_variable condition;
  bool callback_has_started{false};
  bool callback_may_finish{false};
  uint64_t published_elapsed_nanoseconds{0};
  size_t published_callback_count{0};
};

void publish_blocked_completion(
    void* callback_context,
    uint64_t elapsed_nanoseconds,
    int load_succeeded) {
  auto& completion_capture =
      *static_cast<BlockingCompletionCapture*>(callback_context);
  std::unique_lock capture_lock(completion_capture.mutex);
  completion_capture.callback_has_started = true;
  completion_capture.condition.notify_all();
  completion_capture.condition.wait(capture_lock, [&completion_capture]() {
    return completion_capture.callback_may_finish;
  });
  require_condition(load_succeeded == 1, "blocked completion unexpectedly failed");
  completion_capture.published_elapsed_nanoseconds = elapsed_nanoseconds;
  ++completion_capture.published_callback_count;
}

void retain_stack_owned_blocking_capture(void*) {}

void write_completion_pack(const std::filesystem::path& source_pack_path) {
  std::ofstream source_pack_file(
      source_pack_path, std::ios::binary | std::ios::trunc);
  std::vector<uint8_t> source_bytes(kCompletionPackByteCount, 0x27);
  source_pack_file.write(
      reinterpret_cast<const char*>(source_bytes.data()),
      static_cast<std::streamsize>(source_bytes.size()));
  require_condition(
      source_pack_file.good(), "could not write the completion-contract pack");
}

}  // namespace

void should_wait_until_completion_observer_publication_finishes() {
  std::cout
      << "[native-metal-expert-loader] status=progress test=observer_publication phase=setup"
      << std::endl;
  TemporaryDirectory temporary_directory;
  const auto source_pack_path =
      temporary_directory.path() / "observer-publication.expert-pack";
  write_completion_pack(source_pack_path);
  NativeStreamOwner gpu_stream;
  const std::array<int, 1> output_shape = {
      static_cast<int>(kCompletionPackByteCount / sizeof(uint32_t))};
  const std::array<astronomical_metal_expert_loader_output_tensor, 1>
      output_tensors = {{{
          output_shape.data(),
          static_cast<int>(output_shape.size()),
          MLX_UINT32,
      }}};
  const std::array<astronomical_metal_expert_loader_load_range, 1> load_ranges =
      {{{0, 0, 0, kCompletionPackByteCount}}};
  std::vector<mlx_array> output_arrays(1, mlx_array_new());
  astronomical_metal_expert_loader_handle* load_handle = nullptr;
  astronomical_metal_expert_loader_metrics submission_metrics{};
  BlockingCompletionCapture completion_capture;
  require_mlx_success(
      astronomical_metal_expert_loader_start(
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
          publish_blocked_completion,
          retain_stack_owned_blocking_capture),
      "submit observer-publication transaction");

  std::atomic<bool> wait_has_finished{false};
  int wait_status = 1;
  astronomical_metal_expert_loader_metrics completion_metrics{};
  std::thread waiting_thread([&]() {
    wait_status = astronomical_metal_expert_loader_wait(
        load_handle, &completion_metrics);
    wait_has_finished.store(true, std::memory_order_release);
  });
  {
    std::unique_lock capture_lock(completion_capture.mutex);
    require_condition(
        completion_capture.condition.wait_for(
            capture_lock,
            std::chrono::seconds(10),
            [&completion_capture]() {
              return completion_capture.callback_has_started;
            }),
        "completion observer did not start within ten seconds");
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  require_condition(
      !wait_has_finished.load(std::memory_order_acquire),
      "completion wait returned before observer publication finished");
  {
    std::lock_guard capture_lock(completion_capture.mutex);
    completion_capture.callback_may_finish = true;
  }
  completion_capture.condition.notify_all();
  waiting_thread.join();
  require_mlx_success(wait_status, "wait for observer publication");
  require_condition(
      completion_capture.published_callback_count == 1 &&
          completion_capture.published_elapsed_nanoseconds > 0,
      "completion wait did not expose callback-published metrics");

  astronomical_metal_expert_loader_free(load_handle);
  free_output_arrays(output_arrays);
  std::cout
      << "[native-metal-expert-loader] status=success test=observer_publication"
      << std::endl;
}
