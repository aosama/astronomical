#include "astronomical_metal_expert_loader.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/event.h"
#include "mlx/error.h"
#include "mlx/event.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/stream.h"

#include "metal_expert_loader_validation.h"

namespace {

constexpr uint64_t kSharedEventValue = 1;

using SteadyClock = std::chrono::steady_clock;
constexpr auto kNativeCompletionTimeout = std::chrono::seconds(10);

struct IoCompletionState {
  std::mutex mutex;
  std::condition_variable completion_condition;
  bool has_published_completion{false};
  uint64_t elapsed_nanoseconds{0};
  int final_status{0};
};

struct IoSynchronization {
  mlx::core::Error completion_error;
  mlx::core::Event completion_event;
  mlx::core::Stream target_gpu_stream;

  explicit IoSynchronization(mlx::core::Stream target_gpu_stream)
      : completion_event(target_gpu_stream),
        target_gpu_stream(target_gpu_stream) {
    completion_event.set_value(kSharedEventValue);
  }
};

struct IoCompletionObserver {
  void* callback_context;
  astronomical_metal_expert_loader_completion_callback completion_callback;
  astronomical_metal_expert_loader_release_callback release_callback;

  IoCompletionObserver(
      void* callback_context,
      astronomical_metal_expert_loader_completion_callback completion_callback,
      astronomical_metal_expert_loader_release_callback release_callback)
      : callback_context(callback_context),
        completion_callback(completion_callback),
        release_callback(release_callback) {}

  IoCompletionObserver(const IoCompletionObserver&) = delete;
  IoCompletionObserver& operator=(const IoCompletionObserver&) = delete;

  ~IoCompletionObserver() {
    release_callback(callback_context);
  }

  void record(uint64_t queue_elapsed_nanoseconds, bool load_succeeded) const {
    completion_callback(
        callback_context, queue_elapsed_nanoseconds, load_succeeded ? 1 : 0);
  }
};

void clear_output_arrays(mlx_array* output_arrays, size_t output_tensor_count) {
  if (output_arrays == nullptr) {
    return;
  }
  for (size_t output_tensor_index = 0;
       output_tensor_index < output_tensor_count;
       ++output_tensor_index) {
    mlx_array_free_(output_arrays[output_tensor_index]);
    output_arrays[output_tensor_index] = mlx_array_new_();
  }
}

void report_native_failure(const std::exception& native_failure) {
  const std::string failure_message =
      std::string("Metal I/O expert-pack operation failed: ") +
      native_failure.what();
  mlx_error(failure_message.c_str());
}

void report_unknown_native_failure() {
  mlx_error("Metal I/O expert-pack operation failed with an unknown native exception");
}

NS::SharedPtr<MTL::IOCommandQueue> shared_io_command_queue(
    mlx::core::metal::Device& metal_device) {
  static auto io_command_queue = [&metal_device]() {
    auto scoped_memory_pool = mlx::core::metal::new_scoped_memory_pool();
    auto queue_descriptor =
        MTL::IOCommandQueueDescriptor::alloc()->init()->autorelease();
    queue_descriptor->setType(MTL::IOCommandQueueTypeConcurrent);
    queue_descriptor->setPriority(MTL::IOPriorityNormal);
    queue_descriptor->setMaxCommandBufferCount(1);
    NS::Error* native_error = nullptr;
    auto created_io_command_queue = NS::TransferPtr(
        metal_device.mtl_device()->newIOCommandQueue(
            queue_descriptor, &native_error));
    if (!created_io_command_queue) {
      throw std::runtime_error(
          "Metal I/O could not create the shared expert-pack queue");
    }
    return created_io_command_queue;
  }();
  return io_command_queue;
}

}

struct astronomical_metal_expert_loader_handle_ {
  NS::SharedPtr<MTL::IOCommandQueue> io_command_queue;
  NS::SharedPtr<MTL::IOCommandBuffer> io_command_buffer;
  NS::SharedPtr<MTL::IOFileHandle> source_file_handle;
  uint64_t requested_byte_count;
  size_t command_count;
  uint64_t host_encoding_elapsed_nanoseconds;
  std::shared_ptr<IoCompletionState> completion_state;
};

static int start_metal_expert_loader(
    const char* source_file_path,
    const astronomical_metal_expert_loader_output_tensor* output_tensors,
    size_t output_tensor_count,
    const astronomical_metal_expert_loader_load_range* load_ranges,
    size_t load_range_count,
    mlx_stream target_gpu_stream,
    mlx_array* output_arrays,
    astronomical_metal_expert_loader_handle** output_handle,
    astronomical_metal_expert_loader_metrics* output_submission_metrics,
    std::shared_ptr<IoCompletionObserver> completion_observer,
    bool inject_asynchronous_failure) {
  if (output_handle != nullptr) {
    *output_handle = nullptr;
  }
  try {
    if (source_file_path == nullptr ||
        output_arrays == nullptr || output_handle == nullptr) {
      throw std::invalid_argument("Metal I/O expert-pack arguments are invalid");
    }
    const auto encoding_started_at = SteadyClock::now();
    const auto output_tensor_byte_counts = validate_output_tensors(
        output_tensors, output_tensor_count);
    const auto source_file_size_bytes = std::filesystem::file_size(source_file_path);
    validate_load_ranges(
        load_ranges,
        load_range_count,
        output_tensor_byte_counts,
        source_file_size_bytes);
    allocate_output_arrays(
        output_tensors,
        output_tensor_count,
        output_tensor_byte_counts,
        output_arrays);
    auto scoped_memory_pool = mlx::core::metal::new_scoped_memory_pool();
    auto& metal_device =
        mlx::core::metal::device(mlx::core::Device::gpu);
    auto source_file_path_string =
        NS::String::string(source_file_path, NS::UTF8StringEncoding);
    auto source_file_url = NS::URL::fileURLWithPath(source_file_path_string);
    NS::Error* native_error = nullptr;
    auto source_file_handle = NS::TransferPtr(
        metal_device.mtl_device()->newIOFileHandle(source_file_url, &native_error));
    if (!source_file_handle) {
      throw std::runtime_error("Metal I/O could not open the expert pack");
    }
    auto io_command_queue = shared_io_command_queue(metal_device);
    auto io_command_buffer = NS::RetainPtr(io_command_queue->commandBuffer());
    if (!io_command_buffer) {
      throw std::runtime_error(
          "Metal I/O could not create an expert-pack command buffer");
    }
    auto io_synchronization = std::make_shared<IoSynchronization>(
        mlx_stream_get_(target_gpu_stream));
    uint64_t requested_byte_count = 0;
    for (size_t load_range_index = 0; load_range_index < load_range_count;
         ++load_range_index) {
      const auto& load_range = load_ranges[load_range_index];
      auto* destination_buffer = static_cast<MTL::Buffer*>(
          mlx_array_get_(output_arrays[load_range.output_tensor_index])
              .buffer()
              .ptr());
      io_command_buffer->loadBuffer(
          destination_buffer,
          static_cast<NS::UInteger>(load_range.output_tensor_offset_bytes),
          static_cast<NS::UInteger>(load_range.byte_count),
          source_file_handle.get(),
          static_cast<NS::UInteger>(load_range.source_file_offset_bytes));
      if (load_range.byte_count >
          std::numeric_limits<uint64_t>::max() - requested_byte_count) {
        throw std::overflow_error("Metal I/O requested byte count overflowed");
      }
      requested_byte_count += load_range.byte_count;
    }
    if (!inject_asynchronous_failure) {
      // Metal I/O must signal the exact shared event retained by MLX's public
      // Event so GPU work observes the file writes without a host round trip.
      io_command_buffer->signalEvent(
          io_synchronization->completion_event
              .cast<mlx::core::metal::EventImpl>()
              .mtl_event(),
          kSharedEventValue);
    }
    const auto submitted_at = SteadyClock::now();
    auto completion_state = std::make_shared<IoCompletionState>();
    io_command_buffer->addCompletedHandler(
        [io_synchronization,
          completion_state,
          completion_observer,
          inject_asynchronous_failure,
          submitted_at](MTL::IOCommandBuffer* completed_command_buffer) {
          const bool load_succeeded =
              completed_command_buffer->status() == MTL::IOStatusComplete &&
              !inject_asynchronous_failure;
          const auto final_status = load_succeeded
              ? MTL::IOStatusComplete
              : MTL::IOStatusError;
          const auto elapsed_nanoseconds = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  SteadyClock::now() - submitted_at)
                  .count());
          if (!load_succeeded) {
            io_synchronization->completion_error.set_message(
                std::make_shared<std::string>(
                    "Metal I/O expert-pack command buffer failed asynchronously"));
            io_synchronization->completion_event.set_error(
                io_synchronization->completion_error);
            // Failed Metal I/O does not signal its encoded event. Publishing
            // the error first ensures every awakened MLX waiter rejects the
            // incomplete destination instead of deadlocking or consuming it.
            io_synchronization->completion_event
                .cast<mlx::core::metal::EventImpl>()
                .signal(kSharedEventValue);
          }
          {
            std::lock_guard completion_lock(completion_state->mutex);
            completion_state->elapsed_nanoseconds = elapsed_nanoseconds;
            completion_state->final_status = static_cast<int>(final_status);
          }
          if (completion_observer) {
            completion_observer->record(
                elapsed_nanoseconds, load_succeeded);
          }
          {
            // Publication is the completion boundary because callback-owned
            // metrics are intentionally not forced to use atomics or this mutex.
            std::lock_guard completion_lock(completion_state->mutex);
            completion_state->has_published_completion = true;
          }
          completion_state->completion_condition.notify_all();
        });
    io_command_buffer->commit();
    io_synchronization->completion_event.wait(
        io_synchronization->target_gpu_stream);
    auto& target_command_encoder = mlx::core::metal::get_command_encoder(
        io_synchronization->target_gpu_stream);
    // The pinned completion closure remains owned until after MLX copies any
    // waited-event error into the stream, so it is the retirement owner even
    // when the C handle is released from another thread.
    target_command_encoder.commit([io_synchronization]() {});
    const auto host_encoding_elapsed_nanoseconds = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - encoding_started_at)
            .count());
    auto* load_handle = new astronomical_metal_expert_loader_handle{
        std::move(io_command_queue),
        std::move(io_command_buffer),
        std::move(source_file_handle),
        requested_byte_count,
        load_range_count,
        host_encoding_elapsed_nanoseconds,
        std::move(completion_state)};
    if (output_submission_metrics != nullptr) {
      *output_submission_metrics = {
          requested_byte_count,
          load_range_count,
          host_encoding_elapsed_nanoseconds,
          0,
          0};
    }
    *output_handle = load_handle;
    return 0;
  } catch (const std::exception& native_failure) {
    clear_output_arrays(output_arrays, output_tensor_count);
    report_native_failure(native_failure);
    return 1;
  } catch (...) {
    clear_output_arrays(output_arrays, output_tensor_count);
    report_unknown_native_failure();
    return 1;
  }
}

extern "C" int astronomical_metal_expert_loader_start(
    const char* source_file_path,
    const astronomical_metal_expert_loader_output_tensor* output_tensors,
    size_t output_tensor_count,
    const astronomical_metal_expert_loader_load_range* load_ranges,
    size_t load_range_count,
    mlx_stream target_gpu_stream,
    mlx_array* output_arrays,
    astronomical_metal_expert_loader_handle** output_handle,
    astronomical_metal_expert_loader_metrics* output_submission_metrics,
    void* completion_callback_context,
    astronomical_metal_expert_loader_completion_callback completion_callback,
    astronomical_metal_expert_loader_release_callback release_callback) {
  try {
    if (completion_callback_context == nullptr) {
      if (output_submission_metrics != nullptr || completion_callback != nullptr ||
          release_callback != nullptr) {
        throw std::invalid_argument(
            "Metal I/O expert-pack attribution arguments are incomplete");
      }
      return start_metal_expert_loader(
          source_file_path,
          output_tensors,
          output_tensor_count,
          load_ranges,
          load_range_count,
          target_gpu_stream,
          output_arrays,
          output_handle,
          nullptr,
          nullptr,
          false);
    }
    if (output_submission_metrics == nullptr || completion_callback == nullptr ||
        release_callback == nullptr) {
      throw std::invalid_argument(
          "Metal I/O expert-pack attribution arguments are invalid");
    }
    auto completion_observer = std::make_shared<IoCompletionObserver>(
        completion_callback_context, completion_callback, release_callback);
    return start_metal_expert_loader(
        source_file_path,
        output_tensors,
        output_tensor_count,
        load_ranges,
        load_range_count,
        target_gpu_stream,
        output_arrays,
        output_handle,
        output_submission_metrics,
        std::move(completion_observer),
        false);
  } catch (const std::exception& native_failure) {
    if (completion_callback_context != nullptr && release_callback != nullptr) {
      release_callback(completion_callback_context);
    }
    report_native_failure(native_failure);
    return 1;
  } catch (...) {
    if (completion_callback_context != nullptr && release_callback != nullptr) {
      release_callback(completion_callback_context);
    }
    report_unknown_native_failure();
    return 1;
  }
}

#ifdef ASTRONOMICAL_METAL_EXPERT_LOADER_TESTING
extern "C" int astronomical_metal_expert_loader_start_with_async_failure(
    const char* source_file_path,
    const astronomical_metal_expert_loader_output_tensor* output_tensors,
    size_t output_tensor_count,
    const astronomical_metal_expert_loader_load_range* load_ranges,
    size_t load_range_count,
    mlx_stream target_gpu_stream,
    mlx_array* output_arrays,
    astronomical_metal_expert_loader_handle** output_handle,
    astronomical_metal_expert_loader_metrics* output_submission_metrics,
    void* completion_callback_context,
    astronomical_metal_expert_loader_completion_callback completion_callback,
    astronomical_metal_expert_loader_release_callback release_callback) {
  try {
    if (completion_callback_context == nullptr ||
        output_submission_metrics == nullptr || completion_callback == nullptr ||
        release_callback == nullptr) {
      throw std::invalid_argument(
          "asynchronous-failure attribution arguments are invalid");
    }
    auto completion_observer = std::make_shared<IoCompletionObserver>(
        completion_callback_context, completion_callback, release_callback);
    return start_metal_expert_loader(
        source_file_path,
        output_tensors,
        output_tensor_count,
        load_ranges,
        load_range_count,
        target_gpu_stream,
        output_arrays,
        output_handle,
        output_submission_metrics,
        std::move(completion_observer),
        true);
  } catch (const std::exception& native_failure) {
    if (completion_callback_context != nullptr && release_callback != nullptr) {
      release_callback(completion_callback_context);
    }
    report_native_failure(native_failure);
    return 1;
  } catch (...) {
    if (completion_callback_context != nullptr && release_callback != nullptr) {
      release_callback(completion_callback_context);
    }
    report_unknown_native_failure();
    return 1;
  }
}
#endif

extern "C" int astronomical_metal_expert_loader_wait(
    astronomical_metal_expert_loader_handle* load_handle,
    astronomical_metal_expert_loader_metrics* output_metrics) {
  try {
    if (load_handle == nullptr || output_metrics == nullptr) {
      throw std::invalid_argument("Metal I/O expert-pack wait arguments are invalid");
    }
    std::unique_lock completion_lock(load_handle->completion_state->mutex);
    const auto did_complete =
        load_handle->completion_state->completion_condition.wait_for(
            completion_lock,
             kNativeCompletionTimeout,
             [&completion_state = *load_handle->completion_state]() {
               return completion_state.has_published_completion;
             });
    if (!did_complete) {
      throw std::runtime_error(
          "Metal I/O expert-pack completion exceeded 10 seconds");
    }
    const auto final_status = load_handle->completion_state->final_status;
    *output_metrics = {
        load_handle->requested_byte_count,
        load_handle->command_count,
        load_handle->host_encoding_elapsed_nanoseconds,
        load_handle->completion_state->elapsed_nanoseconds,
        final_status};
    if (final_status != static_cast<int>(MTL::IOStatusComplete)) {
      throw std::runtime_error("Metal I/O expert-pack command buffer did not complete");
    }
    return 0;
  } catch (const std::exception& native_failure) {
    report_native_failure(native_failure);
    return 1;
  } catch (...) {
    report_unknown_native_failure();
    return 1;
  }
}

extern "C" void astronomical_metal_expert_loader_free(
    astronomical_metal_expert_loader_handle* load_handle) {
  if (load_handle == nullptr) {
    return;
  }
  auto scoped_memory_pool = mlx::core::metal::new_scoped_memory_pool();
  bool did_publish_completion = false;
  {
    std::unique_lock completion_lock(load_handle->completion_state->mutex);
    did_publish_completion =
        load_handle->completion_state->completion_condition.wait_for(
            completion_lock,
            kNativeCompletionTimeout,
            [&completion_state = *load_handle->completion_state]() {
              return completion_state.has_published_completion;
            });
  }
  if (!did_publish_completion) {
    // A timed-out callback may still access handle-owned native resources.
    // Retaining the small handle is safer than racing asynchronous completion.
    return;
  }
  delete load_handle;
}
