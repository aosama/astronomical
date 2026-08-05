#ifndef ASTRONOMICAL_METAL_EXPERT_LOADER_TEST_SUPPORT_H
#define ASTRONOMICAL_METAL_EXPERT_LOADER_TEST_SUPPORT_H

#include "astronomical_metal_expert_loader.h"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mlx/c/memory.h"
#include "mlx/c/ops.h"
#include "mlx/c/transforms.h"
#include "mlx/c/vector.h"

#include <unistd.h>

inline void require_condition(bool condition, const std::string& description) {
  if (!condition) {
    throw std::runtime_error(description);
  }
}

inline void require_mlx_success(int status, const std::string& operation) {
  require_condition(status == 0, operation + " failed");
}

inline void free_output_arrays(std::vector<mlx_array>& output_arrays) {
  for (auto& output_array : output_arrays) {
    if (output_array.ctx != nullptr) {
      mlx_array_free(output_array);
      output_array = mlx_array_new();
    }
  }
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::string directory_template =
        (std::filesystem::temp_directory_path() / "astronomical-native-metal-XXXXXX")
            .string();
    directory_template.push_back('\0');
    if (mkdtemp(directory_template.data()) == nullptr) {
      throw std::runtime_error("could not create a native test temporary directory");
    }
    directory_path_ = directory_template.data();
  }

  ~TemporaryDirectory() {
    std::error_code removal_error;
    std::filesystem::remove_all(directory_path_, removal_error);
  }

  const std::filesystem::path& path() const {
    return directory_path_;
  }

 private:
  std::filesystem::path directory_path_;
};

class NativeStreamOwner {
 public:
  NativeStreamOwner()
      : device_(mlx_device_new_type(MLX_GPU, 0)),
        stream_(mlx_stream_new_device(device_)) {
    if (device_.ctx == nullptr) {
      throw std::runtime_error("could not create the native MLX GPU device");
    }
    if (stream_.ctx == nullptr) {
      throw std::runtime_error("could not create the native MLX GPU stream");
    }
  }

  ~NativeStreamOwner() {
    if (stream_.ctx != nullptr) {
      mlx_stream_free(stream_);
    }
    if (device_.ctx != nullptr) {
      mlx_device_free(device_);
    }
  }

  mlx_stream get() const {
    return stream_;
  }

 private:
  mlx_device device_;
  mlx_stream stream_;
};

class OperationDeadline {
 public:
  OperationDeadline(std::string operation_name, std::chrono::seconds timeout)
      : operation_name_(std::move(operation_name)),
        watchdog_thread_([this, timeout]() {
          std::unique_lock deadline_lock(deadline_mutex_);
          if (!deadline_condition_.wait_for(
                  deadline_lock, timeout, [this]() { return has_completed_; })) {
            std::cerr
                << "[native-metal-expert-loader] status=error reason=operation_timeout operation="
                << operation_name_ << " timeout_seconds=" << timeout.count()
                << std::endl;
            _Exit(124);
          }
        }) {}

  ~OperationDeadline() {
    {
      std::lock_guard deadline_lock(deadline_mutex_);
      has_completed_ = true;
    }
    deadline_condition_.notify_all();
    watchdog_thread_.join();
  }

 private:
  std::string operation_name_;
  std::mutex deadline_mutex_;
  std::condition_variable deadline_condition_;
  bool has_completed_{false};
  std::thread watchdog_thread_;
};

class CompleteTransactionOwner {
 public:
  CompleteTransactionOwner(
      const std::filesystem::path& source_pack_path,
      const std::vector<astronomical_metal_expert_loader_output_tensor>& output_tensors,
      const std::vector<astronomical_metal_expert_loader_load_range>& load_ranges,
      mlx_stream gpu_stream,
      const std::string& transaction_name,
      bool emit_detailed_progress = true)
      : gpu_stream_(gpu_stream),
        output_arrays_(output_tensors.size(), mlx_array_new()),
        transaction_started_at_(std::chrono::steady_clock::now()),
        transaction_name_(transaction_name),
        emit_detailed_progress_(emit_detailed_progress) {
    if (emit_detailed_progress_) {
      std::cout << "[native-metal-expert-loader] status=progress transaction="
                << transaction_name_ << " phase=io_submit" << std::endl;
    }
    int submission_status = 1;
    {
      OperationDeadline submission_deadline(
          transaction_name_ + "_io_submission", std::chrono::seconds(10));
      submission_status = astronomical_metal_expert_loader_start(
          source_pack_path.c_str(),
          output_tensors.data(),
          output_tensors.size(),
          load_ranges.data(),
          load_ranges.size(),
          gpu_stream_,
          output_arrays_.data(),
          &load_handle_,
          nullptr,
          nullptr,
          nullptr,
          nullptr);
    }
    require_condition(
        submission_status == 0 && load_handle_ != nullptr,
        "native Metal I/O transaction submission failed");
  }

  ~CompleteTransactionOwner() {
    try {
      release();
    } catch (...) {
    }
  }

  void consume_all_outputs_on_gpu() {
    require_condition(load_handle_ != nullptr, "native transaction has already released its loader");
    std::vector<mlx_array> doubled_output_arrays;
    doubled_output_arrays.reserve(output_arrays_.size());
    for (const auto output_array : output_arrays_) {
      mlx_array doubled_output_array = mlx_array_new();
      const auto addition_status = mlx_add(
          &doubled_output_array,
          output_array,
          output_array,
          gpu_stream_);
      if (addition_status != 0) {
        if (doubled_output_array.ctx != nullptr) {
          mlx_array_free(doubled_output_array);
        }
        free_output_arrays(doubled_output_arrays);
        throw std::runtime_error("build native GPU consumer failed");
      }
      doubled_output_arrays.push_back(doubled_output_array);
    }
    mlx_vector_array doubled_output_vector = mlx_vector_array_new_data(
        doubled_output_arrays.data(),
        doubled_output_arrays.size());
    require_condition(
        doubled_output_vector.ctx != nullptr,
        "could not build native GPU consumer output vector");
    if (emit_detailed_progress_) {
      std::cout << "[native-metal-expert-loader] status=progress transaction="
                << transaction_name_ << " phase=gpu_graph_built" << std::endl;
    }
    {
      OperationDeadline evaluation_deadline(
          transaction_name_ + "_gpu_evaluation", std::chrono::seconds(10));
      require_mlx_success(
          mlx_eval(doubled_output_vector),
          "evaluate native GPU consumer");
    }
    if (emit_detailed_progress_) {
      std::cout << "[native-metal-expert-loader] status=progress transaction="
                << transaction_name_ << " phase=gpu_evaluated" << std::endl;
    }
    {
      OperationDeadline synchronization_deadline(
          transaction_name_ + "_gpu_synchronization", std::chrono::seconds(10));
      require_mlx_success(
          mlx_synchronize(gpu_stream_),
          "synchronize native GPU consumer");
    }
    if (emit_detailed_progress_) {
      std::cout << "[native-metal-expert-loader] status=progress transaction="
                << transaction_name_ << " phase=gpu_synchronized" << std::endl;
    }
    require_mlx_success(
        mlx_vector_array_free(doubled_output_vector),
        "free native GPU consumer output vector");
    free_output_arrays(doubled_output_arrays);
    if (emit_detailed_progress_) {
      std::cout << "[native-metal-expert-loader] status=progress transaction="
                << transaction_name_ << " phase=gpu_consumed" << std::endl;
    }
  }

  void wait_for_io_completion() {
    require_condition(load_handle_ != nullptr, "native transaction has already released its loader");
    {
      OperationDeadline completion_deadline(
          transaction_name_ + "_io_completion", std::chrono::seconds(10));
      require_mlx_success(
          astronomical_metal_expert_loader_wait(load_handle_, &io_metrics_),
          "read native Metal I/O completion metrics");
    }
    require_condition(
        io_metrics_.final_status == 3,
        "native Metal I/O transaction finished with a non-complete status");
    has_completion_metrics_ = true;
    if (emit_detailed_progress_) {
      std::cout << "[native-metal-expert-loader] status=progress transaction="
                << transaction_name_ << " phase=io_completed" << std::endl;
    }
  }

  const std::vector<mlx_array>& output_arrays() const {
    return output_arrays_;
  }

  const astronomical_metal_expert_loader_metrics& io_metrics() const {
    require_condition(has_completion_metrics_, "native transaction metrics were not collected");
    return io_metrics_;
  }

  uint64_t elapsed_nanoseconds() const {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - transaction_started_at_)
            .count());
  }

  void release() {
    if (is_released_) {
      return;
    }
    if (load_handle_ != nullptr) {
      OperationDeadline release_deadline(
          transaction_name_ + "_native_release", std::chrono::seconds(10));
      astronomical_metal_expert_loader_free(load_handle_);
      load_handle_ = nullptr;
    }
    free_output_arrays(output_arrays_);
    require_mlx_success(mlx_clear_cache(), "clear native MLX allocator cache");
    is_released_ = true;
    if (emit_detailed_progress_) {
      std::cout << "[native-metal-expert-loader] status=progress transaction="
                << transaction_name_ << " phase=cleanup_complete" << std::endl;
    }
  }

 private:
  mlx_stream gpu_stream_;
  std::vector<mlx_array> output_arrays_;
  astronomical_metal_expert_loader_handle* load_handle_ = nullptr;
  astronomical_metal_expert_loader_metrics io_metrics_{};
  bool has_completion_metrics_ = false;
  bool is_released_ = false;
  std::chrono::steady_clock::time_point transaction_started_at_;
  std::string transaction_name_;
  bool emit_detailed_progress_;
};

#endif
