// Core ML bridge for the Neural Engine research crate (issue #161).
//
// Only public Core ML API is used. The projection's input and output live in
// IOSurface-backed fp16 pixel buffers because that is the layout the Neural
// Engine reads and writes directly; plain heap multi-arrays would add a
// conversion and copy inside every prediction and blur the timing.
//
// Apple exposes model compilation and compute-plan loading only through
// completion handlers, so those two calls use a block plus a semaphore to
// present a synchronous C function to Rust. No other blocks are used.

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <stdint.h>
#import <stdlib.h>
#import <string.h>
#import <time.h>

enum {
  ASTRONOMICAL_PLACEMENT_UNKNOWN = 0,
  ASTRONOMICAL_PLACEMENT_CPU = 1,
  ASTRONOMICAL_PLACEMENT_GPU = 2,
  ASTRONOMICAL_PLACEMENT_NEURAL_ENGINE = 3,
};

// Compilation of a multi-megabyte model and compute-plan analysis each finish
// in well under a second on Apple silicon; a minute is a hang, not a slow Mac.
static const int64_t ASTRONOMICAL_NATIVE_WAIT_SECONDS = 60;

typedef struct {
  int32_t neural_engine_count;
  int32_t neural_engine_core_count;
  int32_t gpu_count;
  int32_t cpu_count;
} AstronomicalComputeDeviceReport;

typedef struct {
  int32_t preferred_placement;
  int32_t neural_engine_supported;
  int32_t gpu_supported;
  int32_t cpu_supported;
  int32_t operations_off_neural_engine;
} AstronomicalLayerPlacement;

typedef struct AstronomicalCoreMlProjection {
  MLModel *model;
  NSURL *compiled_model_url;
  double model_load_seconds;
  double placement_seconds;
  MLMultiArray *input_array;
  MLMultiArray *output_array;
  CVPixelBufferRef input_pixel_buffer;
  CVPixelBufferRef output_pixel_buffer;
  MLPredictionOptions *prediction_options;
  id<MLFeatureProvider> input_features;
  AstronomicalLayerPlacement placement;
  int32_t input_channels;
  int32_t output_channels;
  int32_t token_count;
  // Asynchronous prediction: one serial queue per projection, so a started
  // prediction runs off the caller's thread while the caller drives MLX.
  dispatch_queue_t prediction_queue;
  dispatch_semaphore_t prediction_finished;
  int32_t prediction_in_flight;
  int32_t asynchronous_status;
  uint64_t asynchronous_elapsed_nanoseconds;
} AstronomicalCoreMlProjection;

static uint64_t astronomical_monotonic_nanoseconds(void) {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

static void astronomical_copy_error(char *error_message, uint32_t capacity, NSString *message) {
  if (error_message == NULL || capacity == 0) {
    return;
  }
  const char *utf8_message = message.UTF8String ?: "Core ML bridge failed";
  strncpy(error_message, utf8_message, capacity - 1);
  error_message[capacity - 1] = '\0';
}

static int32_t astronomical_placement_of_device(id<MLComputeDeviceProtocol> device) {
  if ([(id)device isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
    return ASTRONOMICAL_PLACEMENT_NEURAL_ENGINE;
  }
  if ([(id)device isKindOfClass:[MLGPUComputeDevice class]]) {
    return ASTRONOMICAL_PLACEMENT_GPU;
  }
  if ([(id)device isKindOfClass:[MLCPUComputeDevice class]]) {
    return ASTRONOMICAL_PLACEMENT_CPU;
  }
  return ASTRONOMICAL_PLACEMENT_UNKNOWN;
}

int32_t astronomical_core_ml_report_compute_devices(AstronomicalComputeDeviceReport *report) {
  if (report == NULL) {
    return -1;
  }
  memset(report, 0, sizeof(*report));
  @autoreleasepool {
    for (id<MLComputeDeviceProtocol> device in MLAllComputeDevices()) {
      switch (astronomical_placement_of_device(device)) {
        case ASTRONOMICAL_PLACEMENT_NEURAL_ENGINE:
          report->neural_engine_count += 1;
          report->neural_engine_core_count +=
              (int32_t)((MLNeuralEngineComputeDevice *)device).totalCoreCount;
          break;
        case ASTRONOMICAL_PLACEMENT_GPU:
          report->gpu_count += 1;
          break;
        case ASTRONOMICAL_PLACEMENT_CPU:
          report->cpu_count += 1;
          break;
        default:
          break;
      }
    }
  }
  return 0;
}

static MLComputeUnits astronomical_compute_units(int32_t compute_units_code) {
  switch (compute_units_code) {
    case 0:
      return MLComputeUnitsCPUOnly;
    case 1:
      return MLComputeUnitsCPUAndGPU;
    case 2:
      return MLComputeUnitsCPUAndNeuralEngine;
    default:
      return MLComputeUnitsAll;
  }
}

static NSURL *astronomical_compile_model(NSURL *model_url, NSError **error_out) {
  dispatch_semaphore_t completion = dispatch_semaphore_create(0);
  __block NSURL *compiled_url = nil;
  __block NSError *compile_error = nil;
  [MLModel compileModelAtURL:model_url
           completionHandler:^(NSURL *_Nullable compiled_model_url, NSError *_Nullable error) {
             compiled_url = compiled_model_url;
             compile_error = error;
             dispatch_semaphore_signal(completion);
           }];
  long timed_out = dispatch_semaphore_wait(
      completion, dispatch_time(DISPATCH_TIME_NOW, ASTRONOMICAL_NATIVE_WAIT_SECONDS * NSEC_PER_SEC));
  if (timed_out != 0) {
    *error_out = [NSError errorWithDomain:@"AstronomicalCoreMl"
                                     code:1
                                 userInfo:@{NSLocalizedDescriptionKey : @"model compilation timed out"}];
    return nil;
  }
  *error_out = compile_error;
  return compiled_url;
}

// The projection's work is its convolution: the first network layer of a
// NeuralNetwork document, or the first convolution of an ML Program. The
// compute plan names ML Program operations with their opset ("ios18.conv"),
// and its `const` and `constexpr_*` operations report no device usage.
API_AVAILABLE(macos(14.4))
static MLComputePlanDeviceUsage *astronomical_projection_device_usage(MLComputePlan *plan) {
  MLModelStructureNeuralNetwork *network = plan.modelStructure.neuralNetwork;
  if (network != nil) {
    MLModelStructureNeuralNetworkLayer *first_layer = network.layers.firstObject;
    return first_layer == nil ? nil : [plan computeDeviceUsageForNeuralNetworkLayer:first_layer];
  }
  MLModelStructureProgramFunction *main_function = plan.modelStructure.program.functions[@"main"];
  for (MLModelStructureProgramOperation *operation in main_function.block.operations) {
    if ([operation.operatorName isEqualToString:@"conv"] || [operation.operatorName hasSuffix:@".conv"]) {
      return [plan computeDeviceUsageForMLProgramOperation:operation];
    }
  }
  return nil;
}

// A chain such as the whole MLP has work beyond its first convolution, and one
// activation left on the CPU would pass unnoticed by that convolution's verdict.
// `const` and `constexpr_*` operations report no usage, so they are not counted.
API_AVAILABLE(macos(14.4))
static int32_t astronomical_count_operations_off_neural_engine(MLComputePlan *plan) {
  MLModelStructureProgramFunction *main_function = plan.modelStructure.program.functions[@"main"];
  int32_t off_neural_engine_count = 0;
  for (MLModelStructureProgramOperation *operation in main_function.block.operations) {
    MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:operation];
    if (usage != nil
        && astronomical_placement_of_device(usage.preferredComputeDevice)
               != ASTRONOMICAL_PLACEMENT_NEURAL_ENGINE) {
      off_neural_engine_count++;
    }
  }
  return off_neural_engine_count;
}

API_AVAILABLE(macos(14.4))
static void astronomical_record_layer_placement(MLComputePlan *plan, AstronomicalLayerPlacement *placement) {
  placement->operations_off_neural_engine = astronomical_count_operations_off_neural_engine(plan);
  MLComputePlanDeviceUsage *usage = astronomical_projection_device_usage(plan);
  if (usage == nil) {
    return;
  }
  placement->preferred_placement = astronomical_placement_of_device(usage.preferredComputeDevice);
  for (id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices) {
    int32_t supported_placement = astronomical_placement_of_device(device);
    placement->neural_engine_supported |= supported_placement == ASTRONOMICAL_PLACEMENT_NEURAL_ENGINE;
    placement->gpu_supported |= supported_placement == ASTRONOMICAL_PLACEMENT_GPU;
    placement->cpu_supported |= supported_placement == ASTRONOMICAL_PLACEMENT_CPU;
  }
}

API_AVAILABLE(macos(14.4))
static int32_t astronomical_load_layer_placement(NSURL *compiled_url,
                                                 MLModelConfiguration *configuration,
                                                 AstronomicalLayerPlacement *placement) {
  dispatch_semaphore_t completion = dispatch_semaphore_create(0);
  __block MLComputePlan *loaded_plan = nil;
  [MLComputePlan loadContentsOfURL:compiled_url
                     configuration:configuration
                 completionHandler:^(MLComputePlan *_Nullable plan, NSError *_Nullable error) {
                   (void)error;
                   loaded_plan = plan;
                   dispatch_semaphore_signal(completion);
                 }];
  long timed_out = dispatch_semaphore_wait(
      completion, dispatch_time(DISPATCH_TIME_NOW, ASTRONOMICAL_NATIVE_WAIT_SECONDS * NSEC_PER_SEC));
  if (timed_out != 0 || loaded_plan == nil) {
    return -1;
  }
  astronomical_record_layer_placement(loaded_plan, placement);
  return 0;
}

static CVPixelBufferRef astronomical_create_fp16_surface(int32_t row_count, int32_t column_count) {
  NSDictionary *attributes = @{(NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{}};
  CVPixelBufferRef pixel_buffer = NULL;
  CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, (size_t)column_count, (size_t)row_count,
                                        kCVPixelFormatType_OneComponent16Half,
                                        (__bridge CFDictionaryRef)attributes, &pixel_buffer);
  return status == kCVReturnSuccess ? pixel_buffer : NULL;
}

static MLMultiArray *astronomical_channels_by_tokens_array(CVPixelBufferRef pixel_buffer,
                                                           int32_t channel_count,
                                                           int32_t token_count) {
  // Shape [batch 1, channels, height 1, tokens]: the pixel buffer's rows are
  // channels and its columns are tokens, exactly the 1x1-convolution layout.
  return [[MLMultiArray alloc] initWithPixelBuffer:pixel_buffer
                                             shape:@[ @1, @(channel_count), @1, @(token_count) ]];
}

static int32_t astronomical_bind_surfaces(AstronomicalCoreMlProjection *projection, NSError **error_out) {
  projection->input_pixel_buffer =
      astronomical_create_fp16_surface(projection->input_channels, projection->token_count);
  projection->output_pixel_buffer =
      astronomical_create_fp16_surface(projection->output_channels, projection->token_count);
  if (projection->input_pixel_buffer == NULL || projection->output_pixel_buffer == NULL) {
    return -1;
  }
  projection->input_array = astronomical_channels_by_tokens_array(
      projection->input_pixel_buffer, projection->input_channels, projection->token_count);
  projection->output_array = astronomical_channels_by_tokens_array(
      projection->output_pixel_buffer, projection->output_channels, projection->token_count);
  projection->input_features = [[MLDictionaryFeatureProvider alloc]
      initWithDictionary:@{@"activations" : projection->input_array}
                   error:error_out];
  projection->prediction_options = [[MLPredictionOptions alloc] init];
  projection->prediction_options.outputBackings = @{@"projected" : projection->output_array};
  return projection->input_features == nil ? -1 : 0;
}

double astronomical_core_ml_projection_model_load_seconds(
    const AstronomicalCoreMlProjection *projection) {
  return projection == NULL ? 0.0 : projection->model_load_seconds;
}

double astronomical_core_ml_projection_placement_seconds(
    const AstronomicalCoreMlProjection *projection) {
  return projection == NULL ? 0.0 : projection->placement_seconds;
}

int32_t astronomical_core_ml_projection_compiled_model_path(
    AstronomicalCoreMlProjection *projection, char *path_buffer, uint32_t path_capacity) {
  if (projection == NULL || projection->compiled_model_url == nil || path_capacity == 0) {
    return -1;
  }
  const char *compiled_path = projection->compiled_model_url.fileSystemRepresentation;
  if (compiled_path == NULL || strlen(compiled_path) >= path_capacity) {
    return -1;
  }
  strlcpy(path_buffer, compiled_path, path_capacity);
  return 0;
}

void astronomical_core_ml_projection_discard_compiled_model(
    AstronomicalCoreMlProjection *projection) {
  if (projection == NULL || projection->compiled_model_url == nil) {
    return;
  }
  [[NSFileManager defaultManager] removeItemAtURL:projection->compiled_model_url error:nil];
  projection->compiled_model_url = nil;
}

void astronomical_core_ml_projection_release(AstronomicalCoreMlProjection *projection) {
  if (projection == NULL) {
    return;
  }
  // A prediction still running would write into surfaces about to be freed.
  if (projection->prediction_in_flight) {
    dispatch_semaphore_wait(projection->prediction_finished, DISPATCH_TIME_FOREVER);
    projection->prediction_in_flight = 0;
  }
  projection->prediction_queue = nil;
  projection->prediction_finished = nil;
  if (projection->compiled_model_url != nil) {
    [[NSFileManager defaultManager] removeItemAtURL:projection->compiled_model_url error:nil];
  }
  CVPixelBufferRelease(projection->input_pixel_buffer);
  CVPixelBufferRelease(projection->output_pixel_buffer);
  // Reassigning the ARC-managed members to nil releases the Objective-C
  // objects before the C allocation that holds them is freed.
  projection->model = nil;
  projection->compiled_model_url = nil;
  projection->input_array = nil;
  projection->output_array = nil;
  projection->prediction_options = nil;
  projection->input_features = nil;
  free(projection);
}

static int32_t astronomical_load_model(AstronomicalCoreMlProjection *projection, NSURL *model_url,
                                       int32_t compute_units_code, NSError **error_out) {
  // An already compiled program belongs to whoever kept it, so it is loaded where it
  // is and `compiled_model_url` stays nil: release must not remove a cached copy.
  NSURL *compiled_url = model_url;
  if (![model_url.pathExtension isEqualToString:@"mlmodelc"]) {
    projection->compiled_model_url = astronomical_compile_model(model_url, error_out);
    if (projection->compiled_model_url == nil) {
      return -1;
    }
    compiled_url = projection->compiled_model_url;
  }
  MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
  configuration.computeUnits = astronomical_compute_units(compute_units_code);
  CFAbsoluteTime model_load_started_at = CFAbsoluteTimeGetCurrent();
  projection->model = [MLModel modelWithContentsOfURL:compiled_url
                                        configuration:configuration
                                                error:error_out];
  projection->model_load_seconds = CFAbsoluteTimeGetCurrent() - model_load_started_at;
  if (projection->model == nil) {
    return -1;
  }
  // Before 14.4 there is no compute plan, so placement stays unknown; callers
  // that need proof of Neural Engine placement refuse an unknown verdict.
  if (@available(macOS 14.4, *)) {
    CFAbsoluteTime placement_started_at = CFAbsoluteTimeGetCurrent();
    int32_t placement_status =
        astronomical_load_layer_placement(compiled_url, configuration, &projection->placement);
    projection->placement_seconds = CFAbsoluteTimeGetCurrent() - placement_started_at;
    return placement_status;
  }
  return 0;
}

AstronomicalCoreMlProjection *astronomical_core_ml_projection_load(
    const char *model_path, int32_t compute_units_code, int32_t input_channels,
    int32_t output_channels, int32_t token_count, char *error_message, uint32_t error_capacity) {
  if (model_path == NULL || input_channels <= 0 || output_channels <= 0 || token_count <= 0) {
    astronomical_copy_error(error_message, error_capacity, @"invalid projection arguments");
    return NULL;
  }
  @autoreleasepool {
    AstronomicalCoreMlProjection *projection = calloc(1, sizeof(AstronomicalCoreMlProjection));
    if (projection == NULL) {
      astronomical_copy_error(error_message, error_capacity, @"out of memory");
      return NULL;
    }
    projection->input_channels = input_channels;
    projection->output_channels = output_channels;
    projection->token_count = token_count;
    projection->prediction_queue =
        dispatch_queue_create("astronomical.core-ml.projection", DISPATCH_QUEUE_SERIAL);
    projection->prediction_finished = dispatch_semaphore_create(0);
    NSError *error = nil;
    NSURL *model_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:model_path]];
    if (astronomical_load_model(projection, model_url, compute_units_code, &error) != 0
        || astronomical_bind_surfaces(projection, &error) != 0) {
      astronomical_copy_error(error_message, error_capacity,
                              error.localizedDescription ?: @"failed to prepare the Core ML projection");
      astronomical_core_ml_projection_release(projection);
      return NULL;
    }
    return projection;
  }
}

AstronomicalLayerPlacement astronomical_core_ml_projection_placement(
    const AstronomicalCoreMlProjection *projection) {
  AstronomicalLayerPlacement unknown_placement = {0, 0, 0, 0, 0};
  return projection == NULL ? unknown_placement : projection->placement;
}

static int32_t astronomical_copy_surface_rows(CVPixelBufferRef pixel_buffer, uint16_t *host_values,
                                              int32_t row_count, int32_t column_count,
                                              int32_t copy_into_surface) {
  CVPixelBufferLockFlags lock_flags = copy_into_surface ? 0 : kCVPixelBufferLock_ReadOnly;
  if (CVPixelBufferLockBaseAddress(pixel_buffer, lock_flags) != kCVReturnSuccess) {
    return -1;
  }
  uint8_t *surface_base = CVPixelBufferGetBaseAddress(pixel_buffer);
  // Rows are padded to the surface's alignment, so each row is copied on its
  // own instead of as one contiguous block.
  size_t surface_row_bytes = CVPixelBufferGetBytesPerRow(pixel_buffer);
  size_t host_row_bytes = (size_t)column_count * sizeof(uint16_t);
  for (int32_t row_index = 0; row_index < row_count; row_index++) {
    uint8_t *surface_row = surface_base + (size_t)row_index * surface_row_bytes;
    uint16_t *host_row = host_values + (size_t)row_index * (size_t)column_count;
    if (copy_into_surface) {
      memcpy(surface_row, host_row, host_row_bytes);
    } else {
      memcpy(host_row, surface_row, host_row_bytes);
    }
  }
  CVPixelBufferUnlockBaseAddress(pixel_buffer, lock_flags);
  return 0;
}

int32_t astronomical_core_ml_projection_write_input(AstronomicalCoreMlProjection *projection,
                                                    const uint16_t *fp16_activations,
                                                    uint64_t element_count) {
  if (projection == NULL || fp16_activations == NULL || projection->prediction_in_flight
      || element_count != (uint64_t)projection->input_channels * (uint64_t)projection->token_count) {
    return -1;
  }
  return astronomical_copy_surface_rows(projection->input_pixel_buffer, (uint16_t *)fp16_activations,
                                        projection->input_channels, projection->token_count, 1);
}

int32_t astronomical_core_ml_projection_read_output(const AstronomicalCoreMlProjection *projection,
                                                    uint16_t *fp16_projected,
                                                    uint64_t element_count) {
  if (projection == NULL || fp16_projected == NULL || projection->prediction_in_flight
      || element_count != (uint64_t)projection->output_channels * (uint64_t)projection->token_count) {
    return -1;
  }
  return astronomical_copy_surface_rows(projection->output_pixel_buffer, fp16_projected,
                                        projection->output_channels, projection->token_count, 0);
}

int32_t astronomical_core_ml_projection_lock_output(const AstronomicalCoreMlProjection *projection,
                                                    const void **base_address,
                                                    uint64_t *bytes_per_row) {
  if (projection == NULL || base_address == NULL || bytes_per_row == NULL
      || projection->prediction_in_flight) {
    return -1;
  }
  if (CVPixelBufferLockBaseAddress(projection->output_pixel_buffer, kCVPixelBufferLock_ReadOnly)
      != kCVReturnSuccess) {
    return -1;
  }
  *base_address = CVPixelBufferGetBaseAddress(projection->output_pixel_buffer);
  *bytes_per_row = (uint64_t)CVPixelBufferGetBytesPerRow(projection->output_pixel_buffer);
  return 0;
}

void astronomical_core_ml_projection_unlock_output(const AstronomicalCoreMlProjection *projection) {
  if (projection == NULL) {
    return;
  }
  CVPixelBufferUnlockBaseAddress(projection->output_pixel_buffer, kCVPixelBufferLock_ReadOnly);
}

static int32_t astronomical_run_prediction(AstronomicalCoreMlProjection *projection,
                                           uint64_t *elapsed_nanoseconds, NSError **error_out) {
  uint64_t started_at = astronomical_monotonic_nanoseconds();
  id<MLFeatureProvider> prediction = [projection->model predictionFromFeatures:projection->input_features
                                                                       options:projection->prediction_options
                                                                         error:error_out];
  *elapsed_nanoseconds = astronomical_monotonic_nanoseconds() - started_at;
  return prediction == nil ? -1 : 0;
}

int32_t astronomical_core_ml_projection_predict(AstronomicalCoreMlProjection *projection,
                                                uint64_t *elapsed_nanoseconds,
                                                char *error_message, uint32_t error_capacity) {
  if (projection == NULL || elapsed_nanoseconds == NULL || projection->prediction_in_flight) {
    astronomical_copy_error(error_message, error_capacity, @"invalid prediction arguments");
    return -1;
  }
  @autoreleasepool {
    NSError *error = nil;
    if (astronomical_run_prediction(projection, elapsed_nanoseconds, &error) != 0) {
      astronomical_copy_error(error_message, error_capacity,
                              error.localizedDescription ?: @"Core ML prediction failed");
      return -1;
    }
    return 0;
  }
}

// Runs on the projection's serial queue; dispatch_async_f takes a plain C
// function and a context pointer, so no block is needed here.
static void astronomical_run_asynchronous_prediction(void *context) {
  AstronomicalCoreMlProjection *projection = context;
  @autoreleasepool {
    NSError *error = nil;
    projection->asynchronous_status =
        astronomical_run_prediction(projection, &projection->asynchronous_elapsed_nanoseconds, &error);
  }
  dispatch_semaphore_signal(projection->prediction_finished);
}

int32_t astronomical_core_ml_projection_begin_predict(AstronomicalCoreMlProjection *projection) {
  if (projection == NULL || projection->prediction_in_flight) {
    return -1;
  }
  projection->prediction_in_flight = 1;
  dispatch_async_f(projection->prediction_queue, projection, astronomical_run_asynchronous_prediction);
  return 0;
}

// Returns 0 when the started prediction succeeded, -1 when it failed, and -2
// when it is still running after the timeout (it stays in flight).
int32_t astronomical_core_ml_projection_finish_predict(AstronomicalCoreMlProjection *projection,
                                                       uint64_t timeout_nanoseconds,
                                                       uint64_t *elapsed_nanoseconds) {
  if (projection == NULL || elapsed_nanoseconds == NULL || !projection->prediction_in_flight) {
    return -1;
  }
  // A timeout beyond the signed range means "wait without limit"; casting it
  // would wrap negative and time out immediately.
  dispatch_time_t deadline = timeout_nanoseconds > (uint64_t)INT64_MAX
                                 ? DISPATCH_TIME_FOREVER
                                 : dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_nanoseconds);
  if (dispatch_semaphore_wait(projection->prediction_finished, deadline) != 0) {
    return -2;
  }
  projection->prediction_in_flight = 0;
  *elapsed_nanoseconds = projection->asynchronous_elapsed_nanoseconds;
  return projection->asynchronous_status;
}

// Compiles a package and leaves the result at `destination_path` without loading
// it. macOS keeps its own Neural Engine compilation keyed by the compiled model's
// path, so a program meant to be kept must be loaded for the first time where it
// will stay; loading it from the temporary directory first would pay that cost
// twice.
int32_t astronomical_core_ml_compile_model_to(const char *package_path, const char *destination_path,
                                              char *error_message, uint32_t error_capacity) {
  @autoreleasepool {
    NSError *error = nil;
    NSURL *package_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:package_path]];
    NSURL *destination_url =
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:destination_path]];
    NSURL *compiled_url = astronomical_compile_model(package_url, &error);
    if (compiled_url == nil) {
      astronomical_copy_error(error_message, error_capacity,
                              error.localizedDescription ?: @"failed to compile the Core ML model");
      return -1;
    }
    NSFileManager *file_manager = [NSFileManager defaultManager];
    [file_manager removeItemAtURL:destination_url error:nil];
    if (![file_manager moveItemAtURL:compiled_url toURL:destination_url error:&error]) {
      // Only the compiled model is ours to remove: Core ML chooses the folder it
      // sits in, and that folder can be shared with everything else in the
      // temporary directory.
      [file_manager removeItemAtURL:compiled_url error:nil];
      astronomical_copy_error(error_message, error_capacity,
                              error.localizedDescription ?: @"failed to keep the compiled model");
      return -1;
    }
    return 0;
  }
}

