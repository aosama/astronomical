// Core ML Neural Engine inference for the expert-route predictor.
//
// Compute units are CPU_AND_NE only. ALL is forbidden because it would
// allow GPU slices and fight the MLX decode stream. Failures return NULL
// or -1 so the worker can keep the CPU predictor.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <stdint.h>
#import <stdlib.h>
#import <string.h>
#import <time.h>

typedef struct AstronomicalPredictorAne {
  MLModel *model;
  MLMultiArray *input_array;
  MLMultiArray *output_array;
  NSString *input_name;
  NSString *output_name;
  int convolutions_on_neural_engine;
  int layer_count;
  int input_dim;
  int expert_count;
  dispatch_semaphore_t harvest_sema;
  int in_flight;
  int async_status;
  uint64_t async_elapsed_ns;
  float *async_logits;
  int async_logit_count;
  uint64_t async_started_ns;
} AstronomicalPredictorAne;

static uint64_t monotonic_nanoseconds(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void copy_c_error(char *error_message, unsigned error_message_capacity, NSString *message) {
  if (error_message == NULL || error_message_capacity == 0) {
    return;
  }
  const char *utf8_message = message.UTF8String ?: "Core ML predictor failed";
  strncpy(error_message, utf8_message, error_message_capacity - 1);
  error_message[error_message_capacity - 1] = '\0';
}

static int preferred_device_is_neural_engine(id preferred_device) {
  if (preferred_device == nil) {
    return 0;
  }
  if (@available(macOS 14.0, *)) {
    return [preferred_device isKindOfClass:[MLNeuralEngineComputeDevice class]] ? 1 : 0;
  }
  return 0;
}

static int plan_reports_neural_engine(NSURL *model_url, MLModelConfiguration *configuration) {
  if (@available(macOS 14.4, *)) {
    dispatch_semaphore_t completion = dispatch_semaphore_create(0);
    __block int convolution_on_neural_engine = 0;
    [MLComputePlan loadContentsOfURL:model_url
                       configuration:configuration
                   completionHandler:^(MLComputePlan *_Nullable plan, NSError *_Nullable error) {
                     (void)error;
                     MLModelStructureNeuralNetwork *neural_network =
                         plan.modelStructure.neuralNetwork;
                     for (MLModelStructureNeuralNetworkLayer *layer in neural_network.layers) {
                       MLComputePlanDeviceUsage *usage =
                           [plan computeDeviceUsageForNeuralNetworkLayer:layer];
                       if (preferred_device_is_neural_engine(usage.preferredComputeDevice)) {
                         convolution_on_neural_engine = 1;
                         break;
                       }
                     }
                     dispatch_semaphore_signal(completion);
                   }];
    dispatch_semaphore_wait(completion, dispatch_time(DISPATCH_TIME_NOW, 3 * NSEC_PER_SEC));
    return convolution_on_neural_engine;
  }
  return 0;
}

AstronomicalPredictorAne *astronomical_predictor_ane_load(const char *model_path,
                                                          char *error_message,
                                                          unsigned error_message_capacity) {
  @autoreleasepool {
    if (model_path == NULL) {
      copy_c_error(error_message, error_message_capacity, @"missing Core ML path");
      return NULL;
    }
    NSString *path = [NSString stringWithUTF8String:model_path];
    NSURL *model_url = [NSURL fileURLWithPath:path];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    NSError *error = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:model_url configuration:configuration error:&error];
    if (model == nil) {
      copy_c_error(error_message, error_message_capacity,
                   error.localizedDescription ?: @"failed to load Core ML predictor");
      return NULL;
    }
    AstronomicalPredictorAne *handle = calloc(1, sizeof(AstronomicalPredictorAne));
    if (handle == NULL) {
      copy_c_error(error_message, error_message_capacity, @"out of memory");
      return NULL;
    }
    handle->model = model;
    handle->input_name = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
    handle->output_name = model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
    handle->convolutions_on_neural_engine = plan_reports_neural_engine(model_url, configuration);
    handle->harvest_sema = dispatch_semaphore_create(0);
    return handle;
  }
}

int astronomical_predictor_ane_convolutions_on_neural_engine(const AstronomicalPredictorAne *handle) {
  if (handle == NULL) {
    return 0;
  }
  return handle->convolutions_on_neural_engine;
}

static int astronomical_predictor_ane_prepare_arrays(AstronomicalPredictorAne *handle,
                                                    const float *head_inputs,
                                                    int layer_count,
                                                    int input_dim,
                                                    int expert_count) {
  if (handle == NULL || handle->model == NULL || head_inputs == NULL || layer_count <= 0
      || input_dim <= 0 || expert_count <= 0) {
    return -1;
  }
  NSError *error = nil;
  const NSInteger input_channels = (NSInteger)layer_count * (NSInteger)input_dim;
  if (handle->input_array == nil || handle->layer_count != layer_count
      || handle->input_dim != input_dim) {
    handle->input_array = [[MLMultiArray alloc]
        initWithShape:@[ @1, @(input_channels), @1, @1 ]
             dataType:MLMultiArrayDataTypeFloat32
                error:&error];
    if (handle->input_array == nil) {
      return -1;
    }
    handle->layer_count = layer_count;
    handle->input_dim = input_dim;
    handle->expert_count = expert_count;
    const NSInteger output_channels = (NSInteger)layer_count * (NSInteger)expert_count;
    handle->output_array = [[MLMultiArray alloc]
        initWithShape:@[ @1, @(output_channels), @1, @1 ]
             dataType:MLMultiArrayDataTypeFloat32
                error:&error];
  }
  memcpy(handle->input_array.dataPointer, head_inputs, (size_t)input_channels * sizeof(float));
  if (handle->input_name == nil || handle->output_name == nil) {
    return -1;
  }
  return 0;
}

int astronomical_predictor_ane_predict(AstronomicalPredictorAne *handle,
                                       const float *head_inputs,
                                       int layer_count,
                                       int input_dim,
                                       float *logits_out,
                                       int expert_count) {
  @autoreleasepool {
    if (handle == NULL || handle->model == NULL || head_inputs == NULL || logits_out == NULL
        || layer_count <= 0 || input_dim <= 0 || expert_count <= 0) {
      return -1;
    }
    NSError *error = nil;
    const NSInteger input_channels = (NSInteger)layer_count * (NSInteger)input_dim;
    if (handle->input_array == nil || handle->layer_count != layer_count
        || handle->input_dim != input_dim) {
      NSArray<NSNumber *> *shape = @[ @1, @(input_channels), @1, @1 ];
      handle->input_array = [[MLMultiArray alloc] initWithShape:shape
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:&error];
      if (handle->input_array == nil) {
        return -1;
      }
      handle->layer_count = layer_count;
      handle->input_dim = input_dim;
      handle->expert_count = expert_count;
      const NSInteger output_channels = (NSInteger)layer_count * (NSInteger)expert_count;
      handle->output_array = [[MLMultiArray alloc]
          initWithShape:@[ @1, @(output_channels), @1, @1 ]
               dataType:MLMultiArrayDataTypeFloat32
                  error:&error];
    }
    memcpy(handle->input_array.dataPointer, head_inputs, (size_t)input_channels * sizeof(float));
    if (handle->input_name == nil || handle->output_name == nil) {
      return -1;
    }
    MLDictionaryFeatureProvider *provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
          handle->input_name : handle->input_array
        }
                                                         error:&error];
    if (provider == nil) {
      return -1;
    }
    MLPredictionOptions *prediction_options = [[MLPredictionOptions alloc] init];
    if (handle->output_array != nil && handle->output_name != nil) {
      prediction_options.outputBackings = @{handle->output_name : handle->output_array};
    }
    id<MLFeatureProvider> prediction =
        [handle->model predictionFromFeatures:provider options:prediction_options error:&error];
    if (prediction == nil) {
      return -1;
    }
    MLMultiArray *logits = [prediction featureValueForName:handle->output_name].multiArrayValue;
    if (logits == nil) {
      return -1;
    }
    const NSInteger logit_count = (NSInteger)layer_count * (NSInteger)expert_count;
    if (logits.count < logit_count || logits.dataPointer == NULL) {
      return -1;
    }
    memcpy(logits_out, logits.dataPointer, (size_t)logit_count * sizeof(float));
    return 0;
  }
}

int astronomical_predictor_ane_begin(AstronomicalPredictorAne *handle,
                                    const float *head_inputs,
                                    int layer_count,
                                    int input_dim,
                                    int expert_count) {
  @autoreleasepool {
    if (handle == NULL || handle->in_flight != 0) {
      return -1;
    }
    if (astronomical_predictor_ane_prepare_arrays(handle, head_inputs, layer_count, input_dim,
                                                  expert_count) != 0) {
      return -1;
    }
    NSError *error = nil;
    MLDictionaryFeatureProvider *provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
          handle->input_name : handle->input_array
        }
                                                         error:&error];
    if (provider == nil) {
      return -1;
    }
    MLPredictionOptions *prediction_options = [[MLPredictionOptions alloc] init];
    if (handle->output_array != nil && handle->output_name != nil) {
      prediction_options.outputBackings = @{handle->output_name : handle->output_array};
    }
    const int logit_count = layer_count * expert_count;
    if (handle->async_logits == NULL || handle->async_logit_count < logit_count) {
      free(handle->async_logits);
      handle->async_logits = calloc((size_t)logit_count, sizeof(float));
      handle->async_logit_count = logit_count;
      if (handle->async_logits == NULL) {
        return -1;
      }
    }
    handle->in_flight = 1;
    handle->async_status = 1;
    handle->async_started_ns = monotonic_nanoseconds();
    MLModel *model = handle->model;
    NSString *output_name = handle->output_name;
    MLMultiArray *output_array = handle->output_array;
    dispatch_semaphore_t harvest_sema = handle->harvest_sema;
    [model predictionFromFeatures:provider
                          options:prediction_options
                completionHandler:^(id<MLFeatureProvider> prediction, NSError *predict_error) {
                  (void)predict_error;
                  int status = -1;
                  if (prediction != nil) {
                    MLMultiArray *logits =
                        [prediction featureValueForName:output_name].multiArrayValue;
                    if (logits == nil) {
                      logits = output_array;
                    }
                    if (logits != nil && logits.dataPointer != NULL && handle->async_logits != NULL) {
                      memcpy(handle->async_logits, logits.dataPointer,
                             (size_t)logit_count * sizeof(float));
                      status = 0;
                    }
                  }
                  handle->async_elapsed_ns =
                      monotonic_nanoseconds() - handle->async_started_ns;
                  handle->async_status = status;
                  dispatch_semaphore_signal(harvest_sema);
                }];
    return 0;
  }
}

int astronomical_predictor_ane_harvest(AstronomicalPredictorAne *handle,
                                       float *logits_out,
                                       int logit_count,
                                       uint64_t timeout_nanoseconds,
                                       uint64_t *elapsed_nanoseconds) {
  if (handle == NULL || handle->in_flight == 0 || logits_out == NULL) {
    return -1;
  }
  const long timed_out = dispatch_semaphore_wait(
      handle->harvest_sema, dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_nanoseconds));
  if (timed_out != 0) {
    return -1;
  }
  handle->in_flight = 0;
  if (elapsed_nanoseconds != NULL) {
    *elapsed_nanoseconds = handle->async_elapsed_ns;
  }
  if (handle->async_status != 0 || handle->async_logits == NULL
      || handle->async_logit_count < logit_count) {
    return -1;
  }
  memcpy(logits_out, handle->async_logits, (size_t)logit_count * sizeof(float));
  return 0;
}

void astronomical_predictor_ane_free(AstronomicalPredictorAne *handle) {
  if (handle == NULL) {
    return;
  }
  if (handle->in_flight != 0 && handle->harvest_sema != NULL) {
    dispatch_semaphore_wait(handle->harvest_sema, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
    handle->in_flight = 0;
  }
  handle->model = nil;
  handle->input_array = nil;
  handle->output_array = nil;
  handle->input_name = nil;
  handle->output_name = nil;
  free(handle->async_logits);
  handle->async_logits = NULL;
  free(handle);
}
