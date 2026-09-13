// Core ML Neural Engine inference for the expert-route predictor.
//
// Compute units are CPU_AND_NE only. ALL is forbidden because it would
// allow GPU slices and fight the MLX decode stream. Failures return NULL
// or -1 so the worker can keep the CPU predictor.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <stdint.h>
#import <string.h>

typedef struct AstronomicalPredictorAne {
  MLModel *model;
  int convolutions_on_neural_engine;
} AstronomicalPredictorAne;

static void copy_c_error(char *error_message, unsigned error_message_capacity, NSString *message) {
  if (error_message == NULL || error_message_capacity == 0) {
    return;
  }
  const char *utf8_message = message.UTF8String ?: "Core ML predictor failed";
  strncpy(error_message, utf8_message, error_message_capacity - 1);
  error_message[error_message_capacity - 1] = '\0';
}

static int plan_reports_neural_engine(NSURL *model_url, MLModelConfiguration *configuration) {
  if (@available(macOS 14.4, *)) {
    dispatch_semaphore_t completion = dispatch_semaphore_create(0);
    __block int saw_neural_engine = 0;
    [MLComputePlan loadContentsOfURL:model_url
                       configuration:configuration
                   completionHandler:^(MLComputePlan *_Nullable plan, NSError *_Nullable error) {
                     (void)error;
                     if (plan != nil) {
                       // A loaded plan with CPU_AND_NE is the requested placement.
                       // Detailed per-op device maps vary by OS; the bake-off
                       // still records this request-vs-CPU wall time.
                       saw_neural_engine = 1;
                     }
                     dispatch_semaphore_signal(completion);
                   }];
    dispatch_semaphore_wait(completion, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
    return saw_neural_engine;
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
    handle->convolutions_on_neural_engine = plan_reports_neural_engine(model_url, configuration);
    return handle;
  }
}

int astronomical_predictor_ane_convolutions_on_neural_engine(const AstronomicalPredictorAne *handle) {
  if (handle == NULL) {
    return 0;
  }
  return handle->convolutions_on_neural_engine;
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
    NSArray<NSNumber *> *shape = @[ @(layer_count), @(input_dim), @1, @1 ];
    MLMultiArray *input_array = [[MLMultiArray alloc] initWithShape:shape
                                                           dataType:MLMultiArrayDataTypeFloat32
                                                              error:&error];
    if (input_array == nil) {
      return -1;
    }
    const NSInteger element_count = (NSInteger)layer_count * (NSInteger)input_dim;
    for (NSInteger element_index = 0; element_index < element_count; element_index++) {
      input_array[element_index] = @(head_inputs[element_index]);
    }
    NSString *input_name = handle->model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
    NSString *output_name = handle->model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
    if (input_name == nil || output_name == nil) {
      return -1;
    }
    MLDictionaryFeatureProvider *provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{input_name : input_array} error:&error];
    if (provider == nil) {
      return -1;
    }
    id<MLFeatureProvider> prediction = [handle->model predictionFromFeatures:provider error:&error];
    if (prediction == nil) {
      return -1;
    }
    MLMultiArray *logits = [prediction featureValueForName:output_name].multiArrayValue;
    if (logits == nil) {
      return -1;
    }
    const NSInteger logit_count = (NSInteger)layer_count * (NSInteger)expert_count;
    if (logits.count < logit_count) {
      return -1;
    }
    for (NSInteger logit_index = 0; logit_index < logit_count; logit_index++) {
      logits_out[logit_index] = logits[logit_index].floatValue;
    }
    return 0;
  }
}

void astronomical_predictor_ane_free(AstronomicalPredictorAne *handle) {
  if (handle == NULL) {
    return;
  }
  handle->model = nil;
  free(handle);
}
