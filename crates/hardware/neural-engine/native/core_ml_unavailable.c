// The Core ML bridge for targets without Core ML.
//
// Core ML, the Neural Engine and the system's wired-memory counter exist only
// on macOS. On every other target the crate links these definitions instead of
// core_ml_bridge.m and system_wired_memory.c, so its Rust API and every caller
// stay the same: the device report finds no Neural Engine, loading or
// compiling a program fails with a message saying why, and the wired-memory
// reading is zero, the value the Rust side already treats as "unavailable".
// No program can ever be loaded, so the per-program entry points are never
// reached; they return failure for completeness.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char kCoreMlUnavailableMessage[] =
    "Core ML and the Neural Engine are available only on macOS";

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

static void write_unavailable_message(char *error_message, uint32_t error_capacity) {
  if (error_message != NULL && error_capacity > 0) {
    snprintf(error_message, error_capacity, "%s", kCoreMlUnavailableMessage);
  }
}

int32_t astronomical_core_ml_report_compute_devices(AstronomicalComputeDeviceReport *report) {
  if (report != NULL) {
    memset(report, 0, sizeof(*report));
  }
  return -1;
}

void *astronomical_core_ml_program_load(const char *model_path, int32_t compute_units_code,
                                        int32_t checks_placement, int32_t input_channels,
                                        int32_t output_channels, int32_t token_count,
                                        char *error_message, uint32_t error_capacity) {
  (void)model_path;
  (void)compute_units_code;
  (void)checks_placement;
  (void)input_channels;
  (void)output_channels;
  (void)token_count;
  write_unavailable_message(error_message, error_capacity);
  return NULL;
}

int32_t astronomical_core_ml_compile_model_to(const char *package_path,
                                              const char *destination_path, char *error_message,
                                              uint32_t error_capacity) {
  (void)package_path;
  (void)destination_path;
  write_unavailable_message(error_message, error_capacity);
  return -1;
}

AstronomicalLayerPlacement astronomical_core_ml_program_placement(const void *program) {
  (void)program;
  AstronomicalLayerPlacement no_placement;
  memset(&no_placement, 0, sizeof(no_placement));
  return no_placement;
}

int32_t astronomical_core_ml_program_write_input(void *program, const uint16_t *fp16_activations,
                                                 uint64_t element_count) {
  (void)program;
  (void)fp16_activations;
  (void)element_count;
  return -1;
}

int32_t astronomical_core_ml_program_read_output(const void *program, uint16_t *fp16_projected,
                                                 uint64_t element_count) {
  (void)program;
  (void)fp16_projected;
  (void)element_count;
  return -1;
}

int32_t astronomical_core_ml_program_lock_output(const void *program, const void **base_address,
                                                 uint64_t *bytes_per_row) {
  (void)program;
  (void)base_address;
  (void)bytes_per_row;
  return -1;
}

void astronomical_core_ml_program_unlock_output(const void *program) { (void)program; }

int32_t astronomical_core_ml_program_predict(void *program, uint64_t *elapsed_nanoseconds,
                                             char *error_message, uint32_t error_capacity) {
  (void)program;
  (void)elapsed_nanoseconds;
  write_unavailable_message(error_message, error_capacity);
  return -1;
}

int32_t astronomical_core_ml_program_begin_predict(void *program) {
  (void)program;
  return -1;
}

int32_t astronomical_core_ml_program_finish_predict(void *program, uint64_t timeout_nanoseconds,
                                                    uint64_t *elapsed_nanoseconds) {
  (void)program;
  (void)timeout_nanoseconds;
  (void)elapsed_nanoseconds;
  return -1;
}

void astronomical_core_ml_program_copy_asynchronous_error(const void *program, char *error_buffer,
                                                          uint32_t error_capacity) {
  (void)program;
  write_unavailable_message(error_buffer, error_capacity);
}

double astronomical_core_ml_program_model_load_seconds(const void *program) {
  (void)program;
  return 0.0;
}

double astronomical_core_ml_program_placement_seconds(const void *program) {
  (void)program;
  return 0.0;
}

int32_t astronomical_core_ml_program_compiled_model_path(void *program, char *path_buffer,
                                                         uint32_t path_capacity) {
  (void)program;
  if (path_buffer != NULL && path_capacity > 0) {
    path_buffer[0] = '\0';
  }
  return -1;
}

void astronomical_core_ml_program_discard_compiled_model(void *program) { (void)program; }

void astronomical_core_ml_program_release(void *program) { (void)program; }

uint64_t astronomical_system_wired_bytes(void) { return 0; }
