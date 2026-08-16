use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    MlxRuntimeError,
};

use super::decoder_layer_weights::Qwen3_5AffineWeights;
use super::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;

// Adapted from mlx-vlm's exact Qwen3.5 target-verification QMV kernel.
const TARGET_VERIFICATION_QUANTIZED_LINEAR_HEADER: &str = r#"
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int NUM_SIMDGROUPS = 2;
constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

// MLX inserts custom-kernel template arguments inside the generated kernel
// function, so header helpers must receive quantization geometry explicitly
// rather than referring to BITS or GS from global scope.
template <typename T, int Bits>
inline float load_vector_exact(const device T* x, thread float* x_thread) {
  constexpr int pack_factor = (Bits == 5 ? 8 : 32 / Bits);
  constexpr int values_per_thread = pack_factor * PACKS_PER_THREAD;
  float sum = 0.0f;
  if constexpr (Bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if constexpr (Bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] +
          x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }
  return sum;
}

template <int Bits>
inline float qdot_exact(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  constexpr int pack_factor = (Bits == 5 ? 8 : 32 / Bits);
  constexpr int values_per_thread = pack_factor * PACKS_PER_THREAD;
  if constexpr (Bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          x_thread[4 * i] * (ws[i] & 0x000f) +
          x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
          x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
          x_thread[4 * i + 3] * (ws[i] & 0xf000);
    }
  } else if constexpr (Bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      const thread float* xt = x_thread + 8 * i;
      const device uint8_t* wb = w + 5 * i;
      accum += (wb[0] & 0x1f) * xt[0];
      accum += (wb[0] & 0xe0) * xt[1];
      accum += (wb[1] & 0x3) * (xt[1] * 256.0f);
      accum += (wb[1] & 0x7c) * xt[2];
      accum += (wb[1] & 0x80) * xt[3];
      accum += (wb[2] & 0xf) * (xt[3] * 256.0f);
      accum += (wb[2] & 0xf0) * xt[4];
      accum += (wb[3] & 0x1) * (xt[4] * 256.0f);
      accum += (wb[3] & 0x3e) * xt[5];
      accum += (wb[3] & 0xc0) * xt[6];
      accum += (wb[4] & 0x7) * (xt[6] * 256.0f);
      accum += (wb[4] & 0xf8) * xt[7];
    }
  }
  return scale * accum + sum * bias;
}
"#;

const TARGET_VERIFICATION_QUANTIZED_LINEAR_SOURCE: &str = r#"
constexpr int PACK_FACTOR = (BITS == 5 ? 8 : 32 / BITS);
constexpr int BYTES_PER_PACK = (BITS == 5 ? 5 : 32 / 8);
constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;

uint n_tile = threadgroup_position_in_grid.y;
uint batch_index = threadgroup_position_in_grid.z;
uint simd_group_index = simdgroup_index_in_threadgroup;
uint simd_lane_index = thread_index_in_simdgroup;

int output_row = int(n_tile) * BN + int(simd_group_index) * RESULTS_PER_SIMDGROUP;
int packed_input_width_bytes = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
int affine_group_count = K_SIZE / GS;

const device uint8_t* packed_weights_base =
    (const device uint8_t*)packed_weights + output_row * packed_input_width_bytes +
    int(simd_lane_index) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* scales_base =
    quantization_scales + output_row * affine_group_count +
    int(simd_lane_index) / SCALE_STEP_PER_THREAD;
const device T* biases_base =
    quantization_biases + output_row * affine_group_count +
    int(simd_lane_index) / SCALE_STEP_PER_THREAD;
const device T* activations_base =
    activations + int(batch_index) * VERIFY_T * K_SIZE +
    int(simd_lane_index) * VALUES_PER_THREAD;

float projection_sums[VERIFY_T][RESULTS_PER_SIMDGROUP];
float activation_fragments[VERIFY_T][VALUES_PER_THREAD];
for (int token_position_index = 0; token_position_index < VERIFY_T; ++token_position_index) {
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    projection_sums[token_position_index][row] = 0.0f;
  }
}

const device uint8_t* current_packed_weights = packed_weights_base;
const device T* current_scales = scales_base;
const device T* current_biases = biases_base;
const device T* current_activations = activations_base;
for (int input_dimension_offset = 0; input_dimension_offset < K_SIZE;
     input_dimension_offset += BLOCK_SIZE) {
  float activation_sums[VERIFY_T];
  for (int token_position_index = 0; token_position_index < VERIFY_T; ++token_position_index) {
    activation_sums[token_position_index] = load_vector_exact<T, BITS>(
        current_activations + token_position_index * K_SIZE,
        activation_fragments[token_position_index]);
  }
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    const device uint8_t* row_packed_weights =
        current_packed_weights + row * packed_input_width_bytes;
    const device T* row_scales = current_scales + row * affine_group_count;
    const device T* row_biases = current_biases + row * affine_group_count;
    float quantization_scale = float(row_scales[0]);
    float quantization_bias = float(row_biases[0]);
    for (int token_position_index = 0; token_position_index < VERIFY_T;
         ++token_position_index) {
      projection_sums[token_position_index][row] += qdot_exact<BITS>(
          row_packed_weights,
          activation_fragments[token_position_index],
          quantization_scale,
          quantization_bias,
          activation_sums[token_position_index]);
    }
  }
  current_packed_weights += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
  current_scales += BLOCK_SIZE / GS;
  current_biases += BLOCK_SIZE / GS;
  current_activations += BLOCK_SIZE;
}

for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  int output_dimension_index = output_row + row;
  for (int token_position_index = 0; token_position_index < VERIFY_T;
       ++token_position_index) {
    float projection_sum = simd_sum(projection_sums[token_position_index][row]);
    if (simd_lane_index == 0) {
      projected_activations[
          (int(batch_index) * VERIFY_T + token_position_index) * N_SIZE +
          output_dimension_index] = T(projection_sum);
    }
  }
}
"#;

pub(super) fn target_verification_quantized_linear_kernel()
-> Result<MlxMetalKernel, MlxRuntimeError> {
    MlxMetalKernel::new_with_header(
        "astronomical_qwen3_5_target_verification_quantized_linear",
        &[
            "activations",
            "packed_weights",
            "quantization_scales",
            "quantization_biases",
        ],
        &["projected_activations"],
        TARGET_VERIFICATION_QUANTIZED_LINEAR_HEADER,
        TARGET_VERIFICATION_QUANTIZED_LINEAR_SOURCE,
    )
}

impl Qwen3_5Model {
    pub(crate) fn quantized_linear_for_paged_prefill_execution_mode(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5AffineWeights,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let activation_shape = activations.shape();
        if paged_prefill_execution_mode
            != Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
            || activation_shape.len() != 3
            || activation_shape[1] <= 1
        {
            return self.quantized_linear(activations, affine_weights);
        }
        if let Some(projected_activations) = self.try_target_verification_quantized_linear(
            activations,
            affine_weights,
            &activation_shape,
        )? {
            return Ok(projected_activations);
        }

        let mut token_projection_outputs = Vec::with_capacity(activation_shape[1] as usize);
        for token_position_index in 0..activation_shape[1] {
            let token_activations = self.runtime.slice(
                activations,
                &[0, token_position_index, 0],
                &[
                    activation_shape[0],
                    token_position_index + 1,
                    activation_shape[2],
                ],
                &[1, 1, 1],
            )?;
            token_projection_outputs
                .push(self.quantized_linear(&token_activations, affine_weights)?);
        }
        let token_projection_output_references =
            token_projection_outputs.iter().collect::<Vec<_>>();
        Ok(self
            .runtime
            .concatenate_axis(&token_projection_output_references, 1)?)
    }

    fn try_target_verification_quantized_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5AffineWeights,
        activation_shape: &[i32],
    ) -> Result<Option<MlxArray>, Qwen3_5ExecutionError> {
        let Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_group_size,
            quantization_bits,
        } = affine_weights
        else {
            return Ok(None);
        };
        let packed_weight_shape = packed_weight.shape();
        let output_dimension = packed_weight_shape[0];
        let input_dimension = activation_shape[2];
        if !matches!(*quantization_bits, 4 | 5)
            || input_dimension % 512 != 0
            || output_dimension % 8 != 0
            || activations.dtype() != quantization_scales.dtype()
            || quantization_biases.dtype() != quantization_scales.dtype()
            || !matches!(activations.dtype(), MlxDtype::BFloat16 | MlxDtype::Float16)
        {
            return Ok(None);
        }
        let mut kernel_outputs = self.runtime.apply_metal_kernel(
            &self.target_verification_quantized_linear_kernel,
            &[
                activations,
                packed_weight,
                quantization_scales,
                quantization_biases,
            ],
            &[MlxMetalKernelOutput::new(
                vec![activation_shape[0], activation_shape[1], output_dimension],
                activations.dtype(),
            )],
            [32, 2 * (output_dimension / 8), activation_shape[0]],
            [32, 2, 1],
            &[
                MlxMetalKernelTemplateArgument::Dtype {
                    name: "T",
                    dtype: activations.dtype(),
                },
                MlxMetalKernelTemplateArgument::Integer {
                    name: "BITS",
                    integer_template_argument: *quantization_bits,
                },
                MlxMetalKernelTemplateArgument::Integer {
                    name: "GS",
                    integer_template_argument: *quantization_group_size,
                },
                MlxMetalKernelTemplateArgument::Integer {
                    name: "VERIFY_T",
                    integer_template_argument: activation_shape[1],
                },
                MlxMetalKernelTemplateArgument::Integer {
                    name: "K_SIZE",
                    integer_template_argument: input_dimension,
                },
                MlxMetalKernelTemplateArgument::Integer {
                    name: "N_SIZE",
                    integer_template_argument: output_dimension,
                },
            ],
        )?;
        Ok(kernel_outputs.pop())
    }
}
