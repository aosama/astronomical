use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxRuntime, MlxRuntimeError,
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

/// Identifies whether target verification used the specialized Metal kernel or
/// the token-local MLX reference path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3_5TargetVerificationProjectionDispatch {
    OptimizedMetal,
    TokenLocalMlxFallback,
}

/// One target-verification projection and the geometry-derived dispatch used
/// to produce it.
#[derive(Debug)]
pub struct Qwen3_5TargetVerificationProjection {
    projected_activations: MlxArray,
    dispatch: Qwen3_5TargetVerificationProjectionDispatch,
}

impl Qwen3_5TargetVerificationProjection {
    /// Returns the dispatch selected from validated tensor geometry.
    #[must_use]
    pub const fn dispatch(&self) -> Qwen3_5TargetVerificationProjectionDispatch {
        self.dispatch
    }

    /// Consumes the diagnostic wrapper and returns the projected activations.
    #[must_use]
    pub fn into_projected_activations(self) -> MlxArray {
        self.projected_activations
    }
}

/// Builds the retained custom Metal kernel used by eligible target-verification
/// projections.
pub fn target_verification_quantized_linear_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
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

/// Projects one multi-token target-verification window through the specialized
/// Metal route when its geometry is supported. Every other valid affine shape
/// uses repeated one-token MLX quantized matrix multiplication so fallback
/// arithmetic remains identical to ordinary decode.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_target_verification_quantized_linear(
    runtime: &MlxRuntime,
    _target_verification_kernel: &MlxMetalKernel,
    activations: &MlxArray,
    packed_weight: &MlxArray,
    quantization_scales: &MlxArray,
    quantization_biases: &MlxArray,
    quantization_group_size: i32,
    quantization_bits: i32,
) -> Result<Qwen3_5TargetVerificationProjection, MlxRuntimeError> {
    let activation_shape = activations.shape();
    let packed_weight_shape = packed_weight.shape();
    validate_target_verification_projection(
        packed_weight,
        quantization_scales,
        quantization_biases,
        quantization_group_size,
        quantization_bits,
        &activation_shape,
        &packed_weight_shape,
    )?;
    let output_dimension = packed_weight_shape[0];
    let input_dimension = activation_shape[2];
    if target_verification_uses_optimized_dispatch(
        activations,
        quantization_scales,
        quantization_biases,
        quantization_group_size,
        quantization_bits,
        input_dimension,
        output_dimension,
    ) {
        // MLX 0.32 routes two through twelve rows through qmv_wide on capable
        // Apple GPUs, reusing each quantized weight group across verification
        // positions. Direct-MLX qualification proves this route remains bit
        // exact to repeated one-token target decode for retained 4- and 5-bit
        // geometries, so the older custom QMV no longer justifies bypassing it.
        let projected_activations = runtime.quantized_matmul_affine(
            activations,
            packed_weight,
            quantization_scales,
            quantization_biases,
            true,
            quantization_group_size,
            quantization_bits,
        )?;
        return Ok(Qwen3_5TargetVerificationProjection {
            projected_activations,
            dispatch: Qwen3_5TargetVerificationProjectionDispatch::OptimizedMetal,
        });
    }

    let projected_activations = token_local_quantized_linear(
        runtime,
        activations,
        packed_weight,
        quantization_scales,
        quantization_biases,
        quantization_group_size,
        quantization_bits,
        &activation_shape,
    )?;
    Ok(Qwen3_5TargetVerificationProjection {
        projected_activations,
        dispatch: Qwen3_5TargetVerificationProjectionDispatch::TokenLocalMlxFallback,
    })
}

#[allow(clippy::too_many_arguments)]
fn target_verification_uses_optimized_dispatch(
    activations: &MlxArray,
    quantization_scales: &MlxArray,
    quantization_biases: &MlxArray,
    quantization_group_size: i32,
    quantization_bits: i32,
    input_dimension: i32,
    output_dimension: i32,
) -> bool {
    matches!(quantization_bits, 4 | 5)
        && matches!(quantization_group_size, 32 | 64 | 128)
        && input_dimension > 0
        && input_dimension % 512 == 0
        && output_dimension > 0
        && output_dimension % 8 == 0
        && output_dimension
            .checked_div(8)
            .and_then(|tile_count| tile_count.checked_mul(2))
            .is_some()
        && activations.dtype() == quantization_scales.dtype()
        && quantization_biases.dtype() == quantization_scales.dtype()
        && matches!(activations.dtype(), MlxDtype::BFloat16 | MlxDtype::Float16)
}

#[allow(clippy::too_many_arguments)]
fn token_local_quantized_linear(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    packed_weight: &MlxArray,
    quantization_scales: &MlxArray,
    quantization_biases: &MlxArray,
    quantization_group_size: i32,
    quantization_bits: i32,
    activation_shape: &[i32],
) -> Result<MlxArray, MlxRuntimeError> {
    let mut token_projection_outputs = Vec::with_capacity(activation_shape[1] as usize);
    for token_position_index in 0..activation_shape[1] {
        let token_activations = runtime.slice(
            activations,
            &[0, token_position_index, 0],
            &[
                activation_shape[0],
                token_position_index + 1,
                activation_shape[2],
            ],
            &[1, 1, 1],
        )?;
        token_projection_outputs.push(runtime.quantized_matmul_affine(
            &token_activations,
            packed_weight,
            quantization_scales,
            quantization_biases,
            true,
            quantization_group_size,
            quantization_bits,
        )?);
    }
    let token_projection_output_references = token_projection_outputs.iter().collect::<Vec<_>>();
    runtime.concatenate_axis(&token_projection_output_references, 1)
}

#[allow(clippy::too_many_arguments)]
fn validate_target_verification_projection(
    packed_weight: &MlxArray,
    quantization_scales: &MlxArray,
    quantization_biases: &MlxArray,
    quantization_group_size: i32,
    quantization_bits: i32,
    activation_shape: &[i32],
    packed_weight_shape: &[i32],
) -> Result<(), MlxRuntimeError> {
    let has_positive_activation_geometry =
        activation_shape.len() == 3 && activation_shape.iter().all(|dimension| *dimension > 0);
    let has_positive_weight_geometry = packed_weight_shape.len() == 2
        && packed_weight_shape.iter().all(|dimension| *dimension > 0);
    if !has_positive_activation_geometry || !has_positive_weight_geometry {
        return Err(target_verification_projection_error(
            "target-verification arrays must have positive rank-three activations and rank-two weights",
        ));
    }
    if packed_weight.dtype() != MlxDtype::UInt32
        || !matches!(quantization_group_size, 32 | 64 | 128)
        || !matches!(quantization_bits, 2 | 3 | 4 | 5 | 6 | 8)
    {
        return Err(target_verification_projection_error(
            "target-verification affine metadata is not supported by MLX",
        ));
    }
    let input_dimension = activation_shape[2];
    let expanded_weight_input =
        packed_weight_shape[1]
            .checked_mul(32)
            .and_then(|packed_bit_count| {
                (packed_bit_count % quantization_bits == 0)
                    .then_some(packed_bit_count / quantization_bits)
            });
    let expected_affine_shape = vec![
        packed_weight_shape[0],
        input_dimension / quantization_group_size,
    ];
    if expanded_weight_input != Some(input_dimension)
        || input_dimension % quantization_group_size != 0
        || quantization_scales.shape() != expected_affine_shape
        || quantization_biases.shape() != expected_affine_shape
    {
        return Err(target_verification_projection_error(
            "target-verification affine arrays do not match activation and weight geometry",
        ));
    }
    Ok(())
}

fn target_verification_projection_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "project Qwen3.5 target-verification tokens",
        description: description.to_owned(),
    }
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
        let Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_group_size,
            quantization_bits,
        } = affine_weights
        else {
            return self.quantized_linear(activations, affine_weights);
        };
        Ok(qwen3_5_target_verification_quantized_linear(
            &self.runtime,
            &self.target_verification_quantized_linear_kernel,
            activations,
            packed_weight,
            quantization_scales,
            quantization_biases,
            *quantization_group_size,
            *quantization_bits,
        )?
        .into_projected_activations())
    }
}
