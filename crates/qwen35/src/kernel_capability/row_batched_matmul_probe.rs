//! Bounded capability probe for the register-ordered 4-bit matmul.
//!
//! The kernel needs the M5 GPU's matrix units (Metal Performance
//! Primitives); MLX compiles its source only when it first runs, so only an
//! evaluation proves a chip can use it. The probe converts one weight into
//! register order and runs one 6-row, 32-column,
//! 256-wide projection whose answer is exact: integer codes with scale 1 and
//! bias -8 give weights from -8 to 7, and each row holds three ones, so every
//! output is a small integer bf16 stores exactly. The expected values come from
//! the host, independent of MLX.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::{MlxArray, MlxDtype};

use super::{
    CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError, validate_probe_outputs,
};
use crate::row_batched_matmul::{
    RegisterOrderedWeight, RowBatchedMatmulKernels, register_ordered_matmul,
};

impl CustomMetalKernelFamily {
    /// Registry identifier of the row-batched 4-bit matmul.
    pub const ROW_BATCHED_MATMUL: Self = Self::new("row_batched_matmul");
}

pub fn capability_probes(runtime: &MlxRuntime) -> Vec<Box<dyn CustomMetalKernelProbe + '_>> {
    vec![Box::new(RowBatchedMatmulProbe { runtime })]
}

const PROBE_ROW_COUNT: usize = 6;
const PROBE_OUTPUT_WIDTH: usize = 32;
const PROBE_INPUT_WIDTH: usize = 256;
const PROBE_GROUP_COUNT: usize = PROBE_INPUT_WIDTH / 64;
/// Each probe row reads these input positions, one in each of three groups.
const PROBE_ROW_ONE_OFFSETS: [usize; 3] = [0, 64, 128];
const PROBE_BIAS: f32 = -8.0;

struct RowBatchedMatmulProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl CustomMetalKernelProbe for RowBatchedMatmulProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        CustomMetalKernelFamily::ROW_BATCHED_MATMUL
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernels = RowBatchedMatmulKernels::declare().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let probe_output = run_probe_projection(self.runtime, &kernels).map_err(|error| {
            KernelCapabilityError::Execution {
                description: error.to_string(),
            }
        })?;
        validate_probe_outputs(&probe_output, &expected_probe_outputs())
    }
}

/// The 4-bit code of output column `column` at input `input_index`.
fn probe_code(column: usize, input_index: usize) -> u32 {
    ((column * 7 + input_index * 3) % 16) as u32
}

/// Row `row` has a one at each `PROBE_ROW_ONE_OFFSETS` position shifted by `row`.
fn probe_row_value(row: usize, input_index: usize) -> f32 {
    if PROBE_ROW_ONE_OFFSETS
        .iter()
        .any(|offset| offset + row == input_index)
    {
        1.0
    } else {
        0.0
    }
}

fn expected_probe_outputs() -> Vec<f32> {
    let mut expected = Vec::with_capacity(PROBE_ROW_COUNT * PROBE_OUTPUT_WIDTH);
    for row in 0..PROBE_ROW_COUNT {
        for column in 0..PROBE_OUTPUT_WIDTH {
            let mut output = 0.0;
            for offset in PROBE_ROW_ONE_OFFSETS {
                output += probe_code(column, offset + row) as f32 + PROBE_BIAS;
            }
            expected.push(output);
        }
    }
    expected
}

fn run_probe_projection(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
) -> Result<Vec<f32>, MlxRuntimeError> {
    let mut packed_words = Vec::with_capacity(PROBE_OUTPUT_WIDTH * PROBE_INPUT_WIDTH / 8);
    for column in 0..PROBE_OUTPUT_WIDTH {
        for word_index in 0..PROBE_INPUT_WIDTH / 8 {
            let mut word = 0_u32;
            for nibble in 0..8 {
                word |= probe_code(column, word_index * 8 + nibble) << (4 * nibble);
            }
            packed_words.push(word);
        }
    }
    let packed_weight = runtime.array_from_u32(
        &packed_words,
        &[PROBE_OUTPUT_WIDTH as i32, (PROBE_INPUT_WIDTH / 8) as i32],
    )?;
    let scale_shape = [PROBE_OUTPUT_WIDTH as i32, PROBE_GROUP_COUNT as i32];
    let scales = bf16_filled(runtime, 1.0, &scale_shape)?;
    let biases = bf16_filled(runtime, PROBE_BIAS, &scale_shape)?;
    let mut row_values = Vec::with_capacity(PROBE_ROW_COUNT * PROBE_INPUT_WIDTH);
    for row in 0..PROBE_ROW_COUNT {
        for input_index in 0..PROBE_INPUT_WIDTH {
            row_values.push(probe_row_value(row, input_index));
        }
    }
    let rows = runtime.astype(
        &runtime.array_from_f32(
            &row_values,
            &[PROBE_ROW_COUNT as i32, PROBE_INPUT_WIDTH as i32],
        )?,
        MlxDtype::BFloat16,
    )?;
    let converted =
        RegisterOrderedWeight::convert(runtime, kernels, &packed_weight, &scales, &biases)
            .map_err(probe_failure)?;
    let projected =
        register_ordered_matmul(runtime, kernels, &rows, &converted).map_err(probe_failure)?;
    Ok(runtime.array_to_vec_f32(&runtime.astype(&projected, MlxDtype::Float32)?)?)
}

fn bf16_filled(
    runtime: &MlxRuntime,
    value: f32,
    shape: &[i32],
) -> Result<MlxArray, MlxRuntimeError> {
    let element_count: i32 = shape.iter().product();
    let values = vec![value; element_count as usize];
    Ok(runtime.astype(&runtime.array_from_f32(&values, shape)?, MlxDtype::BFloat16)?)
}

fn probe_failure(
    projection_error: crate::row_batched_matmul::RowBatchedProjectionError,
) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "run the register-ordered matmul probe",
        description: projection_error.to_string(),
    }
}
