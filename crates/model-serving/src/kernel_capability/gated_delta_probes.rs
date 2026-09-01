//! Bounded capability probes for the fused gated-delta Metal kernels.
//!
//! Each probe compiles its kernel, executes a minimal valid sequence, and
//! validates the output against the ops-based one-token reference — the same
//! public MLX composition that serves as the production fallback, with the
//! same 1e-3 absolute tolerance the retained direct-MLX contracts use. Probe
//! geometry is bounded by the sequence contract: a 128 key dimension, a
//! 32-divisible value dimension, one key head, and one value head.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use super::{CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError};
use crate::qwen3_5::{
    qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_sequence_ops_fallback,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints_ops_fallback,
};

const PROBE_TOKEN_COUNT: i32 = 3;
const PROBE_KEY_HEAD_COUNT: i32 = 1;
const PROBE_VALUE_HEAD_COUNT: i32 = 1;
const PROBE_KEY_HEAD_DIMENSION: i32 = 128;
const PROBE_VALUE_HEAD_DIMENSION: i32 = 32;
/// The retained fused-versus-ops parity tolerance from the gated-delta
/// direct-MLX contracts.
const PROBE_PARITY_TOLERANCE: f32 = 1e-3;

struct ProbeSequenceArrays {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
    decays: MlxArray,
    update_rates: MlxArray,
    recurrent_state: MlxArray,
}

fn probe_sequence_arrays(runtime: &MlxRuntime) -> Result<ProbeSequenceArrays, MlxRuntimeError> {
    let activation_values =
        (0..(PROBE_TOKEN_COUNT * PROBE_KEY_HEAD_COUNT * PROBE_KEY_HEAD_DIMENSION) as usize)
            .map(|value_index| ((value_index % 17) as f32 - 8.0) / 64.0)
            .collect::<Vec<_>>();
    let queries = patterned_rank_four(runtime, &activation_values)?;
    let keys = patterned_rank_four(runtime, &activation_values)?;
    let value_values = (0
        ..(PROBE_TOKEN_COUNT * PROBE_VALUE_HEAD_COUNT * PROBE_VALUE_HEAD_DIMENSION) as usize)
        .map(|value_index| ((value_index % 13) as f32 - 6.0) / 32.0)
        .collect::<Vec<_>>();
    let values = runtime
        .array_from_f32(
            &value_values,
            &[
                1,
                PROBE_TOKEN_COUNT,
                PROBE_VALUE_HEAD_COUNT,
                PROBE_VALUE_HEAD_DIMENSION,
            ],
        )
        .and_then(|array| runtime.astype(&array, MlxDtype::BFloat16))?;
    let rate_values = (0..(PROBE_TOKEN_COUNT * PROBE_VALUE_HEAD_COUNT) as usize)
        .map(|rate_index| 0.5 + (rate_index % 5) as f32 / 16.0)
        .collect::<Vec<_>>();
    let decays = runtime.array_from_f32(
        &rate_values,
        &[1, PROBE_TOKEN_COUNT, PROBE_VALUE_HEAD_COUNT],
    )?;
    let update_rates = runtime.array_from_f32(
        &rate_values,
        &[1, PROBE_TOKEN_COUNT, PROBE_VALUE_HEAD_COUNT],
    )?;
    let state_values = (0..(PROBE_VALUE_HEAD_COUNT
        * PROBE_VALUE_HEAD_DIMENSION
        * PROBE_KEY_HEAD_DIMENSION) as usize)
        .map(|state_index| ((state_index % 11) as f32 - 5.0) / 32.0)
        .collect::<Vec<_>>();
    let recurrent_state = runtime.array_from_f32(
        &state_values,
        &[
            1,
            PROBE_VALUE_HEAD_COUNT,
            PROBE_VALUE_HEAD_DIMENSION,
            PROBE_KEY_HEAD_DIMENSION,
        ],
    )?;
    Ok(ProbeSequenceArrays {
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    })
}

fn patterned_rank_four(
    runtime: &MlxRuntime,
    activation_values: &[f32],
) -> Result<MlxArray, MlxRuntimeError> {
    runtime
        .array_from_f32(
            activation_values,
            &[
                1,
                PROBE_TOKEN_COUNT,
                PROBE_KEY_HEAD_COUNT,
                PROBE_KEY_HEAD_DIMENSION,
            ],
        )
        .and_then(|array| runtime.astype(&array, MlxDtype::BFloat16))
}

fn float32_values(runtime: &MlxRuntime, array: &MlxArray) -> Result<Vec<f32>, MlxRuntimeError> {
    runtime
        .astype(array, MlxDtype::Float32)
        .and_then(|float32_array| float32_array.to_vec_f32())
}

fn execution_error(error: MlxRuntimeError) -> KernelCapabilityError {
    KernelCapabilityError::Execution {
        description: error.to_string(),
    }
}

fn assert_probe_close(
    probe_values: &[f32],
    reference_values: &[f32],
    description: &str,
) -> Result<(), KernelCapabilityError> {
    if probe_values.len() != reference_values.len() {
        return Err(KernelCapabilityError::OutputMismatch {
            description: format!(
                "{description} produced {} values but the reference produced {}",
                probe_values.len(),
                reference_values.len()
            ),
        });
    }
    for (value_index, (probe_value, reference_value)) in
        probe_values.iter().zip(reference_values.iter()).enumerate()
    {
        if (probe_value - reference_value).abs() > PROBE_PARITY_TOLERANCE {
            return Err(KernelCapabilityError::OutputMismatch {
                description: format!(
                    "{description} value {value_index} read {probe_value:.6} but the reference read {reference_value:.6}"
                ),
            });
        }
    }
    Ok(())
}

pub struct GatedDeltaSequenceProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> GatedDeltaSequenceProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for GatedDeltaSequenceProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        CustomMetalKernelFamily::GatedDeltaSequence
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernel = crate::qwen3_5::qwen3_5_gated_delta_kernel().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let probe_inputs = probe_sequence_arrays(self.runtime).map_err(execution_error)?;
        let (probe_outputs, probe_next_state) = qwen3_5_gated_delta_sequence(
            self.runtime,
            Some(&kernel),
            &probe_inputs.queries,
            &probe_inputs.keys,
            &probe_inputs.values,
            &probe_inputs.decays,
            &probe_inputs.update_rates,
            &probe_inputs.recurrent_state,
        )
        .map_err(execution_error)?;
        let (reference_outputs, reference_next_state) = qwen3_5_gated_delta_sequence_ops_fallback(
            self.runtime,
            &probe_inputs.queries,
            &probe_inputs.keys,
            &probe_inputs.values,
            &probe_inputs.decays,
            &probe_inputs.update_rates,
            &probe_inputs.recurrent_state,
        )
        .map_err(execution_error)?;
        for (probe_array, reference_array, description) in [
            (&probe_outputs, &reference_outputs, "probe sequence outputs"),
            (
                &probe_next_state,
                &reference_next_state,
                "probe next recurrent state",
            ),
        ] {
            let probe_values =
                float32_values(self.runtime, probe_array).map_err(execution_error)?;
            let reference_values =
                float32_values(self.runtime, reference_array).map_err(execution_error)?;
            assert_probe_close(&probe_values, &reference_values, description)?;
        }
        Ok(())
    }
}

pub struct GatedDeltaBoundaryCheckpointProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl<'runtime> GatedDeltaBoundaryCheckpointProbe<'runtime> {
    #[must_use]
    pub const fn new(runtime: &'runtime MlxRuntime) -> Self {
        Self { runtime }
    }
}

impl CustomMetalKernelProbe for GatedDeltaBoundaryCheckpointProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        CustomMetalKernelFamily::GatedDeltaBoundaryCheckpoint
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernel = crate::qwen3_5::qwen3_5_gated_delta_checkpoint_kernel().map_err(|error| {
            KernelCapabilityError::Compilation {
                description: error.to_string(),
            }
        })?;
        let probe_inputs = probe_sequence_arrays(self.runtime).map_err(execution_error)?;
        let probe_result = qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
            self.runtime,
            Some(&kernel),
            &probe_inputs.queries,
            &probe_inputs.keys,
            &probe_inputs.values,
            &probe_inputs.decays,
            &probe_inputs.update_rates,
            &probe_inputs.recurrent_state,
            &[2],
            2,
        )
        .map_err(execution_error)?;
        let reference_result = qwen3_5_gated_delta_sequence_with_boundary_checkpoints_ops_fallback(
            self.runtime,
            &probe_inputs.queries,
            &probe_inputs.keys,
            &probe_inputs.values,
            &probe_inputs.decays,
            &probe_inputs.update_rates,
            &probe_inputs.recurrent_state,
            &[2],
            2,
        )
        .map_err(execution_error)?;
        for (probe_array, reference_array, description) in [
            (
                &probe_result.sequence_outputs,
                &reference_result.sequence_outputs,
                "probe checkpoint sequence outputs",
            ),
            (
                &probe_result.next_recurrent_state,
                &reference_result.next_recurrent_state,
                "probe checkpoint next recurrent state",
            ),
        ] {
            let probe_values =
                float32_values(self.runtime, probe_array).map_err(execution_error)?;
            let reference_values =
                float32_values(self.runtime, reference_array).map_err(execution_error)?;
            assert_probe_close(&probe_values, &reference_values, description)?;
        }
        if probe_result.recurrent_boundary_states.len()
            != reference_result.recurrent_boundary_states.len()
        {
            return Err(KernelCapabilityError::OutputMismatch {
                description: format!(
                    "probe produced {} boundary states but the reference produced {}",
                    probe_result.recurrent_boundary_states.len(),
                    reference_result.recurrent_boundary_states.len()
                ),
            });
        }
        for (boundary_index, (probe_state, reference_state)) in probe_result
            .recurrent_boundary_states
            .iter()
            .zip(reference_result.recurrent_boundary_states.iter())
            .enumerate()
        {
            let probe_values =
                float32_values(self.runtime, probe_state).map_err(execution_error)?;
            let reference_values =
                float32_values(self.runtime, reference_state).map_err(execution_error)?;
            assert_probe_close(
                &probe_values,
                &reference_values,
                &format!("probe boundary state {boundary_index}"),
            )?;
        }
        Ok(())
    }
}
