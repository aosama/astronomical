//! Bounded capability probe for the tiled sorted-expert kernels: gate, up
//! and their SwiGLU in one kernel from the token rows, and the tiled down
//! projection.
//!
//! Like the sorted-expert matmul, they need the M5 GPU's matrix units and
//! MLX compiles them only when they first run, so only an evaluation proves
//! a chip can use them. The probe runs a small routed layer both ways, the
//! tiled kernels and MLX's own route (its sorted `gather_qmm` for gate and
//! up on the sorted copy of the rows, its compiled SwiGLU, its sorted
//! `gather_qmm` for down), and asks for the same values to the bit at both
//! the SwiGLU and the down output.

use mlx_bridge::{MlxArray, MlxCompiledSwiGlu, MlxDtype};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::sorted_expert_matmul_probe::{
    exact_bits_verdict, float_values, probe_routing, probe_value,
};
use super::{CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError};
use crate::sorted_expert_matmul::{
    ExpertStack, TiledExpertKernels, sorted_expert_swiglu_from_token_rows,
    tiled_sorted_expert_matmul,
};

/// Registry identifier of the tiled sorted-expert kernels.
pub const SORTED_EXPERT_TILED_MATMUL: CustomMetalKernelFamily =
    CustomMetalKernelFamily::new("sorted_expert_tiled_matmul");

pub fn capability_probes(runtime: &MlxRuntime) -> Vec<Box<dyn CustomMetalKernelProbe + '_>> {
    vec![Box::new(SortedExpertTiledMatmulProbe { runtime })]
}

const PROBE_EXPERT_COUNT: i32 = 4;
const PROBE_HIDDEN_WIDTH: i32 = 128;
const PROBE_EXPERT_WIDTH: i32 = 64;
const PROBE_TOKEN_COUNT: i32 = 24;
const PROBE_EXPERTS_A_TOKEN: i32 = 2;
const PROBE_GROUP_SIZE: i32 = 64;

struct SortedExpertTiledMatmulProbe<'runtime> {
    runtime: &'runtime MlxRuntime,
}

impl CustomMetalKernelProbe for SortedExpertTiledMatmulProbe<'_> {
    fn family(&self) -> CustomMetalKernelFamily {
        SORTED_EXPERT_TILED_MATMUL
    }

    fn probe(&self) -> Result<(), KernelCapabilityError> {
        let kernels =
            TiledExpertKernels::declare().map_err(|error| KernelCapabilityError::Compilation {
                description: error.to_string(),
            })?;
        let (kernel_values, mlx_values) =
            run_probe_layer(self.runtime, &kernels).map_err(|error| {
                KernelCapabilityError::Execution {
                    description: error.to_string(),
                }
            })?;
        exact_bits_verdict(&kernel_values, &mlx_values)
    }
}

/// A probe expert stack of fixed values, quantized by MLX to 4 bits.
fn probe_expert_stack(
    runtime: &MlxRuntime,
    (output_width, input_width): (i32, i32),
    value_stride: usize,
) -> Result<(MlxArray, MlxArray, MlxArray), MlxRuntimeError> {
    let shape = [PROBE_EXPERT_COUNT, output_width, input_width];
    let value_count = shape.iter().product::<i32>() as usize;
    let weight_values: Vec<f32> = (0..value_count)
        .map(|index| probe_value(index, value_stride))
        .collect();
    let weights = runtime.astype(
        &runtime.array_from_f32(&weight_values, &shape)?,
        MlxDtype::BFloat16,
    )?;
    Ok(runtime.quantize_affine(&weights, PROBE_GROUP_SIZE, 4)?)
}

fn stack_of(experts: &(MlxArray, MlxArray, MlxArray)) -> ExpertStack<'_> {
    ExpertStack {
        packed_weight: &experts.0,
        scales: &experts.1,
        biases: &experts.2,
    }
}

fn mlx_sorted_projection(
    runtime: &MlxRuntime,
    rows: &MlxArray,
    experts: &(MlxArray, MlxArray, MlxArray),
    sorted_experts: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    Ok(runtime.gather_quantized_matmul_affine(
        rows,
        &experts.0,
        &experts.1,
        &experts.2,
        None,
        Some(sorted_experts),
        true,
        PROBE_GROUP_SIZE,
        4,
        true,
    )?)
}

fn kernel_failure(projection_error: impl std::fmt::Display) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "run the tiled sorted-expert probe",
        description: projection_error.to_string(),
    }
}

/// The probe layer's inputs: gate, up and down stacks, token rows and the
/// sorted routing (order, tokens, experts).
struct ProbeLayer {
    gate: (MlxArray, MlxArray, MlxArray),
    up: (MlxArray, MlxArray, MlxArray),
    down: (MlxArray, MlxArray, MlxArray),
    token_rows: MlxArray,
    routing: (MlxArray, MlxArray, MlxArray),
}

fn probe_layer(runtime: &MlxRuntime) -> Result<ProbeLayer, MlxRuntimeError> {
    let token_values: Vec<f32> = (0..(PROBE_TOKEN_COUNT * PROBE_HIDDEN_WIDTH) as usize)
        .map(|index| probe_value(index, 11))
        .collect();
    let token_rows = runtime.astype(
        &runtime.array_from_f32(&token_values, &[PROBE_TOKEN_COUNT, 1, PROBE_HIDDEN_WIDTH])?,
        MlxDtype::BFloat16,
    )?;
    Ok(ProbeLayer {
        gate: probe_expert_stack(runtime, (PROBE_EXPERT_WIDTH, PROBE_HIDDEN_WIDTH), 7)?,
        up: probe_expert_stack(runtime, (PROBE_EXPERT_WIDTH, PROBE_HIDDEN_WIDTH), 5)?,
        down: probe_expert_stack(runtime, (PROBE_HIDDEN_WIDTH, PROBE_EXPERT_WIDTH), 3)?,
        token_rows,
        routing: probe_routing(runtime)?,
    })
}

/// The SwiGLU outputs followed by the down outputs, as float32.
fn layer_values(
    runtime: &MlxRuntime,
    activated: &MlxArray,
    down_outputs: &MlxArray,
) -> Result<Vec<f32>, MlxRuntimeError> {
    let mut values = float_values(runtime, activated)?;
    values.extend(float_values(runtime, down_outputs)?);
    Ok(values)
}

fn kernel_layer_values(
    runtime: &MlxRuntime,
    kernels: &TiledExpertKernels,
    layer: &ProbeLayer,
) -> Result<Vec<f32>, MlxRuntimeError> {
    let (sorted_order, _, sorted_experts) = &layer.routing;
    let activated = sorted_expert_swiglu_from_token_rows(
        runtime,
        kernels,
        (&layer.token_rows, sorted_order, sorted_experts),
        PROBE_EXPERTS_A_TOKEN,
        (&stack_of(&layer.gate), &stack_of(&layer.up)),
    )
    .map_err(kernel_failure)?;
    let down_outputs = tiled_sorted_expert_matmul(
        runtime,
        kernels,
        &activated,
        &stack_of(&layer.down),
        sorted_experts,
    )
    .map_err(kernel_failure)?;
    layer_values(runtime, &activated, &down_outputs)
}

fn mlx_layer_values(runtime: &MlxRuntime, layer: &ProbeLayer) -> Result<Vec<f32>, MlxRuntimeError> {
    let (_, sorted_tokens, sorted_experts) = &layer.routing;
    let sorted_rows = runtime.take_axis(&layer.token_rows, sorted_tokens, 0)?;
    let compiled_swiglu = MlxCompiledSwiGlu::new()?;
    let activated = runtime.apply_compiled_swiglu(
        &compiled_swiglu,
        &mlx_sorted_projection(runtime, &sorted_rows, &layer.gate, sorted_experts)?,
        &mlx_sorted_projection(runtime, &sorted_rows, &layer.up, sorted_experts)?,
    )?;
    let down_outputs = mlx_sorted_projection(runtime, &activated, &layer.down, sorted_experts)?;
    layer_values(runtime, &activated, &down_outputs)
}

/// (kernel values, MLX values) for the probe layer.
fn run_probe_layer(
    runtime: &MlxRuntime,
    kernels: &TiledExpertKernels,
) -> Result<(Vec<f32>, Vec<f32>), MlxRuntimeError> {
    let layer = probe_layer(runtime)?;
    Ok((
        kernel_layer_values(runtime, kernels, &layer)?,
        mlx_layer_values(runtime, &layer)?,
    ))
}

#[cfg(test)]
#[path = "sorted_expert_tiled_matmul_probe_tests.rs"]
mod sorted_expert_tiled_matmul_probe_tests;
