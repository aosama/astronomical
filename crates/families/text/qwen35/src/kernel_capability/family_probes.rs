//! The Qwen 3.5 family's probe list for the worker's once-per-process
//! custom kernel verdicts. A new kernel family adds its probes here.

use runtime_integration::MlxRuntime;

use super::{
    CustomMetalKernelProbe, WorkerKernelCapabilities, fused_jangtq2_gathered_projection_probe,
    gated_delta_probes, prism_hadamard_probes, row_batched_matmul_probe,
    sorted_expert_shared_combination_probe, sorted_expert_weighted_sum_probe,
};
use crate::performance_attribution::PerformanceAttribution;

/// This worker process's verdicts for every custom kernel the family uses,
/// probed once and reused by every model swap and request.
pub fn worker_process_kernel_capabilities(
    runtime: &MlxRuntime,
    performance_attribution: &mut PerformanceAttribution,
) -> &'static WorkerKernelCapabilities {
    engine_core::kernel_capability_registry::worker_process_kernel_capabilities(
        runtime,
        qwen3_5_family_probes,
        performance_attribution,
    )
}

fn qwen3_5_family_probes(runtime: &MlxRuntime) -> Vec<Box<dyn CustomMetalKernelProbe + '_>> {
    let mut family_probes: Vec<Box<dyn CustomMetalKernelProbe + '_>> = Vec::new();
    family_probes.extend(sorted_expert_weighted_sum_probe::capability_probes(runtime));
    family_probes.extend(sorted_expert_shared_combination_probe::capability_probes(
        runtime,
    ));
    family_probes.extend(fused_jangtq2_gathered_projection_probe::capability_probes(
        runtime,
    ));
    family_probes.extend(gated_delta_probes::capability_probes(runtime));
    family_probes.extend(prism_hadamard_probes::capability_probes(runtime));
    family_probes.extend(row_batched_matmul_probe::capability_probes(runtime));
    family_probes
}
