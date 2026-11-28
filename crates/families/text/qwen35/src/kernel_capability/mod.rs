//! The Qwen 3.5 family's custom Metal kernel probes: one module per kernel
//! family, each defining its registry identifier and its probes. The
//! family-neutral verdict registry lives in
//! `engine_core::kernel_capability_registry`; this family's probe list is in
//! `family_probes`.

#[cfg(feature = "direct-mlx")]
mod family_probes;
#[cfg(feature = "direct-mlx")]
pub mod fused_jangtq2_gathered_projection_probe;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta_probes;
#[cfg(feature = "direct-mlx")]
pub mod prism_hadamard_probes;
#[cfg(all(test, feature = "direct-mlx"))]
mod probe_test_runtime;
#[cfg(feature = "direct-mlx")]
pub mod row_batched_matmul_probe;
#[cfg(feature = "direct-mlx")]
pub mod sorted_expert_matmul_probe;
#[cfg(feature = "direct-mlx")]
pub mod sorted_expert_shared_combination_probe;
#[cfg(feature = "direct-mlx")]
pub mod sorted_expert_tiled_matmul_probe;
#[cfg(feature = "direct-mlx")]
pub mod sorted_expert_weighted_sum_probe;

#[cfg(feature = "direct-mlx")]
pub use engine_core::kernel_capability_registry::install_forced_worker_verdicts_for_tests;
pub use engine_core::kernel_capability_registry::{
    CustomKernelVerdict, CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError,
    KernelUnsupportedReason, WorkerKernelCapabilities, validate_probe_outputs,
};
#[cfg(feature = "direct-mlx")]
pub use family_probes::worker_process_kernel_capabilities;
#[cfg(feature = "direct-mlx")]
pub use fused_jangtq2_gathered_projection_probe::FUSED_JANGTQ2_GATHERED_PROJECTION;
#[cfg(feature = "direct-mlx")]
pub use gated_delta_probes::{
    GATED_DELTA_BOUNDARY_CHECKPOINT, GATED_DELTA_SEQUENCE, GDN_DECODE_PREWORK,
};
#[cfg(feature = "direct-mlx")]
pub use prism_hadamard_probes::PRISM_HADAMARD_SIGNED_ROTATION;
#[cfg(feature = "direct-mlx")]
pub use row_batched_matmul_probe::ROW_BATCHED_MATMUL;
#[cfg(feature = "direct-mlx")]
pub use sorted_expert_matmul_probe::SORTED_EXPERT_MATMUL;
#[cfg(feature = "direct-mlx")]
pub use sorted_expert_shared_combination_probe::SORTED_EXPERT_SHARED_COMBINATION;
#[cfg(feature = "direct-mlx")]
pub use sorted_expert_tiled_matmul_probe::{
    SORTED_EXPERT_TILED_MATMUL, SORTED_EXPERT_TILED_MATMUL_6BIT,
    SORTED_EXPERT_TILED_MATMUL_FAMILIES,
};
#[cfg(feature = "direct-mlx")]
pub use sorted_expert_weighted_sum_probe::SORTED_EXPERT_WEIGHTED_SUM;
#[cfg(feature = "direct-mlx")]
pub use sorted_expert_weighted_sum_probe::SortedExpertWeightedSumProbe;
