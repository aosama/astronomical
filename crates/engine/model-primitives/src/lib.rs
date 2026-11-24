//! Model primitives shared by more than one model family (issue #2340): the
//! kernels, ops and layers the industry calls primitives.
//!
//! Beside the kernels and layers (row-batched matmul, sparse experts, token
//! sampling, attention math, quantized and add-on linear layers) it holds the
//! small vocabulary they and the engine record with: performance attribution,
//! the routed-expert values, the serving phase and the sampling seed; and the
//! memory report every forward-pass engine (decision, speech) publishes. Those
//! are not GPU code; they live here because the kernels and layers report through them,
//! and putting them higher up would make the kernels and layers depend on the engine.
//!
//! Every model family in `model-serving` builds on these pieces,
//! and none of them knows about a family, so the dependency runs one way: this
//! crate never depends on the serving crate, which `cargo xtask
//! check-architecture` enforces. A change here rebuilds and retests this crate
//! on its own; the families then pick it up.
//!
//! Unsafe code is denied workspace-wide. The one exception lives in
//! `performance_attribution::macos_process_io`, whose macOS accounting syscall
//! is inseparable from the evidence types it produces (rationale lives there).

#[cfg(feature = "direct-mlx")]
pub mod affine_quantized_linear;

pub mod affine_quantization_scheme;

#[cfg(feature = "direct-mlx")]
pub mod adapter_target;

pub mod attention;

pub mod family_engine_settings;

#[cfg(feature = "direct-mlx")]
pub mod forward_pass_memory;

#[cfg(feature = "direct-mlx")]
pub mod fp32_layers;

#[cfg(feature = "direct-mlx")]
pub mod gpu_token_sampling;

#[cfg(feature = "direct-mlx")]
pub mod linear_adapters;

// Small synthetic weight files for GPU tests in this crate and the family
// crates above it; compiled only for this crate's tests or when a dependent's
// dev build turns on `test-support`.
#[cfg(all(feature = "direct-mlx", any(test, feature = "test-support")))]
pub mod synthetic_weights_test_support;

pub mod performance_attribution;

pub mod route_observation;

pub mod row_batched_matmul;

pub mod sampling_seed;

pub mod serving_phase;

pub mod sparse_experts;

// The public surface the serving crate re-exported from these modules before
// they moved here; the serving crate and the families re-export or import it.
#[cfg(feature = "direct-mlx")]
pub use attention::build_causal_sliding_window_mask;

pub use family_engine_settings::FamilyEngineSettings;

pub use attention::{
    RopeFrequencyError, SlidingWindowVisibilityError, YarnRopeFrequencyDenominators,
    compute_default_rope_frequency_denominators, compute_yarn_rope_frequency_denominators,
    sliding_window_position_is_visible, sliding_window_visibility_table,
};

#[doc(hidden)]
#[cfg(feature = "direct-mlx")]
pub use gpu_token_sampling::{
    build_sampled_token_rows_for_tests, sample_acceptance_coins_for_tests,
    sample_from_relative_probabilities_for_tests,
};

pub use performance_attribution::macos_process_io::{
    MacosProcessIoDelta, sample_current_process_io, sample_process_io_for_process_id,
};

pub use performance_attribution::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceAttributionReport, PerformanceCounter, PerformanceOperation,
    PerformanceOperationMeasurement, process_io_delta_between_samples,
};

pub use performance_attribution::macos_process_io::MacosProcessIoError;

pub use route_observation::{LayerRoutedExpertIds, ObservedExpertRoute};

#[cfg(feature = "direct-mlx")]
pub use sparse_experts::{
    ExpertAssignmentOrder, SORTED_EXPERT_REDUCTION_THREADS_PER_THREADGROUP,
    SORTED_EXPERT_WEIGHTED_REDUCTION_METAL_STATEMENTS, SortedExpertAssignments,
    StackedExpertProjection, expert_assignment_order, gather_expert_projection,
    restore_expert_assignment_order, router_weighted_expert_inputs, sort_expert_assignments,
    sorted_expert_weighted_sum, sorted_expert_weighted_sum_kernel, unsorted_expert_weighted_sum,
};

pub use sparse_experts::{SparseExpertError, invert_assignment_order};
