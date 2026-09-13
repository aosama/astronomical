//! Oracle references for the `qwen4_exp` subsystems production cannot yet
//! certify itself.
//!
//! Primitive attention, rotary, gated-delta recurrence, gathered quantized
//! projections, and sparse-expert reduction already have independent
//! references in this boundary, and the family reuses them. Three pieces of
//! the architecture have no reference anywhere: the sparse-attention
//! indexer's top-k selection, the lookup-table row decode, and the
//! hyper-connection stream mixing on the GPU. This module composes those
//! three from raw MLX operations and compares each against host-side `f64`
//! math, so a production route that drifts fails here rather than producing
//! fluent wrong text.
//!
//! The oracle is intentionally unoptimized: dense materialized masks, whole
//! scoring, dequantized rows. Production must never depend on it.

use astronomical_runtime_integration::{MlxArray, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

pub(crate) mod hyper_connection_executor;
pub(crate) mod hyper_connection_reference;
pub(crate) mod indexer_reference;
pub(crate) mod ple_row_reference;
pub(crate) mod qsa_parity;

pub(crate) fn oracle_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("oracle test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Asserts two evaluated vectors agree within a dtype-appropriate tolerance.
pub(crate) fn assert_f32_close(actual: &[f32], expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: element count must match"
    );
    for (index, (value, expected_value)) in actual.iter().zip(expected).enumerate() {
        let difference = (*value as f64 - expected_value).abs();
        // The tolerance is relative to the expected magnitude, floored at an
        // absolute bound for near-zero expectations: Metal matrix kernels may
        // trade accumulation precision for throughput, and the oracle must
        // absorb that hardware behavior without hiding a real defect.
        let bound = tolerance * expected_value.abs().max(1.0);
        assert!(
            difference <= bound,
            "{context}: element {index} differs by {difference} (bound {bound}): {value} vs {expected_value}"
        );
    }
}

/// Deterministic small-value generator so every test input is reproducible
/// without a random dependency: a splitmix-style sequence scaled into a
/// bounded range.
pub(crate) struct DeterministicValues {
    state: u64,
}

impl DeterministicValues {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    pub(crate) fn next_f32(&mut self, magnitude: f32) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.state >> 40) as f32 / (1 << 24) as f32;
        (unit * 2.0 - 1.0) * magnitude
    }

    pub(crate) fn vec(&mut self, count: usize, magnitude: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_f32(magnitude)).collect()
    }
}

/// Builds one `f32` array from host values.
pub(crate) fn f32_array(
    runtime: &MlxRuntime,
    values: &[f32],
    shape: &[i32],
) -> Result<MlxArray, astronomical_runtime_integration::MlxRuntimeError> {
    runtime.array_from_f32(values, shape)
}
