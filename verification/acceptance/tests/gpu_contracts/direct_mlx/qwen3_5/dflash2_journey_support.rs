//! Small helpers for the DFlash2 journeys on a real target and drafter pair.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;

/// Every value of `array`, read to the host as float32.
pub(crate) fn host_values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    let single_precision = runtime
        .astype(array, MlxDtype::Float32)
        .expect("values convert");
    runtime
        .array_to_vec_f32(&single_precision)
        .expect("values read back")
}

/// The first `row_count` rows of each tapped layer output [1, rows, H],
/// joined in tap order into the drafter's fusion input [1, row_count, taps x H].
pub(crate) fn tap_rows(runtime: &MlxRuntime, layer_taps: &[MlxArray], row_count: i32) -> MlxArray {
    let mut kept_rows = Vec::with_capacity(layer_taps.len());
    for layer_tap in layer_taps {
        let hidden_size = layer_tap.shape()[2];
        kept_rows.push(
            runtime
                .slice(
                    layer_tap,
                    &[0, 0, 0],
                    &[1, row_count, hidden_size],
                    &[1, 1, 1],
                )
                .expect("the kept tap rows slice"),
        );
    }
    let references: Vec<&MlxArray> = kept_rows.iter().collect();
    runtime
        .concatenate_axis(&references, -1)
        .expect("the taps join")
}

/// A seeded uniform stream for acceptance tests and leftover sampling
/// (splitmix64), so a run's verdicts are reproducible.
pub(crate) struct UniformStream {
    state: u64,
}

impl UniformStream {
    pub(crate) fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// A value in [0, 1) with 53 random bits.
    pub(crate) fn next_uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        (mixed >> 11) as f64 / (1_u64 << 53) as f64
    }
}
