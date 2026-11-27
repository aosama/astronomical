//! The six arrays one gated-delta recurrence reads, passed together so each
//! kernel and fallback names them once instead of restating six arguments.

use mlx_bridge::MlxArray;

/// Borrowed views of one recurrence's inputs. The struct only groups them: it
/// holds no copy, so building one costs nothing and every field stays the
/// caller's own array, in the order the fused kernels take them.
#[derive(Clone, Copy)]
pub struct GatedDeltaInputs<'arrays> {
    pub queries: &'arrays MlxArray,
    pub keys: &'arrays MlxArray,
    pub values: &'arrays MlxArray,
    pub decays: &'arrays MlxArray,
    pub update_rates: &'arrays MlxArray,
    pub recurrent_state: &'arrays MlxArray,
}
