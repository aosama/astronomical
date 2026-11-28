//! What the gated-delta recurrence tests share: one MLX runtime, and the
//! recurrence's six inputs held together.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::gated_delta_inputs::GatedDeltaInputs;

/// A direct MLX runtime sized for the small recurrence tests.
pub(super) fn gated_delta_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the gated-delta test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

/// The recurrence's inputs, owned, in the order the layer passes them.
pub(super) struct RecurrenceInputs {
    pub(super) queries: MlxArray,
    pub(super) keys: MlxArray,
    pub(super) values: MlxArray,
    pub(super) decays: MlxArray,
    pub(super) update_rates: MlxArray,
    pub(super) recurrent_state: MlxArray,
}

impl RecurrenceInputs {
    pub(super) fn as_gated_delta_inputs(&self) -> GatedDeltaInputs<'_> {
        GatedDeltaInputs {
            queries: &self.queries,
            keys: &self.keys,
            values: &self.values,
            decays: &self.decays,
            update_rates: &self.update_rates,
            recurrent_state: &self.recurrent_state,
        }
    }
}
