//! Which model is loading, reported while the status alone cannot say.

use serde::{Deserialize, Serialize};

/// Sent when a load starts and cleared when it ends. A load is bounded by the
/// daemon's budget and reports no steps, so this says only that it is happening;
/// the slow Neural Engine programs come afterwards and report through
/// `NeuralEnginePrefillStatus`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLoadInFlight {
    /// The model being loaded, which during a swap is not the model the status still
    /// reports as ready.
    pub model_id: Option<String>,
}

impl ModelLoadInFlight {
    /// A load of a model the reporter cannot name.
    #[must_use]
    pub const fn unnamed() -> Self {
        Self { model_id: None }
    }

    #[must_use]
    pub fn for_model(model_id: &str) -> Self {
        Self {
            model_id: Some(model_id.to_owned()),
        }
    }
}

#[cfg(test)]
#[path = "model_load_in_flight_tests.rs"]
mod tests;
