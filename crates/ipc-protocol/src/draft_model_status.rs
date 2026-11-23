//! Whether the loaded model drafts with a DFlash draft model, and if not, why.

use serde::{Deserialize, Serialize};

/// Reported by the engine once it has looked for a drafter at load, so a person
/// or a tool can tell whether the speed a drafter package promises is in play
/// without reading the worker log. It describes the loaded model; a request
/// whose settings drafting cannot reproduce (structured output, a picture in
/// the prompt, a repetition penalty) still decodes one token at a time.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum DraftModelStatus {
    /// No drafter pairs with the loaded model.
    None,
    /// A paired drafter is loaded and requests draft with it.
    Drafting {
        generation: DraftModelGeneration,
        location: DraftModelLocation,
        /// The most tokens one verify step checks, the draft tree's size.
        tree_node_budget: u32,
    },
    /// A paired drafter is installed but this load does not use it.
    /// `reason` is bounded and carries no local path.
    Unavailable { reason: String },
}

/// Which DFlash design the drafter follows.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftModelGeneration {
    Dflash,
    Dflash2,
}

/// Where the drafter was found.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftModelLocation {
    /// The `drafter` folder inside the model's own directory.
    Bundled,
    /// A drafter elsewhere in the model library that pairs with this model.
    Library,
}

impl DraftModelStatus {
    /// Whether requests the drafter can serve will draft.
    #[must_use]
    pub const fn is_drafting(&self) -> bool {
        matches!(self, Self::Drafting { .. })
    }
}

#[cfg(test)]
#[path = "draft_model_status_tests.rs"]
mod tests;
