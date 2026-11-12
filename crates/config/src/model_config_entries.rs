//! What each `models` entry does against the installed models. The daemon's unmatched-entry
//! report and `astronomical validate config` both read this one classification, so the CLI never
//! tells someone an entry is used when the daemon ignores it.
//!
//! `models` settings (generation defaults, limits, chunking) only change chat models: the daemon
//! reads them for chat models alone, so an entry that names a picture or embedding model is
//! reported as unused rather than as applying.

use crate::{
    AstronomicalConfig, DiscoveredModel, ModelCapabilities, configured_model_id_for, leaf_model_id,
};

/// What kind of non-chat model an entry names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettingsFreeModelKind {
    /// Makes pictures from a description.
    ImageGeneration,
    /// Turns text into numbers for search.
    Embeddings,
    /// Answers typed questions with probabilities.
    Decisions,
    /// Reads text aloud.
    Speech,
}

/// The effect of one `models` entry on the installed models.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelConfigEntryOutcome {
    /// The entry configures this installed chat model.
    Applies { model_id: String },
    /// The entry names an installed model that `models` settings do not change.
    ModelTakesNoSettings {
        model_id: String,
        model_kind: SettingsFreeModelKind,
    },
    /// An installed model has this short name, but it records another organisation's full ID.
    DifferentOrganisation {
        model_id: String,
        installed_provider_model_id: String,
    },
    /// Another entry configures the installed model of this name instead.
    ConfiguredByAnotherEntry {
        model_id: String,
        applied_entry_id: String,
    },
    /// No installed model has this name; the entry waits until one is installed.
    MatchesNoInstalledModel,
}

/// One `models` key and its effect.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelConfigEntryReport {
    pub configured_model_id: String,
    pub outcome: ModelConfigEntryOutcome,
}

impl ModelConfigEntryReport {
    #[must_use]
    pub fn applies(&self) -> bool {
        matches!(self.outcome, ModelConfigEntryOutcome::Applies { .. })
    }
}

/// One discovered model and the entry the matching rule picks for it.
struct ModelEntryAssignment<'discovered> {
    discovered_model: &'discovered DiscoveredModel,
    applied_entry_id: Option<String>,
}

impl AstronomicalConfig {
    /// Classifies every `models` entry, in key order, against `discovered_models`.
    #[must_use]
    pub fn model_config_entry_reports(
        &self,
        discovered_models: &[DiscoveredModel],
    ) -> Vec<ModelConfigEntryReport> {
        // The rule is evaluated once per model, not once per entry and model, because the
        // daemon runs this on every reload and Library refresh.
        let configured_model_ids = self.configured_model_ids();
        let mut assignments = Vec::with_capacity(discovered_models.len());
        for discovered_model in discovered_models {
            let applied_entry_id = configured_model_id_for(
                &discovered_model.model_id,
                discovered_model.provider_model_id.as_deref(),
                configured_model_ids.iter().copied(),
            );
            assignments.push(ModelEntryAssignment {
                discovered_model,
                applied_entry_id: applied_entry_id.map(str::to_owned),
            });
        }
        let mut entry_reports = Vec::with_capacity(configured_model_ids.len());
        for configured_model_id in configured_model_ids {
            entry_reports.push(ModelConfigEntryReport {
                configured_model_id: configured_model_id.to_owned(),
                outcome: model_config_entry_outcome(configured_model_id, &assignments),
            });
        }
        entry_reports
    }
}

fn model_config_entry_outcome(
    configured_model_id: &str,
    assignments: &[ModelEntryAssignment<'_>],
) -> ModelConfigEntryOutcome {
    for assignment in assignments {
        if assignment.applied_entry_id.as_deref() == Some(configured_model_id) {
            return applied_outcome(assignment.discovered_model);
        }
    }
    for assignment in assignments {
        if assignment.discovered_model.model_id == leaf_model_id(configured_model_id) {
            return unused_outcome(configured_model_id, assignment);
        }
    }
    ModelConfigEntryOutcome::MatchesNoInstalledModel
}

fn applied_outcome(discovered_model: &DiscoveredModel) -> ModelConfigEntryOutcome {
    let model_id = discovered_model.model_id.clone();
    match settings_free_kind_of(&discovered_model.capabilities) {
        Some(model_kind) => ModelConfigEntryOutcome::ModelTakesNoSettings {
            model_id,
            model_kind,
        },
        None => ModelConfigEntryOutcome::Applies { model_id },
    }
}

/// Which kind of settings-free model these capabilities describe, or `None`
/// for a chat model, the only kind a `models` entry configures.
fn settings_free_kind_of(model_capabilities: &ModelCapabilities) -> Option<SettingsFreeModelKind> {
    crate::exhaustive_table!(model_capabilities, {
        ModelCapabilities::Chat => None,
        ModelCapabilities::ImageGeneration => Some(SettingsFreeModelKind::ImageGeneration),
        ModelCapabilities::Embeddings => Some(SettingsFreeModelKind::Embeddings),
        ModelCapabilities::Decisions => Some(SettingsFreeModelKind::Decisions),
        ModelCapabilities::Speech => Some(SettingsFreeModelKind::Speech),
    })
}

/// Why an entry naming an installed model is not the one the model uses. A recorded full ID
/// from another organisation is the reason that holds even after other entries are removed, so
/// it is reported before any other entry that happens to configure the model.
fn unused_outcome(
    configured_model_id: &str,
    assignment: &ModelEntryAssignment<'_>,
) -> ModelConfigEntryOutcome {
    let model_id = assignment.discovered_model.model_id.clone();
    if let Some(installed_provider_model_id) = &assignment.discovered_model.provider_model_id
        && installed_provider_model_id != configured_model_id
        && configured_model_id != model_id
    {
        return ModelConfigEntryOutcome::DifferentOrganisation {
            model_id,
            installed_provider_model_id: installed_provider_model_id.clone(),
        };
    }
    match &assignment.applied_entry_id {
        Some(applied_entry_id) => ModelConfigEntryOutcome::ConfiguredByAnotherEntry {
            model_id,
            applied_entry_id: applied_entry_id.clone(),
        },
        None => ModelConfigEntryOutcome::MatchesNoInstalledModel,
    }
}

#[cfg(test)]
#[path = "model_config_entries_tests.rs"]
mod model_config_entries_tests;
