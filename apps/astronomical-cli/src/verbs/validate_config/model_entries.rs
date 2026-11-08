//! The `models` part of the `validate config` report: what each entry does against the models
//! installed on disk now, by the same rule the daemon applies (`model_config_entry_reports`).
//!
//! Discovery runs here, in the CLI, instead of asking the daemon, because people usually check
//! their configuration while Astronomical is stopped. The cost is timing: a model copied in by
//! hand while Astronomical runs is counted here before the daemon notices it, so the report says
//! it describes the models on disk now.

use std::path::PathBuf;

use astronomical_config::{
    AstronomicalConfig, DiscoveredModelError, EffectiveModelDiscoveryError,
    ModelConfigEntryOutcome, ModelConfigEntryReport, SettingsFreeModelKind,
};

/// What the report knows about the `models` entries.
pub(super) enum ModelEntriesCheck {
    /// No `models` entries, so there was nothing to check and no disk scan.
    NoEntries,
    Checked(Vec<ModelConfigEntryReport>),
    /// The installed models could not be listed. The daemon refuses to load in this state, so
    /// the report says why in plain words instead of passing silently.
    DiscoveryFailed {
        configured_model_ids: Vec<String>,
        discovery_error: EffectiveModelDiscoveryError,
    },
}

fn check_model_entries(loaded_config: &AstronomicalConfig) -> ModelEntriesCheck {
    let configured_model_ids = loaded_config.configured_model_ids();
    if configured_model_ids.is_empty() {
        return ModelEntriesCheck::NoEntries;
    }
    let models_directory = loaded_config.instance_paths().models_directory();
    match astronomical_config::discover_effective_models(
        &models_directory,
        loaded_config.model_directories(),
    ) {
        Ok(effective_discovery) => ModelEntriesCheck::Checked(
            loaded_config.model_config_entry_reports(&effective_discovery.discovered_models),
        ),
        Err(discovery_error) => ModelEntriesCheck::DiscoveryFailed {
            configured_model_ids: configured_model_ids
                .into_iter()
                .map(str::to_owned)
                .collect(),
            discovery_error,
        },
    }
}

/// Checks the entries and, when the installed models could not be listed, says so on
/// `warning_output` as well as in the report, since the daemon refuses to load in that state.
pub(super) fn check_model_entries_and_warn(
    loaded_config: &AstronomicalConfig,
    warning_output: &mut dyn std::io::Write,
) -> std::io::Result<ModelEntriesCheck> {
    let model_entries_check = check_model_entries(loaded_config);
    if let ModelEntriesCheck::DiscoveryFailed {
        discovery_error, ..
    } = &model_entries_check
    {
        writeln!(
            warning_output,
            "warning: the model settings were not checked: {}",
            discovery_problem_sentence(discovery_error)
        )?;
    }
    Ok(model_entries_check)
}

fn discovery_problem_sentence(discovery_error: &EffectiveModelDiscoveryError) -> String {
    match discovery_error {
        EffectiveModelDiscoveryError::Discovered(DiscoveredModelError::DuplicateModelId {
            model_id,
            model_directories,
        }) => format!(
            "two installed models are both called {model_id} ({}). Astronomical cannot start \
             until one of them is removed or moved.",
            rendered_directories(model_directories)
        ),
        EffectiveModelDiscoveryError::Discovered(DiscoveredModelError::ReadDirectory {
            directory_path,
            ..
        })
        | EffectiveModelDiscoveryError::AutomaticModelDirectoryMetadata {
            directory_path, ..
        } => format!(
            "the folder {} could not be read, so the installed models could not be listed.",
            directory_path.display()
        ),
    }
}

fn rendered_directories(model_directories: &[PathBuf]) -> String {
    let mut rendered_paths = Vec::with_capacity(model_directories.len());
    for model_directory in model_directories {
        rendered_paths.push(model_directory.display().to_string());
    }
    rendered_paths.join(" and ")
}

pub(super) fn push_text_lines(
    model_entries_check: &ModelEntriesCheck,
    report_lines: &mut Vec<String>,
) {
    match model_entries_check {
        ModelEntriesCheck::NoEntries => report_lines.push("Model settings: none".to_owned()),
        ModelEntriesCheck::Checked(entry_reports) => {
            report_lines.push(format!(
                "Model settings: {} (checked against the models installed now)",
                entry_reports.len()
            ));
            for entry_report in entry_reports {
                report_lines.push(format!(
                    "  {}: {}",
                    entry_report.configured_model_id,
                    outcome_sentence(&entry_report.outcome)
                ));
            }
        }
        ModelEntriesCheck::DiscoveryFailed {
            configured_model_ids,
            discovery_error,
        } => {
            report_lines.push(format!(
                "Model settings: {} (not checked: {})",
                configured_model_ids.len(),
                discovery_problem_sentence(discovery_error)
            ));
            for configured_model_id in configured_model_ids {
                report_lines.push(format!("  {configured_model_id}"));
            }
        }
    }
}

fn outcome_sentence(outcome: &ModelConfigEntryOutcome) -> String {
    match outcome {
        ModelConfigEntryOutcome::Applies { model_id } => format!("used for {model_id}"),
        ModelConfigEntryOutcome::ModelTakesNoSettings {
            model_id,
            model_kind,
        } => format!(
            "not used: {model_id} {}, and these settings only change chat models",
            settings_free_model_description(*model_kind)
        ),
        ModelConfigEntryOutcome::DifferentOrganisation {
            model_id,
            installed_provider_model_id,
        } => format!(
            "not used: your installed {model_id} is {installed_provider_model_id}. Use that \
             name instead."
        ),
        ModelConfigEntryOutcome::ConfiguredByAnotherEntry {
            model_id,
            applied_entry_id,
        } => format!("not used: the settings under {applied_entry_id} are used for {model_id}"),
        ModelConfigEntryOutcome::MatchesNoInstalledModel => {
            "waiting: no installed model has this name yet. It will be used once you download \
             it, so check the spelling if you expected it to work now."
                .to_owned()
        }
    }
}

fn settings_free_model_description(model_kind: SettingsFreeModelKind) -> &'static str {
    match model_kind {
        SettingsFreeModelKind::ImageGeneration => "makes pictures",
        SettingsFreeModelKind::Embeddings => "turns text into numbers for search",
    }
}

pub(super) fn json_document(model_entries_check: &ModelEntriesCheck) -> serde_json::Value {
    match model_entries_check {
        ModelEntriesCheck::NoEntries => {
            serde_json::json!({"checked": true, "problem": null, "entries": []})
        }
        ModelEntriesCheck::Checked(entry_reports) => {
            let mut entry_documents = Vec::with_capacity(entry_reports.len());
            for entry_report in entry_reports {
                entry_documents.push(entry_json_document(entry_report));
            }
            serde_json::json!({"checked": true, "problem": null, "entries": entry_documents})
        }
        ModelEntriesCheck::DiscoveryFailed {
            configured_model_ids,
            discovery_error,
        } => {
            let mut entry_documents = Vec::with_capacity(configured_model_ids.len());
            for configured_model_id in configured_model_ids {
                entry_documents.push(serde_json::json!({
                    "configured_model_id": configured_model_id,
                    "outcome": "not_checked",
                }));
            }
            serde_json::json!({
                "checked": false,
                "problem": discovery_problem_sentence(discovery_error),
                "entries": entry_documents,
            })
        }
    }
}

fn entry_json_document(entry_report: &ModelConfigEntryReport) -> serde_json::Value {
    let configured_model_id = &entry_report.configured_model_id;
    match &entry_report.outcome {
        ModelConfigEntryOutcome::Applies { model_id } => serde_json::json!({
            "configured_model_id": configured_model_id, "outcome": "applies", "model_id": model_id,
        }),
        ModelConfigEntryOutcome::ModelTakesNoSettings {
            model_id,
            model_kind,
        } => serde_json::json!({
            "configured_model_id": configured_model_id, "outcome": "model_takes_no_settings",
            "model_id": model_id,
            "model_kind": match model_kind {
                SettingsFreeModelKind::ImageGeneration => "image_generation",
                SettingsFreeModelKind::Embeddings => "embeddings",
            },
        }),
        ModelConfigEntryOutcome::DifferentOrganisation {
            model_id,
            installed_provider_model_id,
        } => serde_json::json!({
            "configured_model_id": configured_model_id, "outcome": "different_organisation",
            "model_id": model_id, "installed_provider_model_id": installed_provider_model_id,
        }),
        ModelConfigEntryOutcome::ConfiguredByAnotherEntry {
            model_id,
            applied_entry_id,
        } => serde_json::json!({
            "configured_model_id": configured_model_id, "outcome": "configured_by_another_entry",
            "model_id": model_id, "applied_entry_id": applied_entry_id,
        }),
        ModelConfigEntryOutcome::MatchesNoInstalledModel => serde_json::json!({
            "configured_model_id": configured_model_id, "outcome": "matches_no_installed_model",
        }),
    }
}
