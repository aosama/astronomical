//! The `astronomical models` verb: list installed models, show the release
//! catalog, show or persist the effective default model, and drive a
//! download with live progress. Report data goes to stdout; download
//! progress goes to stderr.

use std::{fmt::Write as _, io::Write, path::PathBuf, time::Duration};

use astronomical_ipc_protocol::{DaemonCatalogEntry, DaemonListedModel};

use crate::{
    DaemonProbe, errors::ModelsError, formatting::format_gigabytes,
    model_lifecycle::ModelLifecycle, models_arguments::ModelsCommand,
};

/// Collaborators the models verb needs, injected so tests can stub them.
pub struct ModelsDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<PathBuf>,
    /// Where the report data goes.
    pub stdout: &'a mut dyn Write,
    /// Where download progress goes.
    pub stderr: &'a mut dyn Write,
    /// Bound for each protocol exchange.
    pub request_timeout: Duration,
    /// Bound for the whole download-wait stage.
    pub download_stage_bound: Duration,
    /// Wait between download status polls.
    pub download_poll_interval: Duration,
}

/// Runs one models subcommand against the resident daemon.
pub async fn run_models(
    models_command: &ModelsCommand,
    models_dependencies: &mut ModelsDependencies<'_>,
) -> Result<(), ModelsError> {
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: models_dependencies.candidate_socket_paths.clone(),
        request_timeout: models_dependencies.request_timeout,
    };
    match models_command {
        ModelsCommand::List => {
            let installed_models = daemon_probe.models_list().await?;
            write_installed_models(models_dependencies.stdout, &installed_models)
        }
        ModelsCommand::Supported => {
            let catalog_entries = daemon_probe.catalog().await?;
            write_catalog(models_dependencies.stdout, &catalog_entries)
        }
        ModelsCommand::Default { model_id } => match model_id {
            Some(model_id) => {
                let model_lifecycle = ModelLifecycle {
                    daemon_probe,
                    download_stage_bound: models_dependencies.download_stage_bound,
                    download_poll_interval: models_dependencies.download_poll_interval,
                };
                // Persist the choice only once the model can actually serve:
                // a default pointing at a model this Mac lacks is fetched
                // first, and an id outside the catalog is rejected here.
                let resolved_model_id = model_lifecycle
                    .ensure_downloaded(model_id, &mut |progress_line| {
                        let _ = write!(models_dependencies.stderr, "\r{progress_line}");
                        let _ = models_dependencies.stderr.flush();
                    })
                    .await?;
                let persisted_model_id = model_lifecycle
                    .daemon_probe
                    .default_model_set(&resolved_model_id)
                    .await?;
                writeln!(
                    &mut *models_dependencies.stdout,
                    "default model: {persisted_model_id}"
                )
                .map_err(stdout_unwritable)
            }
            None => {
                let status_snapshot = daemon_probe.status_snapshot().await?;
                writeln!(
                    &mut *models_dependencies.stdout,
                    "default model: {}",
                    status_snapshot
                        .default_model_id
                        .as_deref()
                        .unwrap_or("none")
                )
                .map_err(stdout_unwritable)
            }
        },
        ModelsCommand::Download { model_id } => {
            let model_lifecycle = ModelLifecycle {
                daemon_probe,
                download_stage_bound: models_dependencies.download_stage_bound,
                download_poll_interval: models_dependencies.download_poll_interval,
            };
            let resolved_model_id = model_lifecycle
                .ensure_downloaded(model_id, &mut |progress_line| {
                    let _ = write!(models_dependencies.stderr, "\r{progress_line}");
                    let _ = models_dependencies.stderr.flush();
                })
                .await?;
            let _ = writeln!(
                models_dependencies.stderr,
                "\n{resolved_model_id} is available"
            );
            Ok(())
        }
    }
}

fn stdout_unwritable(output_error: std::io::Error) -> ModelsError {
    ModelsError::StdoutUnwritable {
        cause: output_error.to_string(),
    }
}

/// One aligned line per installed model; `*` marks the resident one.
fn write_installed_models(
    stdout: &mut dyn Write,
    installed_models: &[DaemonListedModel],
) -> Result<(), ModelsError> {
    let model_width = installed_models
        .iter()
        .map(|listed_model| listed_model.model_id.len())
        .max()
        .unwrap_or("MODEL".len());
    let mut report = String::new();
    let _ = writeln!(
        report,
        "{:<8} {model:<model_width$} {family_header:<10} {context_header:<10} SIZE",
        "",
        model = "MODEL",
        family_header = "FAMILY",
        context_header = "CONTEXT",
        model_width = model_width,
    );
    for listed_model in installed_models {
        let _ = writeln!(
            report,
            "{marker:<8} {model_id:<model_width$} {family:<10} {context:<10} {size}",
            marker = if listed_model.is_resident { "*" } else { "" },
            model_id = listed_model.model_id,
            family = listed_model.family,
            context = listed_model
                .context_window
                .map(|context_window| context_window.to_string())
                .unwrap_or_default(),
            size = format_gigabytes(listed_model.size_bytes),
            model_width = model_width,
        );
    }
    stdout
        .write_all(report.as_bytes())
        .map_err(|output_error| ModelsError::StdoutUnwritable {
            cause: output_error.to_string(),
        })
}

/// One aligned line per catalog entry; STATE is `ready`, the active download
/// state, or `not on this Mac`.
fn write_catalog(
    stdout: &mut dyn Write,
    catalog_entries: &[DaemonCatalogEntry],
) -> Result<(), ModelsError> {
    let model_width = catalog_entries
        .iter()
        .map(|catalog_entry| {
            catalog_entry
                .requestable_model_id
                .as_deref()
                .unwrap_or(&catalog_entry.huggingface_id)
                .len()
        })
        .max()
        .unwrap_or("MODEL".len());
    let mut report = String::new();
    let _ = writeln!(
        report,
        "{model_header:<model_width$} {family_header:<10} {size_header:<8} STATE",
        model_header = "MODEL",
        family_header = "FAMILY",
        size_header = "SIZE",
        model_width = model_width,
    );
    for catalog_entry in catalog_entries {
        let model_id = catalog_entry
            .requestable_model_id
            .as_deref()
            .unwrap_or(&catalog_entry.huggingface_id);
        let entry_state = if catalog_entry.ready_on_this_mac {
            "ready".to_owned()
        } else if let Some(download_state) = &catalog_entry.download_state {
            download_state.clone()
        } else {
            "not on this Mac".to_owned()
        };
        let _ = writeln!(
            report,
            "{model_id:<model_width$} {family:<10} {size:<8} {entry_state}",
            model_id = model_id,
            family = catalog_entry.family,
            size = format_gigabytes(catalog_entry.approximate_size_bytes),
            model_width = model_width,
        );
    }
    stdout
        .write_all(report.as_bytes())
        .map_err(|output_error| ModelsError::StdoutUnwritable {
            cause: output_error.to_string(),
        })
}
