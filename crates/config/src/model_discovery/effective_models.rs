//! Effective model discovery in the precedence the supervisor daemon serves.
//!
//! One automatic Library destination (the instance state models directory) is scanned first and
//! wins identity collisions with the authored config roots, because Library publication owns
//! that destination. The supervisor resolver and the model-serving acceptance harness both
//! resolve through this module, so journeys cannot drift from what the daemon advertises.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::model_discovery::{
    DiscoveredModel, DiscoveredModelError, ModelDiscoveryDiagnostic, discover_models,
    discover_models_excluding_ambiguous_identities,
};

/// Why effective model discovery could not complete.
#[derive(Debug, Error)]
pub enum EffectiveModelDiscoveryError {
    /// Metadata access failed before the optional automatic root could be classified as present.
    #[error("failed to inspect automatic model directory {directory_path:?}: {source}")]
    AutomaticModelDirectoryMetadata {
        directory_path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Discovered(#[from] DiscoveredModelError),
}

/// Executable models plus diagnostics in the precedence the supervisor serves.
#[derive(Debug)]
pub struct EffectiveModelDiscovery {
    pub discovered_models: Vec<DiscoveredModel>,
    pub diagnostics: Vec<ModelDiscoveryDiagnostic>,
}

/// Discovers executable models across one automatic Library root plus the authored config roots,
/// in the production precedence the supervisor serves: the automatic Library destination wins
/// identity collisions with authored roots, and an absent automatic root is simply skipped.
pub fn discover_effective_models(
    automatic_models_directory: &Path,
    configured_model_directories: &[PathBuf],
) -> Result<EffectiveModelDiscovery, EffectiveModelDiscoveryError> {
    match automatic_models_directory.try_exists() {
        Ok(true) => {
            let mut effective_models =
                discover_models(&[automatic_models_directory.to_path_buf()])?
                    .into_iter()
                    .flat_map(|directory_scan| directory_scan.discovered_models)
                    .collect::<Vec<_>>();
            let automatic_model_ids = effective_models
                .iter()
                .map(|discovered_model| discovered_model.model_id.clone())
                .collect::<HashSet<_>>();
            let mut configured_discovery_report =
                discover_models_excluding_ambiguous_identities(configured_model_directories)?;
            let configured_models = configured_discovery_report
                .directory_scans
                .into_iter()
                .flat_map(|directory_scan| directory_scan.discovered_models)
                .filter(|discovered_model| {
                    !automatic_model_ids.contains(&discovered_model.model_id)
                });
            // Library publication owns the automatic destination, so authored ambiguity cannot
            // hide a validated Library copy of the same public identity.
            effective_models.extend(configured_models);
            configured_discovery_report
                .diagnostics
                .retain(|diagnostic| !automatic_model_ids.contains(&diagnostic.model_id));
            Ok(EffectiveModelDiscovery {
                discovered_models: effective_models,
                diagnostics: configured_discovery_report.diagnostics,
            })
        }
        Ok(false) => {
            let discovery_report =
                discover_models_excluding_ambiguous_identities(configured_model_directories)?;
            Ok(EffectiveModelDiscovery {
                discovered_models: discovery_report
                    .directory_scans
                    .into_iter()
                    .flat_map(|directory_scan| directory_scan.discovered_models)
                    .collect(),
                diagnostics: discovery_report.diagnostics,
            })
        }
        Err(source) => Err(
            EffectiveModelDiscoveryError::AutomaticModelDirectoryMetadata {
                directory_path: automatic_models_directory.to_path_buf(),
                source,
            },
        ),
    }
}
