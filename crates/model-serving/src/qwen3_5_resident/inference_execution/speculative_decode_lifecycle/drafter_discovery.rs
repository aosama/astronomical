//! Finding the DFlash2 drafter installed for a target.
//!
//! A drafter has no vocabulary of its own: it borrows the target's
//! embedding, head and tapped hidden rows, so it is recognized by structure
//! rather than by name. A model directory is a drafter when its
//! `config.json` carries a `dflash_config`, and it pairs with a target when
//! it names that target's layer count, hidden width and vocabulary. Every
//! token a drafter proposes is still verified by the target, so a pairing
//! can cost speed but never correctness.
//!
//! A drafter bundled inside the target's own directory, in a `drafter`
//! folder, is tried first. It belongs to that one model, so a library can
//! hold the same target with and without drafting side by side, which is how
//! a draft-tree build is compared with its plain twin. A library-wide drafter
//! pairs with every target of its shape.
//!
//! Two library layouts are then searched from the target's own directory:
//! the `huggingface_hub_cache` (`models--org--name/snapshots/<revision>`) and
//! a plain library (`<organization>/<model>`).

use std::path::{Path, PathBuf};

use crate::speculative_decoding::dflash2_drafter::DFlash2DrafterConfiguration;

use super::engine_draft_runtime::Qwen3_5SpeculativeDraftLoadingError;

const HUGGINGFACE_SNAPSHOTS_DIRECTORY_NAME: &str = "snapshots";
const CONFIGURATION_FILE_NAME: &str = "config.json";
/// The folder inside a target's directory that holds its own drafter.
pub const BUNDLED_DRAFTER_DIRECTORY_NAME: &str = "drafter";
/// A drafter's `config.json` is a few kilobytes; larger files are other
/// things and are not read.
const MAXIMUM_CONFIGURATION_BYTES: u64 = 1024 * 1024;

/// The target facts a drafter must name to pair with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrafterPairing {
    pub target_layer_count: i32,
    pub target_hidden_size: i32,
    pub target_vocabulary_size: i32,
}

impl DrafterPairing {
    /// Whether `drafter_configuration` was built for a target of this shape.
    #[must_use]
    pub fn pairs_with(&self, drafter_configuration: &DFlash2DrafterConfiguration) -> bool {
        drafter_configuration.target_layer_count == self.target_layer_count
            && drafter_configuration.hidden_size == self.target_hidden_size
            && drafter_configuration.vocabulary_size == self.target_vocabulary_size
    }
}

/// The drafter that pairs with the target in `target_model_directory`, with
/// its configuration: the target's bundled drafter when it pairs, otherwise
/// the first library drafter (in path order) that does.
pub fn paired_drafter_directory(
    target_model_directory: &Path,
    pairing: &DrafterPairing,
) -> Result<Option<(PathBuf, DFlash2DrafterConfiguration)>, Qwen3_5SpeculativeDraftLoadingError> {
    let mut library_directories = sibling_model_directories(target_model_directory)?;
    library_directories.sort();
    let mut candidate_directories =
        vec![target_model_directory.join(BUNDLED_DRAFTER_DIRECTORY_NAME)];
    candidate_directories.extend(library_directories);
    for candidate_directory in candidate_directories {
        let Some(config_document) = drafter_config_document(&candidate_directory) else {
            continue;
        };
        // A drafter of a kind this engine does not implement is not this
        // target's drafter: it is passed over, never a load failure.
        let drafter_configuration =
            match DFlash2DrafterConfiguration::from_config_document(&config_document) {
                Ok(drafter_configuration) => drafter_configuration,
                Err(configuration_error) => {
                    tracing::debug!(
                        drafter_directory = %candidate_directory.display(),
                        %configuration_error,
                        "passing over a drafter this engine does not implement"
                    );
                    continue;
                }
            };
        if pairing.pairs_with(&drafter_configuration) {
            return Ok(Some((candidate_directory, drafter_configuration)));
        }
    }
    Ok(None)
}

/// Every model directory in the library the target sits in, whichever of
/// the two layouts that is.
fn sibling_model_directories(
    target_model_directory: &Path,
) -> Result<Vec<PathBuf>, Qwen3_5SpeculativeDraftLoadingError> {
    let parent_directory = target_model_directory.parent();
    let is_hub_snapshot = parent_directory
        .and_then(Path::file_name)
        .is_some_and(|name| name == HUGGINGFACE_SNAPSHOTS_DIRECTORY_NAME);
    let library_root = if is_hub_snapshot {
        parent_directory
            .and_then(Path::parent)
            .and_then(Path::parent)
    } else {
        parent_directory.and_then(Path::parent)
    };
    let Some(library_root) = library_root else {
        return Ok(Vec::new());
    };
    let mut model_directories = Vec::new();
    for first_level in child_directories(library_root)? {
        let second_level_root = if is_hub_snapshot {
            first_level.join(HUGGINGFACE_SNAPSHOTS_DIRECTORY_NAME)
        } else {
            first_level
        };
        if second_level_root.is_dir() {
            model_directories.extend(child_directories(&second_level_root)?);
        }
    }
    Ok(model_directories)
}

fn child_directories(
    directory: &Path,
) -> Result<Vec<PathBuf>, Qwen3_5SpeculativeDraftLoadingError> {
    let entries = std::fs::read_dir(directory)
        .map_err(|source| Qwen3_5SpeculativeDraftLoadingError::LibraryUnlistable { source })?;
    let mut directories = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            directories.push(path);
        }
    }
    Ok(directories)
}

/// The directory's `config.json` when it describes a DFlash2 drafter; None
/// for any other model, or a file that is missing, oversized or not JSON.
fn drafter_config_document(model_directory: &Path) -> Option<serde_json::Value> {
    let config_path = model_directory.join(CONFIGURATION_FILE_NAME);
    let config_bytes = std::fs::metadata(&config_path).ok()?.len();
    if config_bytes > MAXIMUM_CONFIGURATION_BYTES {
        return None;
    }
    let config_document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).ok()?).ok()?;
    config_document.get("dflash_config")?;
    Some(config_document)
}
