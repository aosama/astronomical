//! Finding the DFlash drafter installed for a target.
//!
//! A drafter has no vocabulary of its own: it borrows the target's
//! embedding, head and tapped hidden rows, so it is recognized by structure
//! rather than by name. A model directory is a drafter when its
//! `config.json` carries a `dflash_config`, and it pairs with a target when
//! it names that target's layer count, hidden width and vocabulary. Every
//! token a drafter proposes is still verified by the target, so a pairing
//! can cost speed but never correctness.
//!
//! A drafter bundled inside the target's own directory, in a `drafter` or
//! `dflash` folder (`BUNDLED_DRAFTER_FOLDER_NAMES`), is tried first. It belongs to that one model, so a library can
//! hold the same target with and without drafting side by side, which is how
//! a draft-tree build is compared with its plain twin. A library-wide drafter
//! pairs with every target of its shape. When several library drafters pair,
//! a DFlash2 drafter is preferred over a DFlash one and path order decides
//! only within a generation. The order is fixed for stability, not speed:
//! libraries held only DFlash2 drafters before DFlash was served, so
//! downloading a DFlash drafter never silently changes the one a model
//! already drafts with. Which generation drafts faster depends on the target
//! and the text; a package that wants one ships it in its `drafter` folder.
//!
//! Two library layouts are then searched from the target's own directory:
//! the `huggingface_hub_cache` (`models--org--name/snapshots/<revision>`) and
//! a plain library (`<organization>/<model>`).

use std::path::{Path, PathBuf};

use ipc_protocol::BUNDLED_DRAFTER_FOLDER_NAMES;

use super::dflash_drafter::{DFlashDrafterConfiguration, DrafterGeneration};

/// Why the library could not be searched for a drafter.
#[derive(Debug, thiserror::Error)]
pub enum DrafterDiscoveryError {
    #[error("the model library directory cannot be listed to find a DFlash drafter: {source}")]
    LibraryUnlistable { source: std::io::Error },
}

const HUGGINGFACE_SNAPSHOTS_DIRECTORY_NAME: &str = "snapshots";
const CONFIGURATION_FILE_NAME: &str = "config.json";
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
    pub fn pairs_with(&self, drafter_configuration: &DFlashDrafterConfiguration) -> bool {
        drafter_configuration.target_layer_count == self.target_layer_count
            && drafter_configuration.hidden_size == self.target_hidden_size
            && drafter_configuration.vocabulary_size == self.target_vocabulary_size
    }
}

/// The drafter that pairs with the target in `target_model_directory`, with
/// its configuration: the target's bundled drafter when it pairs, otherwise
/// the library drafter that pairs, DFlash2 before DFlash and then in path
/// order.
pub fn paired_drafter_directory(
    target_model_directory: &Path,
    pairing: &DrafterPairing,
) -> Result<Option<(PathBuf, DFlashDrafterConfiguration)>, DrafterDiscoveryError> {
    if let Some(bundled_drafter) = bundled_pairing_drafter(target_model_directory, pairing) {
        return Ok(Some(bundled_drafter));
    }
    let mut library_directories = sibling_model_directories(target_model_directory)?;
    library_directories.sort();
    Ok(preferred_library_drafter(library_directories, pairing))
}

/// The target's own drafter folder that pairs with it, tried in
/// `BUNDLED_DRAFTER_FOLDER_NAMES` order.
fn bundled_pairing_drafter(
    target_model_directory: &Path,
    pairing: &DrafterPairing,
) -> Option<(PathBuf, DFlashDrafterConfiguration)> {
    for bundled_folder_name in BUNDLED_DRAFTER_FOLDER_NAMES {
        let bundled_directory = target_model_directory.join(bundled_folder_name);
        if let Some(drafter_configuration) = pairing_configuration(&bundled_directory, pairing) {
            return Some((bundled_directory, drafter_configuration));
        }
    }
    None
}

/// The pairing drafter among `library_directories` (in path order): the
/// first DFlash2 one, or the first DFlash one when no DFlash2 drafter pairs.
fn preferred_library_drafter(
    library_directories: Vec<PathBuf>,
    pairing: &DrafterPairing,
) -> Option<(PathBuf, DFlashDrafterConfiguration)> {
    let mut chosen: Option<(PathBuf, DFlashDrafterConfiguration)> = None;
    for library_directory in library_directories {
        let Some(drafter_configuration) = pairing_configuration(&library_directory, pairing) else {
            continue;
        };
        let is_preferred = match &chosen {
            None => true,
            Some((_, chosen_configuration)) => {
                generation_preference(drafter_configuration.generation)
                    < generation_preference(chosen_configuration.generation)
            }
        };
        if is_preferred {
            chosen = Some((library_directory, drafter_configuration));
        }
    }
    chosen
}

/// Whether `drafter_directory` is one of the target's own drafter folders,
/// rather than a drafter found across the library.
#[must_use]
pub fn is_bundled_drafter_directory(
    target_model_directory: &Path,
    drafter_directory: &Path,
) -> bool {
    BUNDLED_DRAFTER_FOLDER_NAMES
        .iter()
        .any(|bundled_folder_name| {
            drafter_directory == target_model_directory.join(bundled_folder_name)
        })
}

/// Lower is preferred.
fn generation_preference(generation: DrafterGeneration) -> u8 {
    match generation {
        DrafterGeneration::DFlash2 => 0,
        DrafterGeneration::DFlash => 1,
    }
}

/// The configuration of the drafter in `candidate_directory` when there is
/// one this engine implements and it pairs with the target.
fn pairing_configuration(
    candidate_directory: &Path,
    pairing: &DrafterPairing,
) -> Option<DFlashDrafterConfiguration> {
    let config_document = drafter_config_document(candidate_directory)?;
    // A drafter of a kind this engine does not implement is not this
    // target's drafter: it is passed over, never a load failure.
    let drafter_configuration =
        match DFlashDrafterConfiguration::from_config_document(&config_document) {
            Ok(drafter_configuration) => drafter_configuration,
            Err(configuration_error) => {
                tracing::debug!(
                    drafter_directory = %candidate_directory.display(),
                    %configuration_error,
                    "passing over a drafter this engine does not implement"
                );
                return None;
            }
        };
    pairing
        .pairs_with(&drafter_configuration)
        .then_some(drafter_configuration)
}

/// Every model directory in the library the target sits in, whichever of
/// the two layouts that is.
fn sibling_model_directories(
    target_model_directory: &Path,
) -> Result<Vec<PathBuf>, DrafterDiscoveryError> {
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

fn child_directories(directory: &Path) -> Result<Vec<PathBuf>, DrafterDiscoveryError> {
    let entries = std::fs::read_dir(directory)
        .map_err(|source| DrafterDiscoveryError::LibraryUnlistable { source })?;
    let mut directories = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            directories.push(path);
        }
    }
    Ok(directories)
}

/// The directory's `config.json` when it describes a DFlash drafter of
/// either generation; None
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
