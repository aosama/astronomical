//! Shallow discovery for standalone Qwen MTP artifacts used only as auxiliaries.
//!
//! This scanner intentionally remains separate from executable model-family
//! classification so a drafter can participate in explicit pairing resolution
//! without ever becoming a public chat model.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::classified_artifacts::immutable_model_revision;
use super::{DiscoveredModelError, requestable_model_id};
use crate::model_discovery_huggingface_cache::resolve_huggingface_cache_entry;

const MAXIMUM_AUXILIARY_SCAN_DEPTH: usize = 4;
const MAXIMUM_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_INDEX_BYTES: u64 = 16 * 1024 * 1024;

/// One shallowly discovered standalone Qwen MTP drafter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredQwen3_5MtpDrafter {
    pub model_id: String,
    /// Immutable upstream revision when available, otherwise a config-derived revision.
    pub revision: String,
    /// Publisher or cache revision evidence retained independently from local fallback identity.
    pub upstream_revision: Option<String>,
    pub model_directory: PathBuf,
}

/// Finds standalone Qwen MTP artifacts beneath configured recursive model roots.
///
/// Discovery proves only packaging completeness. Deep tensor, tokenizer, and
/// compatibility validation remains worker-owned so one malformed auxiliary
/// cannot remove its otherwise executable target from public discovery.
pub fn discover_qwen3_5_mtp_drafters(
    configured_model_directories: &[PathBuf],
) -> Result<Vec<DiscoveredQwen3_5MtpDrafter>, DiscoveredModelError> {
    let mut discovered_drafters = Vec::new();
    for configured_model_directory in configured_model_directories {
        scan_directory(
            configured_model_directory,
            0,
            true,
            &mut discovered_drafters,
        )?;
    }
    discovered_drafters.sort_by(|first_drafter, second_drafter| {
        first_drafter
            .model_id
            .cmp(&second_drafter.model_id)
            .then_with(|| first_drafter.revision.cmp(&second_drafter.revision))
    });
    reject_duplicate_model_ids(&discovered_drafters)?;
    Ok(discovered_drafters)
}

fn scan_directory(
    current_directory: &Path,
    depth: usize,
    is_configured_root: bool,
    discovered_drafters: &mut Vec<DiscoveredQwen3_5MtpDrafter>,
) -> Result<(), DiscoveredModelError> {
    if depth > MAXIMUM_AUXILIARY_SCAN_DEPTH {
        return Ok(());
    }

    if current_directory
        .file_name()
        .and_then(|directory_name| directory_name.to_str())
        .is_some_and(|directory_name| directory_name.starts_with("models--"))
        && let Some(cache_entry) = resolve_huggingface_cache_entry(current_directory)
    {
        if let Some(discovered_drafter) =
            discover_drafter(&cache_entry.snapshot_directory, Some(cache_entry.model_id))
        {
            discovered_drafters.push(discovered_drafter);
        }
        return Ok(());
    }

    if current_directory.join("config.json").is_file() {
        if let Some(discovered_drafter) = discover_drafter(current_directory, None) {
            discovered_drafters.push(discovered_drafter);
        }
        return Ok(());
    }

    let directory_entries = match fs::read_dir(current_directory) {
        Ok(directory_entries) => directory_entries,
        Err(source) if is_configured_root => {
            return Err(DiscoveredModelError::ReadDirectory {
                directory_path: current_directory.to_path_buf(),
                source,
            });
        }
        Err(_) => return Ok(()),
    };
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        if entry_path.is_dir()
            && !entry_path
                .file_name()
                .and_then(|directory_name| directory_name.to_str())
                .is_some_and(|directory_name| directory_name.starts_with('.'))
        {
            scan_directory(&entry_path, depth + 1, false, discovered_drafters)?;
        }
    }
    Ok(())
}

fn discover_drafter(
    model_directory: &Path,
    explicit_model_id: Option<String>,
) -> Option<DiscoveredQwen3_5MtpDrafter> {
    let config_bytes =
        read_bounded_file(&model_directory.join("config.json"), MAXIMUM_CONFIG_BYTES)?;
    let config_document: AuxiliaryConfigDocument = serde_json::from_slice(&config_bytes).ok()?;
    if config_document.model_type.as_deref() != Some("qwen3_5_mtp")
        || !model_directory.join("tokenizer.json").is_file()
        || !has_complete_shallow_payload(model_directory)
    {
        return None;
    }

    let model_id = explicit_model_id.or_else(|| requestable_model_id(model_directory))?;
    let upstream_revision = immutable_model_revision(model_directory);
    let revision = upstream_revision
        .clone()
        .unwrap_or_else(|| derive_config_revision(&config_bytes));
    Some(DiscoveredQwen3_5MtpDrafter {
        model_id,
        revision,
        upstream_revision,
        model_directory: model_directory.to_path_buf(),
    })
}

fn has_complete_shallow_payload(model_directory: &Path) -> bool {
    if model_directory.join("model.safetensors").is_file() {
        return true;
    }
    let index_bytes = match read_bounded_file(
        &model_directory.join("model.safetensors.index.json"),
        MAXIMUM_INDEX_BYTES,
    ) {
        Some(index_bytes) => index_bytes,
        None => return false,
    };
    let index_document: AuxiliaryIndexDocument = match serde_json::from_slice(&index_bytes) {
        Ok(index_document) => index_document,
        Err(_) => return false,
    };
    if index_document.weight_map.is_empty() {
        return false;
    }
    let referenced_file_names = index_document.weight_map.values().collect::<BTreeSet<_>>();
    referenced_file_names
        .into_iter()
        .all(|referenced_file_name| {
            is_safe_relative_file_name(referenced_file_name)
                && model_directory.join(referenced_file_name).is_file()
        })
}

fn read_bounded_file(file_path: &Path, maximum_bytes: u64) -> Option<Vec<u8>> {
    let file = fs::File::open(file_path).ok()?;
    if file.metadata().ok()?.len() > maximum_bytes {
        return None;
    }
    let mut file_bytes = Vec::new();
    file.take(maximum_bytes + 1)
        .read_to_end(&mut file_bytes)
        .ok()?;
    (file_bytes.len() as u64 <= maximum_bytes).then_some(file_bytes)
}

fn is_safe_relative_file_name(file_name: &str) -> bool {
    let file_path = Path::new(file_name);
    !file_path.as_os_str().is_empty()
        && !file_path.is_absolute()
        && file_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn derive_config_revision(config_bytes: &[u8]) -> String {
    let config_digest = Sha256::digest(config_bytes);
    config_digest[..6]
        .iter()
        .map(|digest_byte| format!("{digest_byte:02x}"))
        .collect()
}

fn reject_duplicate_model_ids(
    discovered_drafters: &[DiscoveredQwen3_5MtpDrafter],
) -> Result<(), DiscoveredModelError> {
    let mut discovered_model_ids = BTreeSet::new();
    for discovered_drafter in discovered_drafters {
        if !discovered_model_ids.insert(&discovered_drafter.model_id) {
            return Err(DiscoveredModelError::DuplicateAuxiliaryMtpModelId {
                model_id: discovered_drafter.model_id.clone(),
            });
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct AuxiliaryConfigDocument {
    #[serde(default)]
    model_type: Option<String>,
}

#[derive(Deserialize)]
struct AuxiliaryIndexDocument {
    #[serde(default)]
    weight_map: BTreeMap<String, String>,
}
