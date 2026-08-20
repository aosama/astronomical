use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::model_discovery_huggingface_cache::resolve_huggingface_cache_entry;

mod bounded_artifact_file;
mod classified_artifacts;
mod deepseek_v4;
mod flux2_klein;
mod flux2_klein_documents;
mod laguna;
mod model_family;
mod qwen3_5;

pub use classified_artifacts::{
    ClassifiedModelArtifact, discover_classified_model_artifacts, requestable_model_id,
};
pub use flux2_klein::{
    Flux2KleinDirectoryEvidence, Flux2KleinDirectoryVerificationError,
    verify_model_directory as verify_flux2_klein_model_directory,
};
pub use model_family::{ModelFamily, ModelFamilyClassificationError, classify_model_directory};

/// Capability contract for one discovered executable model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelCapabilities {
    Chat(ChatModelCapabilities),
    ImageGeneration(ImageGenerationCapabilities),
}

/// Token-streaming capabilities advertised by a chat model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatModelCapabilities {
    pub context_window: u32,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub supports_vision: bool,
    pub supports_reasoning: bool,
    pub supports_tool_calls: bool,
}

/// Image operations advertised without inventing autoregressive token limits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageGenerationCapabilities {
    pub supports_text_to_image: bool,
    pub supports_image_editing: bool,
    pub supports_multiple_reference_images: bool,
}

/// SPDX model-license identities accepted by executable discovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLicense {
    Apache20,
}

impl ModelLicense {
    /// Returns the canonical SPDX license identifier exposed to API adapters.
    #[must_use]
    pub const fn spdx_identifier(self) -> &'static str {
        match self {
            Self::Apache20 => "Apache-2.0",
        }
    }
}

/// One executable model discovered by recursive directory scanning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredModel {
    /// The model identity from the leaf directory name (e.g. "Ornith-1.0-35B-OptiQ-4bit").
    /// For HuggingFace cache entries, this is derived from the decoded `org/repo` path
    /// with the org prefix stripped (e.g. "Ornith-1.0-35B-6bit" from "mlx-community/Ornith-1.0-35B-6bit").
    pub model_id: String,
    /// Upstream provider identity retained as provenance, never as the local routing key.
    pub provider_model_id: Option<String>,
    /// The architecture family recognized from config.json or model_index.json.
    pub model_family: ModelFamily,
    /// Family-owned artifact revision used by public identity and serving state.
    pub revision: String,
    /// Absolute path to the validated family artifact directory.
    pub model_directory: PathBuf,
    /// Domain-specific operations callers may request from this model.
    pub capabilities: ModelCapabilities,
    /// Validated SPDX license metadata when the family contract declares one.
    pub license: Option<ModelLicense>,
    /// Unique safetensors shard payload bytes measured from the discovered files.
    pub model_size_bytes: u64,
}

/// Failure while scanning configured directories for executable models.
#[derive(Debug, Error)]
pub enum DiscoveredModelError {
    #[error("failed to read directory {directory_path:?}: {source}")]
    ReadDirectory {
        directory_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "duplicate discovered model ID {model_id:?} resolves to multiple directories: {model_directories:?}"
    )]
    DuplicateModelId {
        model_id: String,
        model_directories: Vec<PathBuf>,
    },
}

/// One configured scan directory with its resolved path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelDiscoveryDirectoryScan {
    /// The absolute path to scan.
    pub path: PathBuf,
    /// Models discovered under this directory.
    pub discovered_models: Vec<DiscoveredModel>,
}

/// Recursively scans configured directories for executable model families.
///
/// For each directory, walks subdirectories one level deep looking for
/// `config.json` or `model_index.json`. Family-specific shallow completeness rules decide whether a
/// classified artifact can be returned as executable.
///
/// Scan errors for individual directories are logged and skipped — the
/// function returns all successfully discovered models.
pub fn discover_models(
    model_directories: &[PathBuf],
) -> Result<Vec<ModelDiscoveryDirectoryScan>, DiscoveredModelError> {
    let mut directory_scans = Vec::with_capacity(model_directories.len());
    for directory_path in model_directories {
        let discovered_models = scan_directory_for_executable_models(directory_path)?;
        directory_scans.push(ModelDiscoveryDirectoryScan {
            path: directory_path.clone(),
            discovered_models,
        });
    }
    reject_duplicate_model_ids(&directory_scans)?;
    Ok(directory_scans)
}

/// Scans a single root directory recursively for executable models.
///
/// Walks up to 3 levels deep (supporting paths like
/// `hub/models--org--model/snapshots/abc123/` and `models/Org-Model-OptiQ-4bit/`).
/// Each family root is classified before family-owned executable validation.
fn scan_directory_for_executable_models(
    root_directory: &Path,
) -> Result<Vec<DiscoveredModel>, DiscoveredModelError> {
    let mut discovered_models = Vec::new();
    scan_directory_recursive(root_directory, 0, &mut discovered_models)?;
    Ok(discovered_models)
}

fn scan_directory_recursive(
    current_directory: &Path,
    depth: usize,
    discovered_models: &mut Vec<DiscoveredModel>,
) -> Result<(), DiscoveredModelError> {
    const MAX_SCAN_DEPTH: usize = 4;

    if depth > MAX_SCAN_DEPTH {
        return Ok(());
    }

    // If this directory is a HuggingFace cache entry (models--org--repo/),
    // resolve it to the active snapshot and discover from there.
    if current_directory
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("models--"))
        && let Some(huggingface_cache_entry) = resolve_huggingface_cache_entry(current_directory)
    {
        if let Some(discovered_model) = try_discover_model_with_id(
            &huggingface_cache_entry.snapshot_directory,
            &huggingface_cache_entry.model_id,
        ) {
            discovered_models.push(discovered_model);
        }
        return Ok(());
    }

    let has_pipeline_index = current_directory.join("model_index.json").is_file();
    if (current_directory.join("config.json").is_file() || has_pipeline_index)
        && let Some(discovered_model) = try_discover_model(current_directory)
    {
        discovered_models.push(discovered_model);
        // Don't recurse into a model directory — it won't contain nested models.
        return Ok(());
    }
    // A Diffusers pipeline root is terminal even when unsupported or incomplete;
    // nested component configs are not independently requestable artifacts.
    if has_pipeline_index {
        return Ok(());
    }

    // Recurse into subdirectories.
    let directory_entries = match fs::read_dir(current_directory) {
        Ok(entries) => entries,
        Err(source) => {
            // Silently skip directories we can't read (permissions, etc.).
            if depth == 0 {
                return Err(DiscoveredModelError::ReadDirectory {
                    directory_path: current_directory.to_path_buf(),
                    source,
                });
            }
            return Ok(());
        }
    };

    for directory_entry in directory_entries {
        let directory_entry = match directory_entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let entry_path = directory_entry.path();
        if entry_path.is_dir() {
            // Skip hidden directories (e.g. `.cache`, `.git`).
            if entry_path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.'))
            {
                continue;
            }
            let _ = scan_directory_recursive(&entry_path, depth + 1, discovered_models);
        }
    }

    Ok(())
}

/// Attempts to discover an executable model from a classified family root.
///
/// Uses the leaf directory name as `model_id`. For HuggingFace cache entries
/// where the snapshot hash is not a meaningful model ID, use
/// `try_discover_model_with_id` instead.
///
/// Returns `None` for classified families that are not executable yet.
fn try_discover_model(model_directory: &Path) -> Option<DiscoveredModel> {
    let model_id = model_directory
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    try_discover_model_with_id(model_directory, &model_id)
}

/// Attempts to discover an executable model from a directory with an explicit model ID.
///
/// This variant accepts a custom `model_id`, useful for HuggingFace cache entries
/// where the model_id is derived from the decoded `models--org--repo` directory name
/// rather than the snapshot hash.
fn try_discover_model_with_id(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    // The typed classifier rejects ambiguous duplicate family markers before
    // the looser metadata document can participate in executable discovery.
    let model_family = model_family::classify_model_directory(model_directory)
        .ok()
        .flatten()?;
    match model_family {
        ModelFamily::Qwen3_5 => {
            let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
            let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
            let family_metadata = qwen3_5::discover_model_metadata(model_directory, &config_value)?;
            Some(DiscoveredModel {
                model_id: model_id.to_owned(),
                provider_model_id: None,
                model_family,
                revision: derive_revision_from_config_bytes(&config_bytes),
                model_directory: model_directory.to_path_buf(),
                capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
                    context_window: family_metadata.context_window,
                    max_input_tokens: family_metadata.max_input_tokens,
                    max_output_tokens: family_metadata.max_output_tokens,
                    supports_vision: family_metadata.has_vision,
                    supports_reasoning: family_metadata.supports_reasoning,
                    supports_tool_calls: family_metadata.supports_tool_calls,
                }),
                license: None,
                model_size_bytes: family_metadata.model_size_bytes,
            })
        }
        ModelFamily::Laguna => {
            let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
            let laguna_metadata = laguna::discover_model_metadata(model_directory, &config_bytes)?;
            Some(DiscoveredModel {
                model_id: model_id.to_owned(),
                provider_model_id: None,
                model_family,
                revision: laguna_metadata.revision,
                model_directory: model_directory.to_path_buf(),
                capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
                    context_window: laguna_metadata.context_window,
                    max_input_tokens: laguna_metadata.max_input_tokens,
                    max_output_tokens: laguna_metadata.max_output_tokens,
                    supports_vision: laguna_metadata.has_vision,
                    supports_reasoning: laguna_metadata.supports_reasoning,
                    supports_tool_calls: laguna_metadata.supports_tool_calls,
                }),
                license: None,
                model_size_bytes: laguna_metadata.model_size_bytes,
            })
        }
        ModelFamily::Flux2Klein => {
            let verified_evidence = flux2_klein::verify_model_directory(model_directory).ok()?;
            Some(DiscoveredModel {
                model_id: verified_evidence.canonical_model_id,
                provider_model_id: Some(verified_evidence.provider_model_id),
                model_family,
                revision: verified_evidence.revision,
                model_directory: model_directory.to_path_buf(),
                capabilities: ModelCapabilities::ImageGeneration(verified_evidence.capabilities),
                license: Some(verified_evidence.license),
                model_size_bytes: verified_evidence.model_size_bytes,
            })
        }
        // Classification is intentionally broader than executable discovery.
        ModelFamily::DeepSeekV4 => None,
    }
}

/// Rejects ambiguous public identities before callers can build lookup maps.
fn reject_duplicate_model_ids(
    directory_scans: &[ModelDiscoveryDirectoryScan],
) -> Result<(), DiscoveredModelError> {
    let mut model_id_to_directories: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for discovered_model in directory_scans
        .iter()
        .flat_map(|directory_scan| &directory_scan.discovered_models)
    {
        model_id_to_directories
            .entry(discovered_model.model_id.clone())
            .or_default()
            .push(discovered_model.model_directory.clone());
    }
    for (model_id, mut model_directories) in model_id_to_directories {
        if model_directories.len() < 2 {
            continue;
        }
        model_directories.sort();
        return Err(DiscoveredModelError::DuplicateModelId {
            model_id,
            model_directories,
        });
    }
    Ok(())
}

/// Derives a 12-character hex revision string from the SHA-256 hash of config.json bytes.
fn derive_revision_from_config_bytes(config_bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut sha256_hasher = Sha256::new();
    sha256_hasher.update(config_bytes);
    let config_hash = sha256_hasher.finalize();
    format!(
        "{:012x}",
        u64::from_be_bytes(config_hash[..8].try_into().unwrap_or([0u8; 8]))
    )
}
