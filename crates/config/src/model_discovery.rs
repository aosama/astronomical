use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::model_discovery_huggingface_cache::resolve_huggingface_cache_entry;

/// One supported Qwen3.5-family model discovered by recursive directory scanning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredModel {
    /// The model identity from the leaf directory name (e.g. "Ornith-1.0-35B-OptiQ-4bit").
    /// For HuggingFace cache entries, this is derived from the decoded `org/repo` path
    /// with the org prefix stripped (e.g. "Ornith-1.0-35B-6bit" from "mlx-community/Ornith-1.0-35B-6bit").
    pub model_id: String,
    /// SHA-256 hash of config.json bytes (12 hex chars).
    pub revision: String,
    /// Absolute path to the model directory containing config.json, tokenizer.json, etc.
    pub model_directory: PathBuf,
    /// Total prompt + generation position capacity from config.json.
    pub context_window: u32,
    /// Maximum prompt tokens a client may send (context_window - max_output_tokens).
    pub max_input_tokens: u32,
    /// Per-request output-token ceiling from config or default.
    pub max_output_tokens: u32,
    /// Whether the checkpoint index declares physically present visual weights.
    pub has_vision: bool,
    /// Unique safetensors shard payload bytes measured from the discovered files.
    pub model_size_bytes: u64,
}

/// Failure while scanning directories for supported Qwen3.5-family models.
#[derive(Debug, Error)]
pub enum DiscoveredModelError {
    #[error("failed to read directory {directory_path:?}: {source}")]
    ReadDirectory {
        directory_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read config.json in {model_directory:?}: {source}")]
    ReadConfig {
        model_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config.json in {model_directory:?}: {source}")]
    ParseConfig {
        model_directory: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "config.json in {model_directory:?} is not a supported Qwen3.5-family model (expected model_type qwen3_5, qwen3_5_moe, or qwen3_5_moe_vision, found {found_type:?})"
    )]
    IncompatibleModelType {
        model_directory: PathBuf,
        found_type: Option<String>,
    },
    #[error("missing required file {file_name} in model directory {model_directory:?}")]
    MissingRequiredFile {
        model_directory: PathBuf,
        file_name: String,
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

/// Recursively scans configured directories for supported Qwen3.5-family models.
///
/// For each directory, walks subdirectories one level deep looking for
/// `config.json`. When a subdirectory contains `config.json` with a
/// compatible `model_type`, it also checks for `model.safetensors.index.json`
/// and `tokenizer.json`, then records the model as discovered.
///
/// Scan errors for individual directories are logged and skipped — the
/// function returns all successfully discovered models.
pub fn discover_qwen3_5_models(
    model_directories: &[PathBuf],
    max_output_tokens: u32,
) -> Result<Vec<ModelDiscoveryDirectoryScan>, DiscoveredModelError> {
    let mut directory_scans = Vec::with_capacity(model_directories.len());
    for directory_path in model_directories {
        let discovered_models =
            scan_directory_for_qwen3_5_models(directory_path, max_output_tokens)?;
        directory_scans.push(ModelDiscoveryDirectoryScan {
            path: directory_path.clone(),
            discovered_models,
        });
    }
    Ok(directory_scans)
}

/// Scans a single root directory recursively for supported Qwen3.5 models.
///
/// Walks up to 3 levels deep (supporting paths like
/// `hub/models--org--model/snapshots/abc123/` and `models/Org-Model-OptiQ-4bit/`).
/// Each subdirectory containing `config.json` is checked for Qwen3.5 compatibility.
fn scan_directory_for_qwen3_5_models(
    root_directory: &Path,
    max_output_tokens: u32,
) -> Result<Vec<DiscoveredModel>, DiscoveredModelError> {
    let mut discovered_models = Vec::new();
    scan_directory_recursive(root_directory, 0, max_output_tokens, &mut discovered_models)?;
    Ok(discovered_models)
}

fn scan_directory_recursive(
    current_directory: &Path,
    depth: usize,
    max_output_tokens: u32,
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
            max_output_tokens,
        ) {
            discovered_models.push(discovered_model);
        }
        return Ok(());
    }

    // If this directory looks like a model directory, try to discover it.
    if current_directory.join("config.json").is_file()
        && let Some(discovered_model) = try_discover_model(current_directory, max_output_tokens)
    {
        discovered_models.push(discovered_model);
        // Don't recurse into a model directory — it won't contain nested models.
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
            let _ = scan_directory_recursive(
                &entry_path,
                depth + 1,
                max_output_tokens,
                discovered_models,
            );
        }
    }

    Ok(())
}

/// Attempts to discover a supported Qwen3.5 model from a directory containing `config.json`.
///
/// Uses the leaf directory name as `model_id`. For HuggingFace cache entries
/// where the snapshot hash is not a meaningful model ID, use
/// `try_discover_model_with_id` instead.
///
/// Returns `Some(DiscoveredModel)` if the directory contains a compatible model,
/// `None` if config.json doesn't identify a supported Qwen3.5-family model.
fn try_discover_model(model_directory: &Path, max_output_tokens: u32) -> Option<DiscoveredModel> {
    let model_id = model_directory
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    try_discover_model_with_id(model_directory, &model_id, max_output_tokens)
}

/// Attempts to discover a supported Qwen3.5 model from a directory with an explicit model_id.
///
/// This variant accepts a custom `model_id`, useful for HuggingFace cache entries
/// where the model_id is derived from the decoded `models--org--repo` directory name
/// rather than the snapshot hash.
fn try_discover_model_with_id(
    model_directory: &Path,
    model_id: &str,
    max_output_tokens: u32,
) -> Option<DiscoveredModel> {
    // Check for required files first (fast path).
    if !model_directory
        .join("model.safetensors.index.json")
        .is_file()
    {
        return None;
    }
    if !model_directory.join("tokenizer.json").is_file() {
        return None;
    }

    // Read and parse config.json.
    let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;

    // Check model_type compatibility.
    let model_type = config_value.get("model_type").and_then(|v| v.as_str());
    match model_type {
        Some("qwen3_5") | Some("qwen3_5_moe") | Some("qwen3_5_moe_vision") => {}
        _ => return None,
    }

    // Verify that all shard files referenced in the safetensors index actually
    // exist on disk. Incomplete model downloads (missing shards) pass the
    // config/index/tokenizer checks but crash the worker on hot-swap. The
    // Vision tensors can be embedded in language shards or stored in a required
    // sidecar. The optional MTP sidecar may be absent.
    let index_path = model_directory.join("model.safetensors.index.json");
    let has_vision = if let Ok(index_bytes) = fs::read(&index_path)
        && let Ok(index_document) = serde_json::from_slice::<serde_json::Value>(&index_bytes)
        && let Some(weight_map) = index_document
            .get("weight_map")
            .and_then(|weight_map| weight_map.as_object())
    {
        let mut shard_file_names = HashSet::new();
        let mut required_shard_file_names = HashSet::new();
        for (tensor_name, tensor_shard_file_name) in weight_map {
            if let Some(tensor_shard_file_name) = tensor_shard_file_name.as_str() {
                shard_file_names.insert(tensor_shard_file_name.to_owned());
                if !contains_mtp_component(tensor_name) {
                    required_shard_file_names.insert(tensor_shard_file_name.to_owned());
                }
            }
        }
        for shard_file_name in &shard_file_names {
            let shard_path = model_directory.join(shard_file_name);
            if !shard_path.is_file() && required_shard_file_names.contains(shard_file_name) {
                return None;
            }
        }
        weight_map
            .keys()
            .any(|tensor_name| tensor_name.starts_with("vision_tower."))
    } else {
        false
    };
    let model_size_bytes = measure_model_safetensors_bytes(model_directory)?;

    // Derive revision from SHA-256 of config.json.
    let revision = derive_revision_from_config_bytes(&config_bytes);

    // Extract context window from text_config.
    let max_position_embeddings = config_value
        .get("text_config")
        .and_then(|tc| tc.get("max_position_embeddings"))
        .or_else(|| config_value.get("max_position_embeddings"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    // Derive context_window from max_position_embeddings.
    let context_window = max_position_embeddings;
    let max_input_tokens = context_window.saturating_sub(max_output_tokens);

    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        revision,
        model_directory: model_directory.to_path_buf(),
        context_window,
        max_input_tokens,
        max_output_tokens,
        has_vision,
        model_size_bytes,
    })
}

fn contains_mtp_component(tensor_name: &str) -> bool {
    tensor_name
        .split('.')
        .any(|tensor_name_component| tensor_name_component == "mtp")
}

fn measure_model_safetensors_bytes(model_directory: &Path) -> Option<u64> {
    let mut pending_directories = vec![model_directory.to_path_buf()];
    let mut measured_safetensors_paths = HashSet::new();
    let mut model_size_bytes = 0_u64;
    while let Some(pending_directory) = pending_directories.pop() {
        for directory_entry in fs::read_dir(pending_directory).ok()? {
            let directory_entry = directory_entry.ok()?;
            let entry_path = directory_entry.path();
            let entry_file_type = directory_entry.file_type().ok()?;
            if entry_file_type.is_dir() {
                pending_directories.push(entry_path);
                continue;
            }
            if entry_path
                .extension()
                .and_then(|extension| extension.to_str())
                != Some("safetensors")
                || !entry_path.is_file()
            {
                continue;
            }
            let canonical_safetensors_path = fs::canonicalize(&entry_path).ok()?;
            if measured_safetensors_paths.insert(canonical_safetensors_path) {
                model_size_bytes =
                    model_size_bytes.checked_add(fs::metadata(entry_path).ok()?.len())?;
            }
        }
    }
    Some(model_size_bytes)
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
