//! Persists the user's default model id for one-shot CLI verbs.

use std::path::Path;

use crate::config_document::UserConfigFile;
use crate::config_error::AstronomicalConfigError;
use crate::config_file::{
    parse_and_validate_v1, read_existing_config_file_bytes, validate_user_config_file,
    write_adjacent_schema, write_config_file_bytes_atomically,
};
use crate::duplicate_key_json;
use crate::legacy_config_migration::{
    prepare_legacy_config_migration, preserve_legacy_config_backup,
};

/// Built-in chat model used when the client sends no model and the user has
/// configured no default. Shared by the daemon's effective-default
/// resolution and the CLI's request resolution so the two can never drift.
pub const BUILTIN_DEFAULT_MODEL_ID: &str = "Qwen3.5-2B-4bit";

/// Built-in image model for `astronomical image` when the user names none.
/// The persisted default model is a chat model and cannot draw, so images
/// have their own default. FLUX.2 Klein runs 4 diffusion steps where
/// Qwen-Image runs 40: on one Apple silicon Mac it drew the same 1024x576
/// prompt in about 9 seconds against Qwen's 89, which is the difference
/// between a first image that feels instant and one that feels hung. The
/// price is a larger one-time download (about 14.9 GB against 9.8 GB, in the
/// 2^30-byte gigabytes the CLI prints). Qwen-Image stays available with
/// `--model Qwen-Image-2.1-MLX-4bit`.
pub const BUILTIN_DEFAULT_IMAGE_MODEL_ID: &str = "FLUX.2-klein-4B";

/// Rejects ids that cannot resolve to a catalog entry or discovered model.
fn validate_default_model_id(model_id: &str) -> Result<(), AstronomicalConfigError> {
    if model_id.is_empty() || model_id.trim() != model_id {
        return Err(AstronomicalConfigError::InvalidDefaultModel {
            description: "default model id must be a non-empty model id without surrounding whitespace",
        });
    }
    Ok(())
}

/// Exact before/after bytes for one atomic default-model configuration mutation.
pub struct DefaultModelConfigUpdate {
    pub prior_config_bytes: Option<Vec<u8>>,
    pub candidate_config_bytes: Vec<u8>,
}

/// Builds a validated byte transaction without mutating the source of truth.
pub fn prepare_default_model_update(
    state_directory: impl AsRef<Path>,
    default_model: Option<&str>,
) -> Result<DefaultModelConfigUpdate, AstronomicalConfigError> {
    if let Some(default_model) = default_model {
        validate_default_model_id(default_model)?;
    }

    let config_file_path = state_directory.as_ref().join("config.json");
    let prior_config_bytes = read_existing_config_file_bytes(&config_file_path)?;
    let mut candidate_user_config_file = match prior_config_bytes.as_deref() {
        Some(config_file_bytes) => {
            let config_json = duplicate_key_json::parse_json_rejecting_duplicates(
                &config_file_path,
                config_file_bytes,
            )?;
            if config_json.get("schema_version").is_none() {
                prepare_legacy_config_migration(&config_file_path, config_json)?
            } else {
                parse_and_validate_v1(&config_file_path, config_json)?
            }
        }
        None => UserConfigFile::minimal(),
    };
    candidate_user_config_file.runtime.default_model = default_model.map(str::to_owned);
    validate_user_config_file(&candidate_user_config_file)?;

    let candidate_config_bytes =
        serde_json::to_vec_pretty(&candidate_user_config_file).map_err(|source| {
            AstronomicalConfigError::SerializeConfigFile {
                config_file_path: config_file_path.clone(),
                source,
            }
        })?;
    Ok(DefaultModelConfigUpdate {
        prior_config_bytes,
        candidate_config_bytes,
    })
}

/// Commits a prepared update only while the document it was based on still owns the file.
pub fn commit_default_model_update(
    state_directory: impl AsRef<Path>,
    config_update: &DefaultModelConfigUpdate,
) -> Result<(), AstronomicalConfigError> {
    let config_file_path = state_directory.as_ref().join("config.json");
    if read_existing_config_file_bytes(&config_file_path)? != config_update.prior_config_bytes {
        return Err(AstronomicalConfigError::ConfigChangedDuringUpdate);
    }
    if let Some(prior_config_bytes) = config_update.prior_config_bytes.as_deref() {
        let prior_config_json = duplicate_key_json::parse_json_rejecting_duplicates(
            &config_file_path,
            prior_config_bytes,
        )?;
        if prior_config_json.get("schema_version").is_none() {
            preserve_legacy_config_backup(&config_file_path, prior_config_bytes)?;
        }
    }
    // The schema precedes the document so every committed config remains locally inspectable.
    write_adjacent_schema(&config_file_path)?;
    write_config_file_bytes_atomically(&config_file_path, &config_update.candidate_config_bytes)
}

/// Persists the default model id (or clears it) and returns its exact byte transaction.
pub fn write_default_model(
    state_directory: impl AsRef<Path>,
    default_model: Option<&str>,
) -> Result<DefaultModelConfigUpdate, AstronomicalConfigError> {
    let state_directory = state_directory.as_ref();
    let config_update = prepare_default_model_update(state_directory, default_model)?;
    commit_default_model_update(state_directory, &config_update)?;
    Ok(config_update)
}
