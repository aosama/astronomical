//! The engine-owned DFlash draft runtime.
//!
//! One resident engine may host one draft beside its target. Loading is
//! deliberately permissive about absence — a library with no draft simply
//! runs the classic decode path — and strict about pairing: a draft that
//! does not match the target's layer count, hidden width, or shared
//! vocabulary is rejected, because those mismatches silently corrupt the
//! conditioning contract rather than failing loudly at the first round.

use std::path::{Path, PathBuf};

use astronomical_runtime_integration::MlxRuntime;

use crate::qwen3_5_resident::dflash_draft::draft_configuration::{
    DFlashDraftConfiguration, DFlashDraftConfigurationError,
};
use crate::qwen3_5_resident::dflash_draft::draft_weights_loading::DFlashDraftLoadedWeights;

use super::speculation_session_admission::draft_pairs_with_target;

/// Headroom the draft load insists on beyond its own weight bytes, so the
/// first verify workspace never starts life squeezed against the ceiling.
const DRAFT_LOAD_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

/// The loaded draft and the block length its sessions draft with.
pub struct Qwen3_5SpeculativeDraftRuntime {
    pub(crate) draft: DFlashDraftLoadedWeights,
    pub(crate) draft_configuration: DFlashDraftConfiguration,
    pub(crate) speculation_block_length: usize,
}

impl Qwen3_5SpeculativeDraftRuntime {
    /// Resolves and loads the draft for one target, or `Ok(None)` when no
    /// draft artifact is installed beside it in the model library.
    ///
    /// The draft is discovered by leaf directory name across the library's
    /// organization directories: target and draft ship under different
    /// organizations, so only the library root is a safe search base.
    pub fn load_for_target(
        runtime: &MlxRuntime,
        active_memory_limit_bytes: usize,
        target_model_directory: &Path,
        target_layer_count: usize,
        target_hidden_size: usize,
        target_vocabulary_size: u32,
        speculation_block_length: usize,
    ) -> Result<Option<Self>, Qwen3_5SpeculativeDraftLoadingError> {
        let Some(draft_model_directory) =
            resolve_draft_directory_for_target(target_model_directory)?
        else {
            return Ok(None);
        };
        let draft_config_bytes = std::fs::read(draft_model_directory.join("config.json"))
            .map_err(|source| Qwen3_5SpeculativeDraftLoadingError::ConfigUnreadable { source })?;
        let draft_config_document: serde_json::Value = serde_json::from_slice(&draft_config_bytes)
            .map_err(Qwen3_5SpeculativeDraftLoadingError::ConfigMalformed)?;
        let draft_configuration =
            DFlashDraftConfiguration::from_config_value(&draft_config_document)
                .map_err(Qwen3_5SpeculativeDraftLoadingError::Configuration)?;
        let pairs_with_target = draft_pairs_with_target(
            draft_configuration.target_layer_count(),
            draft_configuration.target_layer_ids(),
            draft_configuration.vocabulary_size(),
            draft_configuration.target_conditioning_input_width(target_hidden_size as i32),
            target_layer_count,
            target_hidden_size,
            target_vocabulary_size,
        );
        if !pairs_with_target {
            return Err(Qwen3_5SpeculativeDraftLoadingError::PairingMismatch {
                draft_directory: draft_model_directory,
            });
        }
        ensure_memory_headroom_for_draft(
            runtime,
            active_memory_limit_bytes,
            draft_model_directory.as_path(),
        )?;
        let draft =
            DFlashDraftLoadedWeights::load(runtime, &draft_model_directory, &draft_configuration)
                .map_err(Qwen3_5SpeculativeDraftLoadingError::Weights)?;
        Ok(Some(Self {
            draft,
            draft_configuration,
            speculation_block_length,
        }))
    }
}

/// Finds the installed draft leaf directory across the library's
/// organization directories, or `None` when the library carries no draft.
fn resolve_draft_directory_for_target(
    target_model_directory: &Path,
) -> Result<Option<PathBuf>, Qwen3_5SpeculativeDraftLoadingError> {
    let Some(library_root) = target_model_directory
        .parent()
        .and_then(|organization_directory| organization_directory.parent())
    else {
        return Ok(None);
    };
    let organization_directories = std::fs::read_dir(library_root)
        .map_err(|source| Qwen3_5SpeculativeDraftLoadingError::LibraryUnlistable { source })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir());
    for organization_entry in organization_directories {
        for draft_leaf_id in installed_draft_leaf_ids() {
            let candidate_directory = organization_entry.path().join(draft_leaf_id);
            if candidate_directory.join("model.safetensors").is_file() {
                return Ok(Some(candidate_directory));
            }
        }
    }
    Ok(None)
}

/// Draft leaf directory names this engine recognizes. The v1 pairing rule
/// is the companion-draft naming convention: the draft ships beside its
/// target family under the family's exact leaf name, so this list changes
/// only when the library retargets checkpoints — the same event that
/// updates the e2e registry roles. An explicit draft-name configuration
/// key would supersede it if libraries ever carry more than one family.
fn installed_draft_leaf_ids() -> [String; 1] {
    [String::from("Ornith-1.5-35B-A3B-DFlash")]
}

/// Refuses the draft when the active limit cannot hold its weights plus
/// headroom beside everything already resident.
fn ensure_memory_headroom_for_draft(
    runtime: &MlxRuntime,
    active_memory_limit_bytes: usize,
    draft_model_directory: &Path,
) -> Result<(), Qwen3_5SpeculativeDraftLoadingError> {
    let draft_weights_bytes = std::fs::metadata(draft_model_directory.join("model.safetensors"))
        .map_err(|source| Qwen3_5SpeculativeDraftLoadingError::WeightsUnreadable { source })?
        .len();
    let memory_snapshot = runtime
        .memory_snapshot()
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Runtime)?;
    let active_limit_bytes = u64::try_from(active_memory_limit_bytes).unwrap_or(u64::MAX);
    let active_memory_bytes = u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(0);
    let free_bytes = active_limit_bytes.saturating_sub(active_memory_bytes);
    if free_bytes < draft_weights_bytes + DRAFT_LOAD_HEADROOM_BYTES {
        return Err(Qwen3_5SpeculativeDraftLoadingError::InsufficientHeadroom {
            free_bytes,
            required_bytes: draft_weights_bytes + DRAFT_LOAD_HEADROOM_BYTES,
        });
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen3_5SpeculativeDraftLoadingError {
    #[error("the model library directory cannot be listed to find a DFlash draft: {source}")]
    LibraryUnlistable { source: std::io::Error },
    #[error("the installed DFlash draft's config.json is unreadable: {source}")]
    ConfigUnreadable { source: std::io::Error },
    #[error("the installed DFlash draft's config.json is malformed: {0}")]
    ConfigMalformed(#[source] serde_json::Error),
    #[error("the installed DFlash draft's configuration is invalid: {0}")]
    Configuration(#[source] DFlashDraftConfigurationError),
    #[error("the DFlash draft at {draft_directory:?} does not pair with the loaded target model")]
    PairingMismatch { draft_directory: PathBuf },
    #[error("the graphics-processor runtime rejected the DFlash draft load: {0}")]
    Runtime(#[source] astronomical_runtime_integration::MlxRuntimeError),
    #[error("the DFlash draft's model.safetensors is unreadable: {source}")]
    WeightsUnreadable { source: std::io::Error },
    #[error("the DFlash draft weights did not load or match their declared shapes: {0}")]
    Weights(
        #[source]
        crate::qwen3_5_resident::dflash_draft::draft_weights_loading::DFlashDraftWeightsLoadingError,
    ),
    #[error("the active memory limit cannot host the DFlash draft: {free_bytes} bytes free, {required_bytes} bytes required")]
    InsufficientHeadroom {
        free_bytes: u64,
        required_bytes: u64,
    },
}
