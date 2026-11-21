//! The engine-owned DFlash2 drafter runtime.
//!
//! One resident engine may host one drafter beside its target. Loading is
//! permissive about absence (a library with no drafter runs the classic
//! decode path) and strict about pairing: a drafter is used only when its
//! configuration names this target's layer count, hidden width and
//! vocabulary, because it borrows the target's embedding, head and tapped
//! hidden rows. Installing the runtime also converts the target's dense
//! projections into register order and declares the tree kernels, since a
//! drafting session verifies a whole draft tree every round; on a GPU
//! without the matrix units those kernels need, no runtime is installed.

use std::path::Path;

use runtime_integration::MlxSafetensors;

use crate::PerformanceAttribution;
use crate::qwen3_5_resident::Qwen3_5ResidentModel;
use crate::speculative_decoding::dflash_drafter::{
    DFlashDrafterConfiguration, DFlashDrafterWeights, DFlashDrafterWeightsError, DraftTreeOptions,
    bind_drafter_weights,
};

use super::draft_tree_size::draft_tree_options_for_target;
use super::drafter_discovery::{DrafterPairing, paired_drafter_directory};

/// Headroom the drafter load insists on beyond its own weight bytes, so the
/// first verify workspace never starts life squeezed against the ceiling.
const DRAFTER_LOAD_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;
const DRAFTER_WEIGHTS_FILE_NAME: &str = "model.safetensors";

/// The loaded drafter a session drafts with.
pub struct Qwen3_5SpeculativeDraftRuntime {
    pub drafter_configuration: DFlashDrafterConfiguration,
    pub drafter_weights: DFlashDrafterWeights,
    /// The tree size and search settings, chosen for this target's kind.
    pub tree_options: DraftTreeOptions,
    /// Keeps the drafter's lazily mapped codebooks alive.
    _drafter_checkpoint: MlxSafetensors,
}

impl Qwen3_5SpeculativeDraftRuntime {
    /// Finds and loads the drafter for `model`, or `Ok(None)` when the
    /// library holds none that pairs with it, or this GPU cannot run the
    /// register-ordered kernels a drafting session needs.
    pub fn load_for_target(
        model: &mut Qwen3_5ResidentModel,
        active_memory_limit_bytes: usize,
        target_model_directory: &Path,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<Self>, Qwen3_5SpeculativeDraftLoadingError> {
        let pairing = DrafterPairing {
            target_layer_count: model.config().layer_count() as i32,
            target_hidden_size: model.config().hidden_size() as i32,
            target_vocabulary_size: model.config().vocabulary_size() as i32,
        };
        let Some((drafter_directory, drafter_configuration)) =
            paired_drafter_directory(target_model_directory, &pairing)?
        else {
            return Ok(None);
        };
        ensure_memory_headroom_for_drafter(model, active_memory_limit_bytes, &drafter_directory)?;
        let Some(converted_projection_count) =
            prepare_target_for_draft_trees(model, performance_attribution)?
        else {
            return Ok(None);
        };
        let (drafter_weights, drafter_checkpoint) =
            load_drafter_weights(model, &drafter_directory, &drafter_configuration)?;
        let tree_options = draft_tree_options_for_target(model.config().expert_count());
        tracing::info!(
            drafter_directory = %drafter_directory.display(),
            converted_projection_count,
            tree_node_budget = tree_options.node_budget,
            "resident engine decode path: DFlash2 draft trees"
        );
        Ok(Some(Self {
            drafter_configuration,
            drafter_weights,
            tree_options,
            _drafter_checkpoint: drafter_checkpoint,
        }))
    }
}

/// Converts the target's dense projections into register order and
/// declares the tree kernels, the row-exact tree attention among them where
/// the target's attention shape fits it: the count converted, or `None` on a
/// GPU that cannot run the register-ordered kernels.
///
/// Conversion only speeds up verification; the tree kernels are what make
/// it possible. So a target with nothing to convert still drafts: an 8-bit
/// or mixed build, or a mixture-of-experts model whose experts stay in MLX's
/// layout for the gathered expert kernels and whose attention may be 8-bit.
fn prepare_target_for_draft_trees(
    model: &mut Qwen3_5ResidentModel,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Option<usize>, Qwen3_5SpeculativeDraftLoadingError> {
    let Some(converted_projection_count) = model
        .convert_projections_to_register_order(performance_attribution)
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Target)?
    else {
        tracing::info!(
            "a paired DFlash2 drafter is installed, but this GPU cannot run the \
             register-ordered kernels drafting needs; decoding stays classic"
        );
        return Ok(None);
    };
    model
        .enable_draft_tree_verification()
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Target)?;
    model
        .enable_row_exact_tree_attention()
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Target)?;
    Ok(Some(converted_projection_count))
}

/// Binds the drafter's checkpoint and requantizes its linears for serving.
fn load_drafter_weights(
    model: &Qwen3_5ResidentModel,
    drafter_directory: &Path,
    drafter_configuration: &DFlashDrafterConfiguration,
) -> Result<(DFlashDrafterWeights, MlxSafetensors), Qwen3_5SpeculativeDraftLoadingError> {
    let weights_file = match std::fs::File::open(drafter_directory.join(DRAFTER_WEIGHTS_FILE_NAME))
    {
        Ok(weights_file) => weights_file,
        Err(source) => {
            return Err(Qwen3_5SpeculativeDraftLoadingError::WeightsUnreadable { source });
        }
    };
    let drafter_checkpoint = model
        .runtime()
        .load_safetensors(weights_file, None)
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Runtime)?;
    let mut drafter_weights =
        bind_drafter_weights(model.runtime(), &drafter_checkpoint, drafter_configuration)
            .map_err(Qwen3_5SpeculativeDraftLoadingError::Weights)?;
    // Served at 4 bits in register order: about 1 GB streamed a round
    // instead of the checkpoint's 3.6 GB of bf16, with no kept-token
    // loss measured against the bf16 drafter on the 27B.
    drafter_weights
        .quantize_linears(model.runtime(), true)
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Runtime)?;
    Ok((drafter_weights, drafter_checkpoint))
}

/// Refuses the drafter when the active limit cannot hold its weights plus
/// headroom beside everything already resident.
fn ensure_memory_headroom_for_drafter(
    model: &Qwen3_5ResidentModel,
    active_memory_limit_bytes: usize,
    drafter_directory: &Path,
) -> Result<(), Qwen3_5SpeculativeDraftLoadingError> {
    let drafter_weights_bytes =
        std::fs::metadata(drafter_directory.join(DRAFTER_WEIGHTS_FILE_NAME))
            .map_err(|source| Qwen3_5SpeculativeDraftLoadingError::WeightsUnreadable { source })?
            .len();
    let memory_snapshot = model
        .runtime()
        .memory_snapshot()
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Runtime)?;
    let active_limit_bytes = u64::try_from(active_memory_limit_bytes).unwrap_or(u64::MAX);
    let active_memory_bytes = u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(0);
    let free_bytes = active_limit_bytes.saturating_sub(active_memory_bytes);
    if free_bytes < drafter_weights_bytes + DRAFTER_LOAD_HEADROOM_BYTES {
        return Err(Qwen3_5SpeculativeDraftLoadingError::InsufficientHeadroom {
            free_bytes,
            required_bytes: drafter_weights_bytes + DRAFTER_LOAD_HEADROOM_BYTES,
        });
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen3_5SpeculativeDraftLoadingError {
    #[error("the model library directory cannot be listed to find a DFlash2 drafter: {source}")]
    LibraryUnlistable { source: std::io::Error },
    #[error("the graphics-processor runtime rejected the DFlash2 drafter load: {0}")]
    Runtime(#[source] runtime_integration::MlxRuntimeError),
    #[error("the target could not be prepared for drafting: {0}")]
    Target(#[source] crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError),
    #[error("the DFlash2 drafter's model.safetensors is unreadable: {source}")]
    WeightsUnreadable { source: std::io::Error },
    #[error("the DFlash2 drafter weights did not bind: {0}")]
    Weights(#[source] DFlashDrafterWeightsError),
    #[error(
        "the active memory limit cannot host the DFlash2 drafter: {free_bytes} bytes free, {required_bytes} bytes required"
    )]
    InsufficientHeadroom {
        free_bytes: u64,
        required_bytes: u64,
    },
}
