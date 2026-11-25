//! The engine-owned DFlash drafter runtime, of either generation.
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

use ipc_protocol::DraftModelLocation;
use runtime_integration::MlxSafetensors;

use super::draft_round_target::DraftRoundTarget;
use crate::PerformanceAttribution;
use engine_core::speculative_decoding::dflash_drafter::{
    DFlashDrafterConfiguration, DFlashDrafterWeights, DFlashDrafterWeightsError, DraftTreeOptions,
    bind_drafter_weights,
};

use super::draft_tree_size::draft_tree_options_for_target;
use engine_core::speculative_decoding::drafter_discovery::{
    BUNDLED_DRAFTER_DIRECTORY_NAME, DrafterPairing, paired_drafter_directory,
};

/// Headroom the drafter load insists on beyond its own weight bytes, so the
/// first verify workspace never starts life squeezed against the ceiling.
const DRAFTER_LOAD_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;
const DRAFTER_WEIGHTS_FILE_NAME: &str = "model.safetensors";

/// What looking for and loading a drafter found.
pub enum Qwen3_5DraftRuntimeLoadOutcome {
    /// A paired drafter is loaded; boxed because it holds the drafter's weights.
    Loaded(Box<Qwen3_5SpeculativeDraftRuntime>),
    /// No drafter in the model's folder or the library pairs with it.
    NoPairedDrafter,
    /// A drafter pairs, but this GPU cannot run the kernels drafting needs.
    KernelsUnavailable,
}

/// `Bundled` when the drafter is the model's own `drafter` folder.
fn drafter_location(target_model_directory: &Path, drafter_directory: &Path) -> DraftModelLocation {
    if drafter_directory == target_model_directory.join(BUNDLED_DRAFTER_DIRECTORY_NAME) {
        DraftModelLocation::Bundled
    } else {
        DraftModelLocation::Library
    }
}

/// The loaded drafter a session drafts with.
pub struct Qwen3_5SpeculativeDraftRuntime {
    pub drafter_configuration: DFlashDrafterConfiguration,
    pub drafter_weights: DFlashDrafterWeights,
    /// The tree size and search settings, chosen for this target's kind.
    pub tree_options: DraftTreeOptions,
    /// Whether the drafter came from the model's own folder or the library.
    pub drafter_location: DraftModelLocation,
    /// What the drafter's weights hold in memory, when the engine that
    /// hosts it measured that; zero where nobody plans memory around it.
    pub hosted_weight_bytes: u64,
    /// Keeps the drafter's lazily mapped codebooks alive.
    _drafter_checkpoint: MlxSafetensors,
}

/// Printed where a model's accelerators are: the weights themselves say
/// nothing a log line wants.
impl std::fmt::Debug for Qwen3_5SpeculativeDraftRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen3_5SpeculativeDraftRuntime")
            .field("generation", &self.drafter_configuration.generation)
            .field("tree_node_budget", &self.tree_options.node_budget)
            .field("hosted_weight_bytes", &self.hosted_weight_bytes)
            .finish_non_exhaustive()
    }
}

impl Qwen3_5SpeculativeDraftRuntime {
    /// Finds and loads the drafter for `model`, saying apart a library that
    /// holds none that pairs with it from a GPU that cannot run the
    /// register-ordered kernels a drafting session needs, because status
    /// reports the second as a drafter installed but unused.
    pub fn load_for_target<Target: DraftRoundTarget>(
        model: &mut Target,
        active_memory_limit_bytes: usize,
        target_model_directory: &Path,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5DraftRuntimeLoadOutcome, Qwen3_5SpeculativeDraftLoadingError> {
        let pairing = DrafterPairing {
            target_layer_count: model.draft_tree_model_base().config().layer_count() as i32,
            target_hidden_size: model.draft_tree_model_base().config().hidden_size() as i32,
            target_vocabulary_size: model.draft_tree_model_base().config().vocabulary_size() as i32,
        };
        let Some((drafter_directory, drafter_configuration)) =
            paired_drafter_directory(target_model_directory, &pairing)?
        else {
            return Ok(Qwen3_5DraftRuntimeLoadOutcome::NoPairedDrafter);
        };
        let drafter_location = drafter_location(target_model_directory, &drafter_directory);
        secure_room_for_the_drafter(model, active_memory_limit_bytes, &drafter_directory)?;
        let Some(converted_projection_count) = model
            .prepare_for_draft_trees(performance_attribution)
            .map_err(Qwen3_5SpeculativeDraftLoadingError::Target)?
        else {
            return Ok(Qwen3_5DraftRuntimeLoadOutcome::KernelsUnavailable);
        };
        let (drafter_weights, drafter_checkpoint) =
            load_drafter_weights(model, &drafter_directory, &drafter_configuration)?;
        let tree_options =
            draft_tree_options_for_target(model.draft_tree_model_base().config().expert_count());
        tracing::info!(
            drafter_directory = %drafter_directory.display(),
            drafter_generation = ?drafter_configuration.generation,
            converted_projection_count,
            tree_node_budget = tree_options.node_budget,
            "decode path: DFlash draft trees"
        );
        Ok(Qwen3_5DraftRuntimeLoadOutcome::Loaded(Box::new(Self {
            drafter_configuration,
            drafter_weights,
            tree_options,
            drafter_location,
            hosted_weight_bytes: 0,
            _drafter_checkpoint: drafter_checkpoint,
        })))
    }
}

/// Binds the drafter's checkpoint and requantizes its linears for serving.
fn load_drafter_weights<Target: DraftRoundTarget>(
    model: &Target,
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
        .draft_tree_model_base()
        .runtime()
        .load_safetensors(weights_file, None)
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Runtime)?;
    let mut drafter_weights = bind_drafter_weights(
        model.draft_tree_model_base().runtime(),
        &drafter_checkpoint,
        drafter_configuration,
    )
    .map_err(Qwen3_5SpeculativeDraftLoadingError::Weights)?;
    // Served at 4 bits in register order: about 1 GB streamed a round
    // instead of the checkpoint's 3.6 GB of bf16, with no kept-token
    // loss measured against the bf16 drafter on the 27B.
    drafter_weights
        .quantize_linears(model.draft_tree_model_base().runtime(), true)
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Runtime)?;
    Ok((drafter_weights, drafter_checkpoint))
}

/// Has the target make room for the drafter's load, or, on an engine that
/// cannot, refuses the drafter unless the room is already there.
fn secure_room_for_the_drafter<Target: DraftRoundTarget>(
    model: &mut Target,
    active_memory_limit_bytes: usize,
    drafter_directory: &Path,
) -> Result<(), Qwen3_5SpeculativeDraftLoadingError> {
    let drafter_load_bytes = drafter_load_byte_count(drafter_directory)?;
    if model.make_room_for_a_drafter(drafter_load_bytes) {
        return Ok(());
    }
    ensure_memory_headroom_for_drafter(model, active_memory_limit_bytes, drafter_load_bytes)
}

/// What loading the drafter needs under the limit: its checkpoint, which
/// is held whole until its linears are requantized, and the headroom.
fn drafter_load_byte_count(
    drafter_directory: &Path,
) -> Result<u64, Qwen3_5SpeculativeDraftLoadingError> {
    let drafter_weights_bytes =
        std::fs::metadata(drafter_directory.join(DRAFTER_WEIGHTS_FILE_NAME))
            .map_err(|source| Qwen3_5SpeculativeDraftLoadingError::WeightsUnreadable { source })?
            .len();
    Ok(drafter_weights_bytes + DRAFTER_LOAD_HEADROOM_BYTES)
}

/// Refuses the drafter when the active limit cannot hold its weights plus
/// headroom beside everything already resident.
fn ensure_memory_headroom_for_drafter<Target: DraftRoundTarget>(
    model: &Target,
    active_memory_limit_bytes: usize,
    drafter_load_bytes: u64,
) -> Result<(), Qwen3_5SpeculativeDraftLoadingError> {
    let memory_snapshot = model
        .draft_tree_model_base()
        .runtime()
        .memory_snapshot()
        .map_err(Qwen3_5SpeculativeDraftLoadingError::Runtime)?;
    let active_limit_bytes = u64::try_from(active_memory_limit_bytes).unwrap_or(u64::MAX);
    let active_memory_bytes = u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(0);
    let free_bytes = active_limit_bytes.saturating_sub(active_memory_bytes);
    if free_bytes < drafter_load_bytes {
        return Err(Qwen3_5SpeculativeDraftLoadingError::InsufficientHeadroom {
            free_bytes,
            required_bytes: drafter_load_bytes,
        });
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen3_5SpeculativeDraftLoadingError {
    #[error(transparent)]
    LibraryUnlistable(
        #[from] engine_core::speculative_decoding::drafter_discovery::DrafterDiscoveryError,
    ),
    #[error("the graphics-processor runtime rejected the DFlash drafter load: {0}")]
    Runtime(#[source] runtime_integration::MlxRuntimeError),
    #[error("the target could not be prepared for drafting: {0}")]
    Target(#[source] text_generation::execution_error::TextExecutionError),
    #[error("the DFlash drafter's model.safetensors is unreadable: {source}")]
    WeightsUnreadable { source: std::io::Error },
    #[error("the DFlash drafter weights did not bind: {0}")]
    Weights(#[source] DFlashDrafterWeightsError),
    #[error(
        "the active memory limit cannot host the DFlash drafter: {free_bytes} bytes free, {required_bytes} bytes required"
    )]
    InsufficientHeadroom {
        free_bytes: u64,
        required_bytes: u64,
    },
}
