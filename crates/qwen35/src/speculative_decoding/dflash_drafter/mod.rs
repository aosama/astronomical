//! The DFlash2 block drafter (z-lab, MIT), translated from z-lab's MLX
//! reference `dflash/model_mlx.py`; see `third-party/THIRD_PARTY_NOTICES`.
//!
//! One pass drafts a whole block: the last committed token followed by mask
//! slots goes through a few layers that attend to fused target hidden rows,
//! the target's own head scores every slot, and the candidate selector walks
//! one coherent path through each slot's best candidates, or the draft tree
//! searches them for the branches most likely to be kept.

#[cfg_attr(not(feature = "direct-mlx"), allow(dead_code))]
pub mod block_attention_mask;
#[cfg(feature = "direct-mlx")]
pub mod candidate_selector;
#[cfg(feature = "direct-mlx")]
pub mod draft_lattice;
pub mod draft_tree;
#[cfg(feature = "direct-mlx")]
mod drafter_attention;
pub mod drafter_configuration;
#[cfg(feature = "direct-mlx")]
pub mod drafter_context_state;
#[cfg(feature = "direct-mlx")]
pub mod drafter_forward;
#[cfg(feature = "direct-mlx")]
pub mod drafter_linear;
#[cfg(feature = "direct-mlx")]
pub mod drafter_weights;
#[cfg(feature = "direct-mlx")]
mod grouped_dynamic_convolution;

#[cfg(test)]
mod block_attention_mask_tests;
#[cfg(test)]
mod draft_tree_tests;
#[cfg(test)]
mod drafter_configuration_tests;

#[cfg(feature = "direct-mlx")]
pub use candidate_selector::{PathChoice, ProposedPath, select_path};
#[cfg(feature = "direct-mlx")]
pub use draft_lattice::draft_lattice;
pub use draft_tree::{
    DraftLattice, DraftTree, DraftTreeOptions, KeyedNoise, TENSORFOLD_TREE_BLOCK_ROW_COUNT,
    best_first_draft_tree, tree_block_row_count,
};
pub use drafter_configuration::{
    DFlashDrafterConfiguration, DFlashDrafterConfigurationError, DrafterLayerAttention,
};
#[cfg(feature = "direct-mlx")]
pub use drafter_context_state::DrafterContextStates;
#[cfg(feature = "direct-mlx")]
pub use drafter_forward::{DFlashDraftTarget, drafted_hidden_rows};
#[cfg(feature = "direct-mlx")]
pub use drafter_weights::{DFlashDrafterWeights, DFlashDrafterWeightsError, bind_drafter_weights};
