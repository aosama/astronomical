//! The lattice of a drafter without a candidate selector (DFlash): each
//! slot's best candidates by the target head's logit, read back from the GPU
//! in one synchronization.
//!
//! DFlash2's selector adds a score for how well a candidate follows its
//! parent, so the tree can rank the same slot differently under different
//! branches. Without it every branch sees the same ranking, and the tree
//! spends its nodes on each slot's runners-up instead.

use runtime_integration::{MlxRuntime, MlxRuntimeError};

use mlx_bridge::MlxArray;

use super::draft_lattice::SlotCandidates;
use super::draft_tree::DraftLattice;
use super::drafter_configuration::DFlashDrafterConfiguration;

/// The lattice of one drafted block from `proposal_logits`
/// [1, slots, vocabulary], the target head's scores of the drafter's rows.
pub fn head_only_draft_lattice(
    runtime: &MlxRuntime,
    configuration: &DFlashDrafterConfiguration,
    proposal_logits: &MlxArray,
) -> Result<DraftLattice, MlxRuntimeError> {
    let slot_count = proposal_logits.shape()[1];
    let candidates = SlotCandidates::kept(runtime, configuration, proposal_logits)?;
    runtime.evaluate_arrays(&[&candidates.token_ids, &candidates.logits])?;
    Ok(DraftLattice::without_edges(
        slot_count as usize,
        configuration.lattice_candidate_count() as usize,
        candidates.token_ids.to_vec_u32()?,
        candidates.logits.to_vec_f32()?,
    ))
}
