//! Everything the draft tree searches, computed on the GPU and read back in
//! one synchronization: each slot's kept candidates and logits, and the
//! selector's edge score for every parent a candidate could hang under.
//!
//! A node's parent is always one of the previous slot's candidates (or the
//! anchor, for the first slot), so a `[slots - 1, candidates, candidates]`
//! table holds every edge the search can ask for: 6 x 16 x 16 scores for the
//! 27B's drafter, against walking the codebooks once per node on the host.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::{MlxArray, MlxDtype};

use super::candidate_selector::kept_candidate_ids;
use super::draft_tree::DraftLattice;
use super::drafter_configuration::DFlash2DrafterConfiguration;
use super::drafter_weights::CandidateSelectorWeights;

/// The lattice of one drafted block: `proposal_hidden_rows` [1, slots, H]
/// from the drafter, `proposal_logits` [1, slots, vocabulary] from the
/// target's head, and the `anchor_token_id` the block follows.
pub fn draft_lattice(
    runtime: &MlxRuntime,
    weights: &CandidateSelectorWeights,
    configuration: &DFlash2DrafterConfiguration,
    proposal_hidden_rows: &MlxArray,
    proposal_logits: &MlxArray,
    anchor_token_id: u32,
) -> Result<DraftLattice, MlxRuntimeError> {
    let slot_count = proposal_logits.shape()[1];
    let candidates = SlotCandidates::kept(runtime, configuration, proposal_logits)?;
    let projected = projected_slots(runtime, weights, proposal_hidden_rows, slot_count)?;
    let codebook_rows = CodebookRows::of(runtime, weights, &candidates.token_ids)?;
    let anchor_edges = anchor_edge_scores(
        runtime,
        weights,
        &projected,
        &codebook_rows,
        anchor_token_id,
    )?;
    let slot_edges = slot_edge_scores(runtime, &projected, &codebook_rows, slot_count)?;
    runtime.evaluate_arrays(&[
        &candidates.token_ids,
        &candidates.logits,
        &anchor_edges,
        &slot_edges,
    ])?;
    Ok(DraftLattice {
        depth: slot_count as usize,
        candidate_count: configuration.selector_candidate_count as usize,
        candidate_ids: candidates.token_ids.to_vec_u32()?,
        unary_scores: candidates.logits.to_vec_f32()?,
        anchor_edge_scores: anchor_edges.to_vec_f32()?,
        edge_scores: slot_edges.to_vec_f32()?,
    })
}

/// Each slot's kept candidates: `[slots, candidates]` uint32 ids and their
/// float32 logits.
struct SlotCandidates {
    token_ids: MlxArray,
    logits: MlxArray,
}

impl SlotCandidates {
    fn kept(
        runtime: &MlxRuntime,
        configuration: &DFlash2DrafterConfiguration,
        proposal_logits: &MlxArray,
    ) -> Result<Self, MlxRuntimeError> {
        let slot_count = proposal_logits.shape()[1];
        let candidate_count = configuration.selector_candidate_count;
        let candidate_ids = kept_candidate_ids(runtime, configuration, proposal_logits)?;
        let logits = runtime.take_along_axis(proposal_logits, &candidate_ids, -1)?;
        Ok(Self {
            token_ids: runtime.reshape(
                &runtime.astype(&candidate_ids, MlxDtype::UInt32)?,
                &[slot_count, candidate_count],
            )?,
            logits: runtime.reshape(
                &runtime.astype(&logits, MlxDtype::Float32)?,
                &[slot_count, candidate_count],
            )?,
        })
    }
}

/// `[slots, rank]` float32: each slot's hidden row through the selector's
/// projection. Edge sums run in float32, as TensorFold's run in double
/// precision from the same bf16 values: a bf16 sum over 256 ranks can reorder
/// close candidates.
fn projected_slots(
    runtime: &MlxRuntime,
    weights: &CandidateSelectorWeights,
    proposal_hidden_rows: &MlxArray,
    slot_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let projected = weights
        .hidden_projection
        .forward(runtime, proposal_hidden_rows)?;
    let rank = projected.shape()[2];
    Ok(runtime.astype(
        &runtime.reshape(&projected, &[slot_count, rank])?,
        MlxDtype::Float32,
    )?)
}

/// The codebook rows of every kept candidate, `[slots, candidates, rank]`
/// float32.
struct CodebookRows {
    predecessor: MlxArray,
    successor: MlxArray,
}

impl CodebookRows {
    fn of(
        runtime: &MlxRuntime,
        weights: &CandidateSelectorWeights,
        candidate_ids: &MlxArray,
    ) -> Result<Self, MlxRuntimeError> {
        let [slot_count, candidate_count] = [candidate_ids.shape()[0], candidate_ids.shape()[1]];
        let rank = weights.predecessor_codebook.shape()[1];
        let flat_candidate_ids = runtime.reshape(candidate_ids, &[slot_count * candidate_count])?;
        let shape = [slot_count, candidate_count, rank];
        Ok(Self {
            predecessor: float32_rows(
                runtime,
                &weights.predecessor_codebook,
                &flat_candidate_ids,
                shape,
            )?,
            successor: float32_rows(
                runtime,
                &weights.successor_codebook,
                &flat_candidate_ids,
                shape,
            )?,
        })
    }
}

/// `[candidates]`: each first-slot candidate's edge after the anchor.
fn anchor_edge_scores(
    runtime: &MlxRuntime,
    weights: &CandidateSelectorWeights,
    projected: &MlxArray,
    codebook_rows: &CodebookRows,
    anchor_token_id: u32,
) -> Result<MlxArray, MlxRuntimeError> {
    let rank = projected.shape()[1];
    let candidate_count = codebook_rows.successor.shape()[1];
    let anchor = runtime.array_from_u32(&[anchor_token_id], &[1])?;
    let anchor_row = runtime.astype(
        &runtime.take_axis(&weights.predecessor_codebook, &anchor, 0)?,
        MlxDtype::Float32,
    )?;
    let first_projected = runtime.slice(projected, &[0, 0], &[1, rank], &[1, 1])?;
    let weighted_anchor = runtime.multiply(&anchor_row, &first_projected)?;
    let first_successors = runtime.reshape(
        &runtime.slice(
            &codebook_rows.successor,
            &[0, 0, 0],
            &[1, candidate_count, rank],
            &[1, 1, 1],
        )?,
        &[candidate_count, rank],
    )?;
    let products = runtime.multiply(&first_successors, &weighted_anchor)?;
    Ok(runtime.sum_axis(&products, -1, false)?)
}

/// `[slots - 1, candidates, candidates]`: at `[s][i][j]`, the edge of slot
/// `s + 1`'s candidate `j` after slot `s`'s candidate `i`, which weighs the
/// rank with slot `s + 1`'s projected row.
fn slot_edge_scores(
    runtime: &MlxRuntime,
    projected: &MlxArray,
    codebook_rows: &CodebookRows,
    slot_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let rank = projected.shape()[1];
    let candidate_count = codebook_rows.successor.shape()[1];
    let parents = runtime.slice(
        &codebook_rows.predecessor,
        &[0, 0, 0],
        &[slot_count - 1, candidate_count, rank],
        &[1, 1, 1],
    )?;
    let child_projected = runtime.reshape(
        &runtime.slice(projected, &[1, 0], &[slot_count, rank], &[1, 1])?,
        &[slot_count - 1, 1, rank],
    )?;
    let weighted_parents = runtime.multiply(&parents, &child_projected)?;
    let children = runtime.slice(
        &codebook_rows.successor,
        &[1, 0, 0],
        &[slot_count, candidate_count, rank],
        &[1, 1, 1],
    )?;
    let children_transposed = runtime.transpose_axes(&children, &[0, 2, 1])?;
    Ok(runtime.matmul(&weighted_parents, &children_transposed)?)
}

/// Codebook rows of `token_ids` as float32, shaped `shape`.
fn float32_rows(
    runtime: &MlxRuntime,
    codebook: &MlxArray,
    token_ids: &MlxArray,
    shape: [i32; 3],
) -> Result<MlxArray, MlxRuntimeError> {
    Ok(runtime.astype(
        &runtime.reshape(&runtime.take_axis(codebook, token_ids, 0)?, &shape)?,
        MlxDtype::Float32,
    )?)
}
