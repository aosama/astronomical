//! DFlash2's candidate selector: keeps the top candidates at each proposal
//! slot and walks one coherent path through them.
//!
//! Each slot's score for a candidate is its head logit plus a low-rank edge
//! term for how well it follows the token chosen at the slot before:
//! `sum_r predecessor[prev][r] * projected_hidden[slot][r] * successor[cand][r]`.
//! The first slot follows the anchor. Greedy decoding takes the best score;
//! sampled decoding draws from `q = softmax(scores / temperature)` over the
//! candidates and keeps `q`, which lossless rejection sampling needs.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::drafter_configuration::DFlashDrafterConfiguration;
use super::drafter_weights::CandidateSelectorWeights;

/// How the path picks one candidate per slot.
pub enum PathChoice<'key> {
    Greedy,
    /// Samples at `temperature`, splitting `random_state` once per slot so
    /// the request's random stream advances exactly as many times as slots.
    Sampled {
        temperature: f32,
        random_state: &'key mut MlxArray,
    },
}

/// One drafted path and what produced it.
pub struct ProposedPath {
    /// [1, proposals] token ids, in order.
    pub token_ids: MlxArray,
    /// [1, proposals, candidates] the candidate ids kept at each slot.
    pub candidate_ids: MlxArray,
    /// [1, proposals, candidates] `q` over each slot's candidates when sampled.
    pub candidate_probabilities: Option<MlxArray>,
}

pub fn select_path(
    runtime: &MlxRuntime,
    weights: &CandidateSelectorWeights,
    configuration: &DFlashDrafterConfiguration,
    proposal_hidden_rows: &MlxArray,
    proposal_logits: &MlxArray,
    anchor_token_id: &MlxArray,
    path_choice: &mut PathChoice<'_>,
) -> Result<ProposedPath, MlxRuntimeError> {
    let proposal_count = proposal_logits.shape()[1];
    let candidate_ids = kept_candidate_ids(runtime, configuration, proposal_logits)?;
    let selector_inputs = SelectorInputs {
        candidate_logits: runtime.take_along_axis(proposal_logits, &candidate_ids, -1)?,
        projected_hidden: weights
            .hidden_projection
            .forward(runtime, proposal_hidden_rows)?,
        candidate_ids,
    };
    let mut predecessor_id = anchor_token_id.retain()?;
    let mut path_ids = Vec::with_capacity(proposal_count as usize);
    let mut slot_probabilities = Vec::new();
    for slot in 0..proposal_count {
        let slot_view = SlotView {
            runtime,
            slot,
            candidate_count: configuration.lattice_candidate_count(),
            rank: weights.predecessor_codebook.shape()[1],
        };
        let slot_choice = choose_slot(
            &slot_view,
            weights,
            &selector_inputs,
            &predecessor_id,
            path_choice,
        )?;
        if let Some(probabilities) = slot_choice.candidate_probabilities {
            slot_probabilities.push(probabilities);
        }
        predecessor_id = slot_choice.chosen_token_id;
        path_ids.push(predecessor_id.retain()?);
    }
    let candidate_probabilities = if slot_probabilities.is_empty() {
        None
    } else {
        Some(stack_slots(runtime, &slot_probabilities)?)
    };
    Ok(ProposedPath {
        token_ids: stack_slots(runtime, &path_ids)?,
        candidate_ids: selector_inputs.candidate_ids,
        candidate_probabilities,
    })
}

/// The selector's per-slot inputs, computed once for the whole block.
struct SelectorInputs {
    /// [1, proposals, candidates]
    candidate_ids: MlxArray,
    /// [1, proposals, candidates]
    candidate_logits: MlxArray,
    /// [1, proposals, rank]
    projected_hidden: MlxArray,
}

/// The `lattice_candidate_count` highest-logit ids at every slot, best first,
/// [1, proposals, candidates], from one top-k pass over the vocabulary.
pub(super) fn kept_candidate_ids(
    runtime: &MlxRuntime,
    configuration: &DFlashDrafterConfiguration,
    proposal_logits: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    crate::gpu_token_sampling::vocabulary_top_k(
        runtime,
        proposal_logits,
        configuration.lattice_candidate_count(),
    )
    .map_err(top_k_failure)
}

fn top_k_failure(
    top_k_error: model_primitives::gpu_token_sampling::TokenSamplingError,
) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "keep the drafter's candidates at each slot",
        // Through the engine error so the text keeps its "fatal engine error" prefix.
        description: crate::InferenceEngineError::from(top_k_error).to_string(),
    }
}

/// One slot's pick, and `q` over its candidates when sampled.
struct SlotChoice {
    /// [1]
    chosen_token_id: MlxArray,
    candidate_probabilities: Option<MlxArray>,
}

fn choose_slot(
    slot_view: &SlotView<'_>,
    weights: &CandidateSelectorWeights,
    selector_inputs: &SelectorInputs,
    predecessor_id: &MlxArray,
    path_choice: &mut PathChoice<'_>,
) -> Result<SlotChoice, MlxRuntimeError> {
    let runtime = slot_view.runtime;
    let slot_candidates = slot_view.candidates(&selector_inputs.candidate_ids)?;
    let edge_scores = slot_view.edge_scores(
        weights,
        predecessor_id,
        &selector_inputs.projected_hidden,
        &slot_candidates,
    )?;
    let scores = runtime.add(
        &slot_view.candidate_logits(&selector_inputs.candidate_logits)?,
        &edge_scores,
    )?;
    let (chosen_index, candidate_probabilities) = match path_choice {
        PathChoice::Greedy => (runtime.argmax_axis(&scores, -1)?, None),
        PathChoice::Sampled {
            temperature,
            random_state,
        } => {
            let tempered = runtime.multiply_scalar(&scores, 1.0 / *temperature)?;
            let probabilities = runtime.softmax_axis(&tempered, -1)?;
            let (next_random_state, slot_key) = runtime.split_random_key(random_state)?;
            **random_state = next_random_state;
            (
                runtime.categorical_sample_with_key(&tempered, -1, &slot_key)?,
                Some(probabilities),
            )
        }
    };
    let chosen = runtime.take_along_axis(
        &slot_candidates,
        &runtime.reshape(&chosen_index, &[1, 1])?,
        -1,
    )?;
    Ok(SlotChoice {
        chosen_token_id: runtime.reshape(&chosen, &[1])?,
        candidate_probabilities,
    })
}

fn stack_slots(runtime: &MlxRuntime, per_slot: &[MlxArray]) -> Result<MlxArray, MlxRuntimeError> {
    let references: Vec<&MlxArray> = per_slot.iter().collect();
    Ok(runtime.stack_axis(&references, 1)?)
}

/// One proposal slot's slices of the batched selector inputs.
struct SlotView<'view> {
    runtime: &'view MlxRuntime,
    slot: i32,
    candidate_count: i32,
    rank: i32,
}

impl SlotView<'_> {
    /// [1, candidates] ids at this slot.
    fn candidates(&self, candidate_ids: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.slot_row(candidate_ids, self.candidate_count)
    }

    fn candidate_logits(&self, candidate_logits: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.slot_row(candidate_logits, self.candidate_count)
    }

    /// [1, candidates] edge score of each candidate following `predecessor_id`.
    fn edge_scores(
        &self,
        weights: &CandidateSelectorWeights,
        predecessor_id: &MlxArray,
        projected_hidden: &MlxArray,
        slot_candidates: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let runtime = self.runtime;
        let predecessor_row =
            runtime.take_axis(&weights.predecessor_codebook, predecessor_id, 0)?;
        let weighted_predecessor = runtime.multiply(
            &predecessor_row,
            &self.slot_row(projected_hidden, self.rank)?,
        )?;
        let successor_rows = runtime.reshape(
            &runtime.take_axis(
                &weights.successor_codebook,
                &runtime.reshape(slot_candidates, &[self.candidate_count])?,
                0,
            )?,
            &[1, self.candidate_count, self.rank],
        )?;
        let products = runtime.multiply(
            &runtime.expand_dims(&weighted_predecessor, 1)?,
            &successor_rows,
        )?;
        Ok(runtime.sum_axis(&products, -1, false)?)
    }

    /// `[1, slots, width]` → this slot's `[1, width]`.
    fn slot_row(&self, per_slot: &MlxArray, width: i32) -> Result<MlxArray, MlxRuntimeError> {
        let row = self.runtime.slice(
            per_slot,
            &[0, self.slot, 0],
            &[1, self.slot + 1, width],
            &[1, 1, 1],
        )?;
        Ok(self.runtime.reshape(&row, &[1, width])?)
    }
}
