//! One speculative decoding round: draft a tree, verify it in one target
//! pass, keep one chain.
//!
//! A round takes the bonus token (the last token the target itself chose,
//! not yet in its state), drafts a block with the DFlash2 drafter, searches
//! the block's candidate lattice for a tree of likely continuations, and
//! verifies every row of that tree in a single target pass. The target then
//! picks its own token after every row: the argmax under greedy decoding,
//! or the keyed sample at the row's position under sampled decoding. The
//! round keeps the chain those picks follow through the tree and commits
//! exactly that chain to the target's state. Every emitted token is one the
//! target chose itself; the tree only decides how many of them one pass
//! yields.

use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model::draft_tree_verification::DraftTreeTarget;
use crate::qwen3_5_resident::model::{Qwen3_5ResidentModel, TextExecutionError};
use engine_core::speculative_decoding::dflash_drafter::{
    DFlashDraftTarget, DraftLattice, DraftTreeOptions, DrafterContextStates, KeyedNoise,
    best_first_draft_tree, draft_lattice, drafted_hidden_rows, head_only_draft_lattice,
    tree_block_row_count,
};
use engine_core::speculative_decoding::keyed_sampling::{RequestKeyedNoise, keyed_sample};
use engine_core::speculative_decoding::tree_rows::TreeRows;
use engine_core::speculative_decoding::verification_decision::TargetDistribution;
use text_generation::decoder_state::request_state::RequestDecoderStateStack;

use super::engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime;
use super::speculation_session_admission::VerificationTokenChoice;

/// One request's drafting state: the drafter's kept context, the target
/// tap rows it has not absorbed yet, and the request's sampling key.
pub struct DraftSession {
    context_states: Option<DrafterContextStates>,
    /// [1, rows, taps x hidden]: taps of rows committed since the drafter
    /// last ran, starting at `first_pending_position`.
    pending_tap_rows: Option<MlxArray>,
    first_pending_position: i32,
    keyed_seed: u64,
}

impl DraftSession {
    pub fn new(keyed_seed: u64) -> Self {
        Self {
            context_states: None,
            pending_tap_rows: None,
            first_pending_position: 0,
            keyed_seed,
        }
    }

    /// Hands the drafter the target's taps (one [1, rows, hidden] array per
    /// tapped layer) for rows committed from `first_position` on. Rows the
    /// drafter's window will never show are dropped as they fall out of it:
    /// a long prompt would otherwise hold about 50 MB of taps per thousand
    /// tokens until the first round, for rows no drafter layer reads.
    pub fn append_tap_rows(
        &mut self,
        model: &Qwen3_5ResidentModel,
        context_row_limit: Option<i32>,
        layer_taps: &[MlxArray],
        first_position: i32,
    ) -> Result<(), TextExecutionError> {
        let runtime = model.runtime();
        let tap_references: Vec<&MlxArray> = layer_taps.iter().collect();
        let joined_taps = runtime.concatenate_axis(&tap_references, -1)?;
        let pending_tap_rows = match self.pending_tap_rows.take() {
            Some(pending) => runtime.concatenate_axis(&[&pending, &joined_taps], 1)?,
            None => {
                self.first_pending_position = first_position;
                joined_taps
            }
        };
        let pending_shape = pending_tap_rows.shape();
        let dropped_row_count = rows_beyond_the_context_limit(pending_shape[1], context_row_limit);
        self.pending_tap_rows = Some(if dropped_row_count > 0 {
            self.first_pending_position += dropped_row_count;
            runtime.slice(
                &pending_tap_rows,
                &[0, dropped_row_count, 0],
                &pending_shape,
                &[1, 1, 1],
            )?
        } else {
            pending_tap_rows
        });
        Ok(())
    }

    /// Runs the pending tap rows so they hold no graph of the forward that
    /// produced them.
    pub fn evaluate_pending_tap_rows(
        &self,
        model: &Qwen3_5ResidentModel,
    ) -> Result<(), TextExecutionError> {
        if let Some(pending_tap_rows) = self.pending_tap_rows.as_ref() {
            model.runtime().evaluate_arrays(&[pending_tap_rows])?;
        }
        Ok(())
    }
}

/// How many of the oldest `pending_row_count` rows lie beyond what the
/// drafter can attend to.
pub fn rows_beyond_the_context_limit(
    pending_row_count: i32,
    context_row_limit: Option<i32>,
) -> i32 {
    match context_row_limit {
        Some(row_limit) => (pending_row_count - row_limit).max(0),
        None => 0,
    }
}

/// Everything one round needs besides the owners it mutates.
#[derive(Debug, Clone, Copy)]
pub struct DraftTreeRoundRequest {
    /// The token the target itself chose last; it anchors the tree.
    pub bonus_token_id: u32,
    /// Tokens already in the target's state: the bonus token's position.
    pub context_length_tokens: u32,
    /// How the target picks one token after every verified row.
    pub target_token_choice: VerificationTokenChoice,
}

/// What one round produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftTreeRoundOutcome {
    /// Tokens the round emits, in order: the bonus, the kept drafts, and
    /// the target's pick after the last kept row (not yet in its state).
    pub emitted_token_ids: Vec<u32>,
    /// How many drafted tokens the target's picks confirmed.
    pub accepted_proposal_count: usize,
    /// Drafted tokens the target verified this round.
    pub drafted_token_count: usize,
    /// Tokens in the target's state after the round.
    pub context_length_after_round: u32,
}

/// Runs one round and leaves `request_decoder_state` advanced over the
/// bonus token and the kept drafts.
pub fn run_draft_tree_round(
    model: &Qwen3_5ResidentModel,
    request_decoder_state: &mut RequestDecoderStateStack,
    draft_runtime: &Qwen3_5SpeculativeDraftRuntime,
    draft_session: &mut DraftSession,
    round_request: &DraftTreeRoundRequest,
) -> Result<DraftTreeRoundOutcome, TextExecutionError> {
    let anchor_position = i32::try_from(round_request.context_length_tokens).map_err(|_| {
        TextExecutionError::InvalidInput {
            description: "the context length exceeds the MLX int32 range",
        }
    })?;
    let tree_rows = drafted_tree_rows(model, draft_runtime, draft_session, round_request)?;
    let tap_layer_ids = &draft_runtime.drafter_configuration.target_tap_layer_ids;
    let verification = model.verify_draft_tree(
        &tree_rows,
        round_request.context_length_tokens,
        request_decoder_state,
        tap_layer_ids,
    )?;
    let picks = target_picks(
        model,
        &verification.per_row_logits,
        &tree_rows.positions(anchor_position),
        round_request.target_token_choice,
        draft_session.keyed_seed,
    )?;
    let kept = tree_rows.kept_chain(&picks);
    let mut kept_row_ids = Vec::with_capacity(kept.rows.len());
    let mut emitted_token_ids = Vec::with_capacity(kept.rows.len() + 1);
    for kept_row in &kept.rows {
        kept_row_ids.push(*kept_row as i32);
        emitted_token_ids.push(tree_rows.token_ids[*kept_row]);
    }
    emitted_token_ids.push(kept.next_token_id);
    let kept_row_array = model
        .runtime()
        .array_from_i32(&kept_row_ids, &[kept_row_ids.len() as i32])?;
    let mut kept_taps = Vec::with_capacity(verification.hidden_state_taps.len());
    for layer_tap in &verification.hidden_state_taps {
        kept_taps.push(model.runtime().take_axis(layer_tap, &kept_row_array, 1)?);
    }
    draft_session.append_tap_rows(
        model,
        draft_runtime.drafter_configuration.context_row_limit(),
        &kept_taps,
        anchor_position,
    )?;
    model.commit_draft_tree(request_decoder_state, verification, &kept.rows)?;
    Ok(DraftTreeRoundOutcome {
        emitted_token_ids,
        accepted_proposal_count: kept.rows.len() - 1,
        drafted_token_count: tree_rows.row_count() - 1,
        context_length_after_round: round_request.context_length_tokens + kept.rows.len() as u32,
    })
}

/// Drafts one block from the bonus token and searches its lattice: the
/// tree's rows, the bonus first.
fn drafted_tree_rows(
    model: &Qwen3_5ResidentModel,
    draft_runtime: &Qwen3_5SpeculativeDraftRuntime,
    draft_session: &mut DraftSession,
    round_request: &DraftTreeRoundRequest,
) -> Result<TreeRows, TextExecutionError> {
    let runtime = model.runtime();
    let configuration = &draft_runtime.drafter_configuration;
    let pending_tap_rows =
        draft_session
            .pending_tap_rows
            .take()
            .ok_or(TextExecutionError::InvalidInput {
                description: "a draft round needs the target's taps for the rows before it",
            })?;
    let first_pending_position = draft_session.first_pending_position;
    if draft_session.context_states.is_none() {
        draft_session.context_states = Some(DrafterContextStates::new(
            configuration,
            first_pending_position,
        ));
    }
    let Some(context_states) = draft_session.context_states.as_mut() else {
        return Err(TextExecutionError::InvalidInput {
            description: "the drafter's context states were not created",
        });
    };
    let tree_options = draft_runtime.tree_options;
    let block_row_count =
        tree_block_row_count(tree_options.node_budget, configuration.block_size as usize);
    let mut block_ids = vec![configuration.mask_token_id; block_row_count];
    block_ids[0] = round_request.bonus_token_id;
    let block = runtime.array_from_u32(&block_ids, &[1, block_row_count as i32])?;
    let target: &dyn DFlashDraftTarget = model;
    let hidden_rows = drafted_hidden_rows(
        runtime,
        &draft_runtime.drafter_weights,
        configuration,
        target,
        context_states,
        &block,
        &pending_tap_rows,
    )?;
    let draft_logits = target.head_logits(runtime, &hidden_rows)?;
    let lattice = drafted_lattice(
        model,
        draft_runtime,
        &hidden_rows,
        &draft_logits,
        round_request.bonus_token_id,
    )?;
    let keyed_noise = RequestKeyedNoise {
        seed: draft_session.keyed_seed,
    };
    let (noise, sampling_temperature): (Option<&dyn KeyedNoise>, f64) =
        match round_request.target_token_choice {
            VerificationTokenChoice::HighestLogit => (None, 1.0),
            VerificationTokenChoice::TopKTopP {
                temperature_thousandths,
                ..
            } => (
                Some(&keyed_noise),
                f64::from(temperature_thousandths) / 1_000.0,
            ),
        };
    let tree = best_first_draft_tree(
        &lattice,
        u64::from(round_request.context_length_tokens),
        &DraftTreeOptions {
            sampling_temperature,
            ..tree_options
        },
        noise,
    );
    Ok(TreeRows::from_draft_tree(
        round_request.bonus_token_id,
        &tree,
    ))
}

/// The block's candidate lattice: scored by DFlash2's candidate selector,
/// or by the target head alone for a DFlash drafter, which has none.
fn drafted_lattice(
    model: &Qwen3_5ResidentModel,
    draft_runtime: &Qwen3_5SpeculativeDraftRuntime,
    hidden_rows: &MlxArray,
    draft_logits: &MlxArray,
    bonus_token_id: u32,
) -> Result<DraftLattice, TextExecutionError> {
    let runtime = model.runtime();
    let configuration = &draft_runtime.drafter_configuration;
    let lattice = match &draft_runtime.drafter_weights.candidate_selector {
        Some(selector_weights) => draft_lattice(
            runtime,
            selector_weights,
            configuration,
            hidden_rows,
            draft_logits,
            bonus_token_id,
        )?,
        None => head_only_draft_lattice(runtime, configuration, draft_logits)?,
    };
    Ok(lattice)
}

/// The target's pick after every row of `per_row_logits` [1, rows, V]: the
/// argmax, or the keyed sample at the position after the row's own.
fn target_picks(
    model: &Qwen3_5ResidentModel,
    per_row_logits: &MlxArray,
    row_positions: &[i32],
    target_token_choice: VerificationTokenChoice,
    keyed_seed: u64,
) -> Result<Vec<u32>, TextExecutionError> {
    let runtime = model.runtime();
    let (temperature_thousandths, top_k, top_p_thousandths) = match target_token_choice {
        VerificationTokenChoice::HighestLogit => {
            let argmax = runtime.astype(
                &runtime.argmax_axis(per_row_logits, -1)?,
                mlx_bridge::MlxDtype::UInt32,
            )?;
            runtime.evaluate_arrays(&[&argmax])?;
            return Ok(argmax.to_vec_u32()?);
        }
        VerificationTokenChoice::TopKTopP {
            temperature_thousandths,
            top_k,
            top_p_thousandths,
        } => (temperature_thousandths, top_k, top_p_thousandths),
    };
    let (distribution_ids, distribution_probabilities) = model
        .sampling_distribution_rows(
            per_row_logits,
            temperature_thousandths,
            top_p_thousandths,
            top_k,
        )
        .map_err(sampling_failure)?;
    runtime.evaluate_arrays(&[&distribution_ids, &distribution_probabilities])?;
    let kept_ids = distribution_ids.to_vec_u32()?;
    let kept_probabilities = runtime.array_to_vec_f32(&distribution_probabilities)?;
    let kept_per_row = kept_ids.len() / row_positions.len();
    let mut picks = Vec::with_capacity(row_positions.len());
    for (row_index, row_position) in row_positions.iter().enumerate() {
        let row_range = row_index * kept_per_row..(row_index + 1) * kept_per_row;
        let distribution = TargetDistribution {
            token_ids: &kept_ids[row_range.clone()],
            probabilities: &kept_probabilities[row_range],
        };
        // A row's pick takes the position after the row's own.
        let pick = keyed_sample(keyed_seed, *row_position as u64 + 1, &distribution).ok_or(
            TextExecutionError::InvalidInput {
                description: "a verified row's sampling distribution kept no token",
            },
        )?;
        picks.push(pick);
    }
    Ok(picks)
}

fn sampling_failure(sampling_error: crate::InferenceEngineError) -> TextExecutionError {
    TextExecutionError::from(runtime_integration::MlxRuntimeError::RuntimeOperation {
        operation: "read the target's sampling distribution for a draft tree",
        description: sampling_error.to_string(),
    })
}
