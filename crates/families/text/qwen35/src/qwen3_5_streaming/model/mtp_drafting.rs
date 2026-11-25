//! Drafting reply tokens with Flash-Next's multi-token-prediction head and
//! checking them on the streamed target in one forward.
//!
//! A plain reply token costs one target forward: every layer is built once
//! and its missing experts are read from storage. A round here builds every
//! layer once for several tokens: the head drafts the tokens after the one
//! the target last chose, one target forward runs that token and the drafts
//! together, and the reply keeps each draft for as long as it is the token
//! the target itself chooses at that position. The reply is therefore
//! exactly the target's own; a refused draft only costs the storage reads
//! its row caused.
//!
//! The head sees every reply token with the target's real lanes: after a
//! round its rows are cut back to where the round began and the kept tokens
//! are run again with the lanes the verify forward produced.

use engine_core::speculative_decoding::keyed_sampling::keyed_sample;
use engine_core::speculative_decoding::verification_decision::TargetDistribution;
use mlx_bridge::{MlxArray, MlxDtype};
use text_generation::execution_error::TextExecutionError;

use crate::gpu_token_sampling::sampling_distribution_rows;

use crate::{PerformanceAttribution, RequestDecoderStateStack};

use super::Qwen3_5Model;
use super::mtp_head::MtpHeadState;

/// How the target chooses the reply's token after each verified row: the
/// request's own sampling, so a drafted reply follows the same law as an
/// undrafted one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetTokenChoice {
    /// `None` takes the highest-scoring token.
    pub sampling: Option<TopKTopPSampling>,
    /// Keys every sampled position, so a position's choice does not depend
    /// on how many rows were verified with it.
    pub keyed_seed: u64,
}

/// Top-k, then top-p, then temperature, as the reply path samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopKTopPSampling {
    pub temperature_thousandths: u16,
    pub top_k: u16,
    pub top_p_thousandths: u16,
}

impl TargetTokenChoice {
    /// The choice for the highest-scoring token.
    #[must_use]
    pub const fn highest_scoring() -> Self {
        Self {
            sampling: None,
            keyed_seed: 0,
        }
    }
}

/// The head's first guess for the next round and the lanes it continues from.
struct NextDraft {
    /// `[1, 1, vocabulary]`, lazy: the head's logits for the token after
    /// the one the target last chose.
    head_logits: MlxArray,
    /// `[1, 1, lanes * hidden]`: the head's lanes at that position.
    head_lanes: MlxArray,
}

/// One reply's drafting state.
pub struct MtpDraftingSession {
    head_state: MtpHeadState,
    next_draft: Option<NextDraft>,
    /// Drafts offered and drafts the target agreed with, for reporting.
    pub drafted_token_count: u64,
    pub accepted_draft_count: u64,
    pub round_count: u64,
}

/// Where one round starts.
#[derive(Clone, Copy, Debug)]
pub struct MtpRoundStart {
    /// The token the target last chose, which has not run through it yet.
    pub last_chosen_token_id: u32,
    /// That token's position.
    pub starting_position_tokens: u32,
    /// The most tokens the head drafts after it.
    pub maximum_draft_count: usize,
}

/// What one round adds to the reply.
pub struct MtpRoundOutcome {
    /// The tokens the round produced, in order: the accepted drafts, then
    /// the target's own choice after them. The last one has not been run
    /// through the target yet and starts the next round.
    pub reply_token_ids: Vec<u32>,
    /// Tokens the head drafted this round and how many the target kept.
    pub drafted_token_count: usize,
    pub accepted_draft_count: usize,
}

impl Qwen3_5Model {
    /// Starts drafting for a reply: runs the head over the prompt with the
    /// target's lanes (`prompt_lanes`, `[1, prompt tokens, lanes * hidden]`)
    /// so its attention has seen what the target has. `first_reply_token_id`
    /// is the token the target chose after the prompt.
    pub fn begin_mtp_drafting(
        &self,
        prompt_token_ids: &[u32],
        prompt_lanes: &MlxArray,
        first_reply_token_id: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MtpDraftingSession, TextExecutionError> {
        if prompt_token_ids.is_empty() {
            return Err(TextExecutionError::InvalidInput {
                description: "drafting starts after a prompt of at least one token",
            });
        }
        let mut session = MtpDraftingSession {
            head_state: MtpHeadState::empty(self.chunking.full_attention_key_value_growth_tokens)?,
            next_draft: None,
            drafted_token_count: 0,
            accepted_draft_count: 0,
            round_count: 0,
        };
        // Each prompt token's lanes pair with the token after it.
        let mut following_token_ids = prompt_token_ids[1..].to_vec();
        following_token_ids.push(first_reply_token_id);
        self.run_head_over_kept_tokens(
            &mut session,
            &following_token_ids,
            prompt_lanes,
            performance_attribution,
        )?;
        Ok(session)
    }

    /// One round: draft the tokens after the last chosen one, verify them
    /// on the target, and keep what the target agrees with.
    pub fn advance_mtp_drafting_round(
        &self,
        session: &mut MtpDraftingSession,
        round_start: MtpRoundStart,
        request_decoder_state: &mut RequestDecoderStateStack,
        target_token_choice: TargetTokenChoice,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MtpRoundOutcome, TextExecutionError> {
        let head_row_count_before_round = session.head_state.row_count();
        let draft_token_ids = self.draft_reply_tokens(
            session,
            round_start,
            target_token_choice,
            performance_attribution,
        )?;
        let mut verified_token_ids = Vec::with_capacity(draft_token_ids.len() + 1);
        verified_token_ids.push(round_start.last_chosen_token_id);
        verified_token_ids.extend_from_slice(&draft_token_ids);
        let verification = self.verify_drafted_rows(
            &verified_token_ids,
            round_start.starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )?;
        // The target's own choice after each row; the reply keeps the
        // drafts that match it and then the first choice that does not.
        let target_token_ids = self.target_token_picks(
            &verification.per_row_logits,
            round_start.starting_position_tokens,
            target_token_choice,
        )?;
        let mut accepted_draft_count = 0;
        while accepted_draft_count < draft_token_ids.len()
            && target_token_ids[accepted_draft_count] == draft_token_ids[accepted_draft_count]
        {
            accepted_draft_count += 1;
        }
        let reply_token_ids = target_token_ids[..=accepted_draft_count].to_vec();
        session.round_count += 1;
        session.drafted_token_count += draft_token_ids.len() as u64;
        session.accepted_draft_count += accepted_draft_count as u64;
        let kept_row_count = accepted_draft_count + 1;
        let kept_lanes = self.runtime.slice(
            &verification.lanes,
            &[0, 0, 0],
            &[1, kept_row_count as i32, verification.lanes.shape()[2]],
            &[1, 1, 1],
        )?;
        self.keep_drafted_rows(verification, kept_row_count, request_decoder_state)?;
        // The head's own draft rows give way to the kept tokens with the
        // target's lanes.
        session.head_state.truncate_rows(
            head_row_count_before_round,
            self.config
                .qwen4_exp_architecture()
                .map_or(1, |architecture| architecture.indexer_compress_ratio as i32),
        )?;
        self.run_head_over_kept_tokens(
            session,
            &reply_token_ids,
            &kept_lanes,
            performance_attribution,
        )?;
        Ok(MtpRoundOutcome {
            reply_token_ids,
            drafted_token_count: draft_token_ids.len(),
            accepted_draft_count,
        })
    }

    /// The target's choice after every row of `per_row_logits`
    /// (`[1, rows, vocabulary]`), the first row being at
    /// `first_row_position`. A row's choice takes the position after its own.
    pub fn target_token_picks(
        &self,
        per_row_logits: &MlxArray,
        first_row_position: u32,
        target_token_choice: TargetTokenChoice,
    ) -> Result<Vec<u32>, TextExecutionError> {
        let runtime = &self.runtime;
        let Some(sampling) = target_token_choice.sampling else {
            let highest_scoring =
                runtime.astype(&runtime.argmax_axis(per_row_logits, -1)?, MlxDtype::UInt32)?;
            runtime.evaluate_arrays(&[&highest_scoring])?;
            return Ok(highest_scoring.to_vec_u32()?);
        };
        let (kept_token_ids, kept_probabilities) = sampling_distribution_rows(
            runtime,
            per_row_logits,
            sampling.temperature_thousandths,
            sampling.top_p_thousandths,
            sampling.top_k,
        )
        .map_err(|sampling_error| {
            TextExecutionError::from(runtime_integration::MlxRuntimeError::RuntimeOperation {
                operation: "read the target's sampling distribution for drafted rows",
                description: sampling_error.reason,
            })
        })?;
        runtime.evaluate_arrays(&[&kept_token_ids, &kept_probabilities])?;
        let kept_token_ids = kept_token_ids.to_vec_u32()?;
        let kept_probabilities = runtime.array_to_vec_f32(&kept_probabilities)?;
        let row_count = per_row_logits.shape()[1] as usize;
        let kept_per_row = kept_token_ids.len() / row_count.max(1);
        let mut picks = Vec::with_capacity(row_count);
        for row_index in 0..row_count {
            let row_range = row_index * kept_per_row..(row_index + 1) * kept_per_row;
            let row_distribution = TargetDistribution {
                token_ids: &kept_token_ids[row_range.clone()],
                probabilities: &kept_probabilities[row_range],
            };
            let sampled_position = u64::from(first_row_position) + row_index as u64 + 1;
            picks.push(
                keyed_sample(
                    target_token_choice.keyed_seed,
                    sampled_position,
                    &row_distribution,
                )
                .ok_or(TextExecutionError::InvalidInput {
                    description: "a verified row's sampling distribution kept no token",
                })?,
            );
        }
        Ok(picks)
    }

    /// `[1, 1, vocabulary]`: the target's logits after one verified row.
    fn verified_row_logits(
        &self,
        per_row_logits: &MlxArray,
        row_index: usize,
    ) -> Result<MlxArray, TextExecutionError> {
        let row_index = row_index as i32;
        Ok(self.runtime.slice(
            per_row_logits,
            &[0, row_index, 0],
            &[1, row_index + 1, per_row_logits.shape()[2]],
            &[1, 1, 1],
        )?)
    }

    /// Runs the head over tokens the reply keeps, each with the target's
    /// lanes at the position before it, and leaves its guess for the token
    /// after the last one as the next round's first draft.
    fn run_head_over_kept_tokens(
        &self,
        session: &mut MtpDraftingSession,
        kept_token_ids: &[u32],
        preceding_target_lanes: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        let head = self.mtp_head().ok_or(TextExecutionError::InvalidInput {
            description: "this checkpoint ships no multi-token-prediction head",
        })?;
        let token_count = kept_token_ids.len() as i32;
        let signed_token_ids: Vec<i32> = kept_token_ids
            .iter()
            .map(|token_id| *token_id as i32)
            .collect();
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_count])?;
        let head_output = self.forward_mtp_head(
            head,
            &token_indices,
            preceding_target_lanes,
            &mut session.head_state,
            performance_attribution,
        )?;
        let lane_width = head_output.lanes.shape()[2];
        let last_row_lanes = self.runtime.slice(
            &head_output.lanes,
            &[0, token_count - 1, 0],
            &[1, token_count, lane_width],
            &[1, 1, 1],
        )?;
        let last_row_logits =
            self.verified_row_logits(&head_output.logits, kept_token_ids.len() - 1)?;
        session.next_draft = Some(NextDraft {
            head_logits: last_row_logits,
            head_lanes: last_row_lanes,
        });
        Ok(())
    }

    /// The head's drafts for this round. Each draft is chosen from the
    /// head's logits the way the target will choose at the same position:
    /// the highest-scoring token, or the request's sampling with the same
    /// keyed noise. Sharing the noise is what makes a sampled draft agree
    /// with the target whenever the two distributions are close; drafting
    /// the head's top token instead was kept only 52% of the time at
    /// temperature 1.0. Each further draft feeds the head its previous draft
    /// and its own lanes.
    fn draft_reply_tokens(
        &self,
        session: &mut MtpDraftingSession,
        round_start: MtpRoundStart,
        target_token_choice: TargetTokenChoice,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Vec<u32>, TextExecutionError> {
        let head = self.mtp_head().ok_or(TextExecutionError::InvalidInput {
            description: "this checkpoint ships no multi-token-prediction head",
        })?;
        let next_draft = session
            .next_draft
            .take()
            .ok_or(TextExecutionError::InvalidInput {
                description: "a drafting round needs the head's guess from the round before",
            })?;
        if round_start.maximum_draft_count == 0 {
            return Err(TextExecutionError::InvalidInput {
                description: "a drafting round drafts at least one token",
            });
        }
        let mut head_logits = next_draft.head_logits;
        let mut head_lanes = next_draft.head_lanes;
        let mut draft_token_ids = Vec::with_capacity(round_start.maximum_draft_count);
        loop {
            // The draft stands where the target's choice after the row
            // before it will stand, so both read the same keyed noise.
            let preceding_row_position =
                round_start.starting_position_tokens + draft_token_ids.len() as u32;
            let draft_token_id =
                self.target_token_picks(&head_logits, preceding_row_position, target_token_choice)?
                    [0];
            draft_token_ids.push(draft_token_id);
            if draft_token_ids.len() == round_start.maximum_draft_count {
                return Ok(draft_token_ids);
            }
            let draft_token_index = self
                .runtime
                .array_from_i32(&[draft_token_id as i32], &[1, 1])?;
            let head_output = self.forward_mtp_head(
                head,
                &draft_token_index,
                &head_lanes,
                &mut session.head_state,
                performance_attribution,
            )?;
            head_logits = head_output.logits;
            head_lanes = head_output.lanes;
        }
    }
}
