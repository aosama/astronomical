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

use mlx_bridge::{MlxArray, MlxDtype};
use text_generation::execution_error::TextExecutionError;

use crate::{PerformanceAttribution, RequestDecoderStateStack};

use super::Qwen3_5Model;
use super::mtp_head::MtpHeadState;

/// Chooses the reply's next token from the target's logits at one position,
/// with whatever sampling the request asked for. Called once per reply
/// token, in order, so a sampler's own state advances exactly as it would
/// without drafting.
pub trait ReplyTokenChooser {
    /// `position_logits` is `[1, 1, vocabulary]` float32.
    fn choose_reply_token(&mut self, position_logits: &MlxArray)
    -> Result<u32, TextExecutionError>;
}

/// The head's first guess for the next round and the lanes it continues from.
struct NextDraft {
    /// `[1, 1]` int32, lazy: the head's choice for the token after the one
    /// the target last chose.
    first_draft_token: MlxArray,
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
        token_chooser: &mut dyn ReplyTokenChooser,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MtpRoundOutcome, TextExecutionError> {
        let head_row_count_before_round = session.head_state.row_count();
        let draft_token_ids = self.draft_reply_tokens(
            session,
            round_start.maximum_draft_count,
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
        // The target's own choice after each row, until one differs from
        // the draft that was run in its place.
        let mut reply_token_ids = Vec::with_capacity(verified_token_ids.len());
        let mut accepted_draft_count = 0;
        for (row_index, draft_token_id) in draft_token_ids.iter().enumerate() {
            let target_token_id = token_chooser.choose_reply_token(
                &self.verified_row_logits(&verification.per_row_logits, row_index)?,
            )?;
            reply_token_ids.push(target_token_id);
            if target_token_id != *draft_token_id {
                break;
            }
            accepted_draft_count += 1;
        }
        if accepted_draft_count == draft_token_ids.len() {
            reply_token_ids.push(token_chooser.choose_reply_token(
                &self.verified_row_logits(&verification.per_row_logits, accepted_draft_count)?,
            )?);
        }
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
        Ok(MtpRoundOutcome { reply_token_ids })
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
            first_draft_token: self.highest_scoring_token(&last_row_logits)?,
            head_lanes: last_row_lanes,
        });
        Ok(())
    }

    /// `[1, 1]` int32: the highest-scoring token of `[1, 1, vocabulary]` logits.
    fn highest_scoring_token(
        &self,
        position_logits: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        Ok(self.runtime.astype(
            &self.runtime.argmax_axis(position_logits, -1)?,
            MlxDtype::Int32,
        )?)
    }

    /// The head's drafts for this round, read to the host in one wait. The
    /// first comes from the last round; each further one feeds the head its
    /// own previous guess and lanes.
    fn draft_reply_tokens(
        &self,
        session: &mut MtpDraftingSession,
        maximum_draft_count: usize,
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
        if maximum_draft_count == 0 {
            return Err(TextExecutionError::InvalidInput {
                description: "a drafting round drafts at least one token",
            });
        }
        let mut draft_tokens = vec![next_draft.first_draft_token];
        let mut head_lanes = next_draft.head_lanes;
        while draft_tokens.len() < maximum_draft_count {
            let previous_draft = &draft_tokens[draft_tokens.len() - 1];
            let head_output = self.forward_mtp_head(
                head,
                previous_draft,
                &head_lanes,
                &mut session.head_state,
                performance_attribution,
            )?;
            draft_tokens.push(self.highest_scoring_token(&head_output.logits)?);
            head_lanes = head_output.lanes;
        }
        let draft_token_references: Vec<&MlxArray> = draft_tokens.iter().collect();
        let joined_drafts = self.runtime.concatenate_axis(&draft_token_references, 1)?;
        let mut evaluation_arrays = vec![&joined_drafts];
        session
            .head_state
            .append_evaluation_arrays(&mut evaluation_arrays);
        self.runtime.evaluate_arrays(&evaluation_arrays)?;
        let mut draft_token_ids = Vec::with_capacity(draft_tokens.len());
        for draft_token_id in joined_drafts.to_vec_i32()? {
            draft_token_ids.push(u32::try_from(draft_token_id).map_err(|_| {
                TextExecutionError::InvalidInput {
                    description: "the head drafted a negative token id",
                }
            })?);
        }
        Ok(draft_token_ids)
    }
}
