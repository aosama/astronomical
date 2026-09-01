//! Sampled multi-token-prediction proposal and `min(1, p/q)` verification.
//!
//! The greedy verifier accepts only when a drafted token equals the target's top
//! token, which is distribution-correct only at temperature 0. This module owns
//! the sampled counterpart:
//!
//! 1. Drafts are sampled from the head distribution through the same shared
//!    temperature + top-k + top-p pipeline the target sampler uses.
//! 2. Each drafted token is accepted with probability `min(1, p/q)`, where `p`
//!    is the target's masked sampling distribution and `q` the draft's.
//! 3. On rejection the emitted token is drawn from mass proportional to
//!    `max(0, p − q)`; on a fully accepted window the bonus token is drawn from
//!    the target distribution. Both keep every emitted token distributed exactly
//!    as sampling the target alone at the request's sampler settings.
//!
//! The pure coin-to-decision math lives in `verification_decision.rs`; this
//! module owns only GPU graph construction and synchronization. Decode-state
//! checkpoints, rollback, and prefix commit are shared with the greedy mode
//! through `accepted_prefix_commit.rs`.

use astronomical_runtime_integration::MlxArray;

use crate::gpu_token_sampling::{
    masked_logits_after_top_k_and_top_p, sample_acceptance_coins,
    sample_categorical_after_temperature, sample_from_relative_probabilities,
};
use crate::qwen3_5::decoder::{
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint, RequestDecoderStateStack,
};
use crate::qwen3_5::model::{
    Qwen3_5ExecutionError, Qwen3_5Model, Qwen3_5TargetForwardOutput, forward_state_arrays,
};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::request_state::Qwen3_5MtpRequestState;
use super::target_verification::{
    MtpVerificationWindow, completed_verification_prefix_boundaries,
    forward_mtp_verification_window_with_performance_attribution,
};
use crate::memory::MtpDraftDepth;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::qwen3_5) struct MtpSampledSamplingSettings {
    pub(in crate::qwen3_5) temperature_thousandths: u16,
    pub(in crate::qwen3_5) top_k: u16,
    pub(in crate::qwen3_5) top_p_thousandths: u16,
}

impl MtpSampledSamplingSettings {
    fn temperature(&self) -> f32 {
        f32::from(self.temperature_thousandths) / 1_000.0
    }
}

/// Sampled drafts plus the masked draft distribution that produced each one.
///
/// Verification needs `q` as a full distribution for the residual correction, so
/// each probability row stays resident until the shared window evaluation.
pub(in crate::qwen3_5) struct MtpSampledProposalChain {
    pub(crate) draft_token_ids: Vec<u32>,
    draft_probability_rows: Vec<MlxArray>,
}

/// Sampled accept outcomes plus the forward data the shared commit path consumes.
pub(in crate::qwen3_5) struct SampledMtpVerificationOutput {
    pub(crate) accepted_coin_flags: Vec<bool>,
    pub(crate) post_prefix_token_id: u32,
    pub(crate) target_forward_output: Qwen3_5TargetForwardOutput,
    pub(crate) prefix_boundaries: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

impl Qwen3_5Model {
    /// Reuses the stored MTP layer while sampling every draft depth.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::qwen3_5) fn propose_sampled_mtp_chain_with_performance_attribution(
        &self,
        target_hidden_seed: &MlxArray,
        current_token_indices: &MlxArray,
        effective_depth: MtpDraftDepth,
        mtp_request_state: &mut Qwen3_5MtpRequestState,
        sampling_settings: MtpSampledSamplingSettings,
        random_state: &mut MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MtpSampledProposalChain, Qwen3_5ExecutionError> {
        let draft_count = usize::from(effective_depth.get());
        let mut draft_token_arrays = Vec::with_capacity(draft_count);
        let mut post_normalization_hidden_rows = Vec::with_capacity(draft_count);
        let mut draft_probability_rows = Vec::with_capacity(draft_count);
        let vocabulary_size_i32 = self.config().vocabulary_size() as i32;
        performance_attribution.measure_operation(
            PerformanceOperation::MtpHeadForwardGraphConstruction,
            |performance_attribution| {
                for draft_position in 0..effective_depth.get() {
                    let hidden_states_for_fusion = if draft_position == 0 {
                        target_hidden_seed
                    } else {
                        post_normalization_hidden_rows.last().ok_or(
                            Qwen3_5ExecutionError::InvalidInput {
                                description: "MTP proposal chain lost its prior hidden row",
                            },
                        )?
                    };
                    let token_indices_for_fusion = if draft_position == 0 {
                        current_token_indices
                    } else {
                        draft_token_arrays
                            .last()
                            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                                description: "MTP proposal chain lost its prior draft token",
                            })?
                    };
                    let forward_output = self.build_mtp_draft_graph(
                        hidden_states_for_fusion,
                        token_indices_for_fusion,
                        mtp_request_state,
                        performance_attribution,
                    )?;
                    let (draft_logits_array, post_normalization_hidden_states) =
                        forward_output.into_arrays();
                    // The masked distribution is what the shared sampler draws
                    // from; the sampled token and its draft probability both
                    // derive from it so verification compares like with like.
                    let masked_draft_logits = masked_logits_after_top_k_and_top_p(
                        self.runtime(),
                        &draft_logits_array,
                        vocabulary_size_i32,
                        sampling_settings.top_k,
                        sampling_settings.top_p_thousandths,
                    )?;
                    let sampled_draft_token = sample_categorical_after_temperature(
                        self.runtime(),
                        &masked_draft_logits,
                        sampling_settings.temperature_thousandths,
                        random_state,
                    )?;
                    let scaled_masked_draft_logits = self
                        .runtime()
                        .multiply_scalar(
                            &masked_draft_logits,
                            sampling_settings.temperature().recip(),
                        )
                        .map_err(Qwen3_5ExecutionError::from)?;
                    let draft_probability_row = self
                        .runtime()
                        .softmax_axis(&scaled_masked_draft_logits, -1)
                        .map_err(Qwen3_5ExecutionError::from)?;
                    draft_token_arrays.push(sampled_draft_token);
                    post_normalization_hidden_rows.push(post_normalization_hidden_states);
                    draft_probability_rows.push(draft_probability_row);
                }
                Ok::<(), Qwen3_5ExecutionError>(())
            },
        )?;
        let draft_token_references = draft_token_arrays.iter().collect::<Vec<_>>();
        let draft_token_vector = self
            .runtime()
            .concatenate_axis(&draft_token_references, 1)?;
        let mut evaluation_roots = Vec::with_capacity(
            post_normalization_hidden_rows.len() + draft_probability_rows.len() + 1,
        );
        evaluation_roots.push(&draft_token_vector);
        evaluation_roots.extend(post_normalization_hidden_rows.iter());
        evaluation_roots.extend(draft_probability_rows.iter());
        self.evaluate_mtp_updated_state(
            mtp_request_state,
            &evaluation_roots,
            performance_attribution,
        )?;
        let draft_token_ids = draft_token_vector.to_vec_u32()?;
        Ok(MtpSampledProposalChain {
            draft_token_ids,
            draft_probability_rows,
        })
    }

    /// Verifies one sampled draft window with per-position acceptance coins.
    ///
    /// The window forward, boundary capture, and state evaluation are shared with
    /// the greedy verifier; only the decision inputs differ. One synchronization
    /// covers the accept coins (which transitively evaluate the target forward
    /// graph), the residual rows, the decoder state, and the boundary tensors.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::qwen3_5) fn verify_sampled_mtp_window_with_performance_attribution(
        &self,
        verifier_input_token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        proposal: MtpSampledProposalChain,
        sampling_settings: MtpSampledSamplingSettings,
        force_first_rejection_for_tests: bool,
        random_state: &mut MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<SampledMtpVerificationOutput, Qwen3_5ExecutionError> {
        let MtpVerificationWindow {
            target_forward_output,
            boundary_collector,
            completed_verifier_prefix_rows,
        } = forward_mtp_verification_window_with_performance_attribution(
            self,
            verifier_input_token_ids,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )?;
        let all_position_logits = target_forward_output.all_position_logits().ok_or(
            Qwen3_5ExecutionError::InvalidInput {
                description: "target verification forward did not retain all-position logits",
            },
        )?;
        let draft_count = proposal.draft_token_ids.len();
        let draft_count_i32 = draft_count as i32;
        let vocabulary_size = self.config().vocabulary_size() as usize;
        let vocabulary_size_i32 = vocabulary_size as i32;
        // Target sampling distributions for the verifier rows. Rows 0..k verify
        // the drafted tokens; row k is the bonus distribution.
        let masked_target_logits = masked_logits_after_top_k_and_top_p(
            self.runtime(),
            all_position_logits,
            vocabulary_size_i32,
            sampling_settings.top_k,
            sampling_settings.top_p_thousandths,
        )?;
        let scaled_masked_target_logits = self
            .runtime()
            .multiply_scalar(
                &masked_target_logits,
                sampling_settings.temperature().recip(),
            )
            .map_err(Qwen3_5ExecutionError::from)?;
        let target_probabilities = self
            .runtime()
            .softmax_axis(&scaled_masked_target_logits, -1)
            .map_err(Qwen3_5ExecutionError::from)?;
        let draft_probabilities = self.runtime().concatenate_axis(
            &proposal.draft_probability_rows.iter().collect::<Vec<_>>(),
            1,
        )?;
        let target_verify_rows = self.runtime().slice(
            &target_probabilities,
            &[0, 0, 0],
            &[1, draft_count_i32, vocabulary_size_i32],
            &[1, 1, 1],
        )?;
        let draft_token_index_row = self
            .runtime()
            .array_from_u32(&proposal.draft_token_ids, &[1, 1, draft_count_i32])?;
        let target_probabilities_at_drafts =
            self.runtime()
                .take_along_axis(&target_verify_rows, &draft_token_index_row, -1)?;
        let draft_probabilities_at_drafts =
            self.runtime()
                .take_along_axis(&draft_probabilities, &draft_token_index_row, -1)?;
        // Acceptance probability min(1, p/q); a zero draft probability is an
        // automatic rejection, mirroring the pure verifier decision boundary.
        let one_scalar = self.runtime().array_from_f32(&[1.0], &[])?;
        let zero_scalar = self.runtime().array_from_f32(&[0.0], &[])?;
        let raw_ratio = self
            .runtime()
            .divide(
                &target_probabilities_at_drafts,
                &draft_probabilities_at_drafts,
            )
            .map_err(Qwen3_5ExecutionError::from)?;
        let ratio_above_one = self.runtime().greater(&raw_ratio, &one_scalar)?;
        let capped_ratio =
            self.runtime()
                .where_select(&ratio_above_one, &one_scalar, &raw_ratio)?;
        let draft_mass_is_positive = self
            .runtime()
            .greater(&draft_probabilities_at_drafts, &zero_scalar)?;
        let acceptance_probabilities =
            self.runtime()
                .where_select(&draft_mass_is_positive, &capped_ratio, &zero_scalar)?;
        // Residual rows: mass proportional to max(0, p − q) per verifier row.
        let probability_surplus = self
            .runtime()
            .subtract(&target_verify_rows, &draft_probabilities)
            .map_err(Qwen3_5ExecutionError::from)?;
        let surplus_is_positive = self.runtime().greater(&probability_surplus, &zero_scalar)?;
        let zero_probability_rows = self
            .runtime()
            .broadcast_to(&zero_scalar, &[1, draft_count_i32, vocabulary_size_i32])?;
        let residual_rows = self.runtime().where_select(
            &surplus_is_positive,
            &probability_surplus,
            &zero_probability_rows,
        )?;
        let residual_mass = self.runtime().sum_axis(&residual_rows, -1, true)?;
        let coin_outcomes = sample_acceptance_coins(
            self.runtime(),
            &acceptance_probabilities,
            random_state,
            draft_count,
        )?;
        let (coin_values, residual_mass_values) = performance_attribution.measure_operation(
            PerformanceOperation::MtpTargetVerificationSynchronizationWait,
            |_performance_attribution| -> Result<(Vec<u32>, Vec<f32>), Qwen3_5ExecutionError> {
                let mut evaluation_roots =
                    forward_state_arrays(&coin_outcomes, request_decoder_state)?;
                evaluation_roots
                    .push(target_forward_output.pre_final_normalization_hidden_states());
                if let Some(boundary_collector) = boundary_collector.as_ref() {
                    evaluation_roots.extend(boundary_collector.evaluation_arrays());
                }
                evaluation_roots.push(&residual_mass);
                self.runtime().evaluate_arrays(&evaluation_roots)?;
                Ok((coin_outcomes.to_vec_u32()?, residual_mass.to_vec_f32()?))
            },
        )?;
        let mut accepted_coin_flags: Vec<bool> = coin_values
            .iter()
            .map(|coin_value| *coin_value == 1)
            .collect();
        if force_first_rejection_for_tests && !accepted_coin_flags.is_empty() {
            accepted_coin_flags[0] = false;
        }
        let post_prefix_token_id = match accepted_coin_flags.iter().position(|accepted| !accepted) {
            Some(rejection_position) => {
                if residual_mass_values
                    .get(rejection_position)
                    .is_some_and(|residual_mass| *residual_mass > 0.0)
                {
                    let residual_row = self.runtime().slice(
                        &residual_rows,
                        &[0, rejection_position as i32, 0],
                        &[1, rejection_position as i32 + 1, vocabulary_size_i32],
                        &[1, 1, 1],
                    )?;
                    sample_from_relative_probabilities(self.runtime(), &residual_row, random_state)?
                        .item_u32()
                        .map_err(Qwen3_5ExecutionError::from)?
                } else {
                    // The residual is numerically empty only when the draft and
                    // target distributions coincide; the emitted token then comes
                    // from the unchanged target distribution.
                    let unchanged_target_row = self.runtime().slice(
                        &masked_target_logits,
                        &[0, rejection_position as i32, 0],
                        &[1, rejection_position as i32 + 1, vocabulary_size_i32],
                        &[1, 1, 1],
                    )?;
                    sample_categorical_after_temperature(
                        self.runtime(),
                        &unchanged_target_row,
                        sampling_settings.temperature_thousandths,
                        random_state,
                    )?
                    .item_u32()
                    .map_err(Qwen3_5ExecutionError::from)?
                }
            }
            None => {
                let masked_bonus_row = self.runtime().slice(
                    &masked_target_logits,
                    &[0, draft_count_i32, 0],
                    &[1, draft_count_i32 + 1, vocabulary_size_i32],
                    &[1, 1, 1],
                )?;
                sample_categorical_after_temperature(
                    self.runtime(),
                    &masked_bonus_row,
                    sampling_settings.temperature_thousandths,
                    random_state,
                )?
                .item_u32()
                .map_err(Qwen3_5ExecutionError::from)?
            }
        };
        let prefix_boundaries = completed_verification_prefix_boundaries(
            boundary_collector,
            &completed_verifier_prefix_rows,
            verifier_input_token_ids,
        )?;
        Ok(SampledMtpVerificationOutput {
            accepted_coin_flags,
            post_prefix_token_id,
            target_forward_output,
            prefix_boundaries,
        })
    }
}
