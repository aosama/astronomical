//! Cross-layer speculative gate (issue #593).
//!
//! Inside a transformer, each decoder layer's input is the previous layer's
//! output plus a residual. Because the residual stream moves slowly between
//! adjacent layers, the input that layer N+1's router gate is *about to* see
//! is close to the input layer N's gate just saw. Evaluating layer N+1's own
//! native gate weights on layer N's router input therefore predicts, without
//! any training, which experts layer N+1 will route to — early enough to warm
//! their storage reads while layer N still occupies the graphics processor.
//!
//! The published evidence for this signal: FATE (arXiv 2502.12224) measures
//! 78.8 percent plain top-k coverage and 97 percent with widened candidate
//! sets, training-free; YALIS (arXiv 2603.19289) sustains 83-90 percent hit
//! rates across modern MoEs from the same quasi-hidden-state signal.
//!
//! Boundaries this module keeps:
//! - The native router stays the only execution authority. Predictions change
//!   when storage pages are read, never which experts compute.
//! - The speculative gate runs as extra lazy graph nodes joined to the route
//!   evaluation the decode path already waits on, so prediction adds no new
//!   synchronization wait.
//! - Every failure is fail-open: a missing layer, a dense successor, or a
//!   copy error simply skips prediction for that token.

use std::path::PathBuf;

use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::expert_paging::quantized_expert_manifest::build_quantized_expert_page_manifest_from_plan;
use crate::expert_paging::streaming_expert_pack_pages::build_streaming_expert_page_manifest;
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5DecoderFeedForwardWeights;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5ExpertPager;
use crate::qwen3_5_moe::expert_paging::speculative_page_warmer::SpeculativeWarmRequest;
use crate::qwen3_5_moe::model::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, PerformanceCounter};

/// Extra candidates kept beyond the strict top-k. FATE measured that near-miss
/// experts rank just below the routing cut, so widening the candidate set
/// trades cheap background storage reads for a large coverage gain.
const SPECULATIVE_CANDIDATE_WIDENING_MULTIPLIER: usize = 2;

/// One layer's speculative prediction, produced during the previous layer's
/// route evaluation.
#[derive(Clone, Debug)]
pub(crate) struct SpeculativeRouteCandidates {
    pub next_layer_index: usize,
    /// Candidate expert IDs in descending gate-score order.
    pub expert_ids: Vec<usize>,
}

impl Qwen3_5Model {
    /// Builds lazy router logits for layer N+1 from layer N's router input.
    ///
    /// Returns `None` whenever prediction does not apply: non-decode execution,
    /// the trunk's last layer, a dense successor layer, or a model without the
    /// paged expert stores the warming would serve. The returned array is lazy
    /// and joins the caller's route evaluation.
    pub(crate) fn build_speculative_next_layer_router_logits(
        &self,
        hidden_states: &MlxArray,
        layer_index: usize,
        token_count: i32,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
    ) -> Result<Option<MlxArray>, Qwen3_5ExecutionError> {
        if token_count != 1
            || paged_prefill_execution_mode
                != Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault
            || self.expert_pager.is_none()
            || self.retained_experts.is_none()
        {
            return Ok(None);
        }
        let next_layer_index = layer_index + 1;
        if next_layer_index >= usize::try_from(self.config.layer_count()).unwrap_or(0) {
            return Ok(None);
        }
        let Some(next_decoder_layer_weights) =
            self.weights.decoder_layer_weights.get(next_layer_index)
        else {
            return Ok(None);
        };
        let Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(next_mixture_weights) =
            &next_decoder_layer_weights.mlp_weights
        else {
            return Ok(None);
        };
        let speculative_router_logits = match &next_mixture_weights.router_projection {
            crate::qwen3_5_moe::model::feed_forward_weights::Qwen3_5MoERouterGateWeights::Affine(
                quantized_weights,
            ) => self.quantized_linear_for_paged_prefill_execution_mode(
                hidden_states,
                quantized_weights,
                paged_prefill_execution_mode,
            )?,
            crate::qwen3_5_moe::model::feed_forward_weights::Qwen3_5MoERouterGateWeights::Unquantized(
                unquantized_weight,
            ) => {
                let transposed_gate_weight = self.runtime.transpose_axes(unquantized_weight, &[1, 0])?;
                self.runtime.matmul(hidden_states, &transposed_gate_weight)?
            }
        };
        // The host copy reads float32; gate projections may emit activation
        // dtype, so the prediction logits are narrowed explicitly.
        let speculative_router_logits = self
            .runtime
            .astype(&speculative_router_logits, MlxDtype::Float32)?;
        Ok(Some(speculative_router_logits))
    }

    /// Turns evaluated speculative logits into widened candidate expert IDs.
    ///
    /// The candidate count doubles the strict top-k: the extra candidates are
    /// the near-miss experts the widened prefetching evidence says convert a
    /// large share of routing misses into warm reads.
    #[must_use]
    pub(crate) fn select_speculative_candidate_expert_ids(
        router_logits: &[f32],
        experts_per_token: usize,
    ) -> Vec<usize> {
        let candidate_count = experts_per_token
            .saturating_mul(SPECULATIVE_CANDIDATE_WIDENING_MULTIPLIER)
            .min(router_logits.len());
        let mut ranked_expert_ids: Vec<usize> = (0..router_logits.len()).collect();
        ranked_expert_ids.sort_unstable_by(|left, right| {
            router_logits[*right]
                .partial_cmp(&router_logits[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked_expert_ids.truncate(candidate_count);
        ranked_expert_ids
    }

    /// Scores one layer's speculative prediction against the true route.
    ///
    /// A hit is a true routed expert that appeared anywhere in the candidate
    /// set. The denominator is the true routed expert count, so the metric
    /// answers "of the experts the router demanded, how many had we already
    /// prepared storage for".
    #[must_use]
    pub(crate) fn count_speculative_route_hits(
        predicted_expert_ids: &[usize],
        true_routed_expert_ids: &[usize],
    ) -> (u64, u64) {
        let evaluated_expert_count = true_routed_expert_ids.len() as u64;
        let hit_count = true_routed_expert_ids
            .iter()
            .filter(|true_expert_id| predicted_expert_ids.contains(true_expert_id))
            .count() as u64;
        (hit_count, evaluated_expert_count)
    }

    /// Copies the decode route and, when a speculative prediction exists,
    /// evaluates it in the SAME graphics-processor wait, then reads both to
    /// the host. One synchronization boundary serves route materialization
    /// and prediction alike.
    pub(crate) fn copy_selected_expert_ids_with_speculative_route(
        &self,
        selected_indices: &MlxArray,
        next_layer_index: usize,
        speculative_router_logits: Option<&MlxArray>,
    ) -> Result<(Vec<usize>, Option<SpeculativeRouteCandidates>), Qwen3_5ExecutionError> {
        let contiguous_ids = self
            .runtime
            .build_contiguous_row_major_copy(selected_indices)?;
        let Some(speculative_router_logits) = speculative_router_logits else {
            contiguous_ids.evaluate()?;
            let selected_expert_ids = contiguous_ids
                .copy_evaluated_u32_values()?
                .into_iter()
                .map(|expert_id| expert_id as usize)
                .collect();
            return Ok((selected_expert_ids, None));
        };
        self.runtime
            .evaluate_arrays(&[&contiguous_ids, speculative_router_logits])?;
        let selected_expert_ids = contiguous_ids
            .copy_evaluated_u32_values()?
            .into_iter()
            .map(|expert_id| expert_id as usize)
            .collect();
        let speculative_logits = speculative_router_logits.to_vec_f32()?;
        let candidate_expert_ids = Self::select_speculative_candidate_expert_ids(
            &speculative_logits,
            usize::try_from(self.config.experts_per_token()).unwrap_or(1),
        );
        Ok((
            selected_expert_ids,
            Some(SpeculativeRouteCandidates {
                next_layer_index,
                expert_ids: candidate_expert_ids,
            }),
        ))
    }

    /// Scores the true route of `layer_index` against the prediction the
    /// previous layer's speculative gate dispatched, then clears the slot.
    pub(crate) fn record_speculative_route_outcome(
        &self,
        layer_index: usize,
        true_routed_expert_ids: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let predicted_expert_ids = self
            .speculative_route_predictions_by_layer
            .borrow_mut()
            .get_mut(layer_index)
            .and_then(|prediction_slot| prediction_slot.take());
        let Some(predicted_expert_ids) = predicted_expert_ids else {
            return;
        };
        let (hit_count, evaluated_expert_count) =
            Self::count_speculative_route_hits(&predicted_expert_ids, true_routed_expert_ids);
        performance_attribution.record_counter(
            PerformanceCounter::ExpertRouteSpeculativeHitCount,
            hit_count,
        );
        performance_attribution.record_counter(
            PerformanceCounter::ExpertRouteSpeculativeEvaluatedExpertCount,
            evaluated_expert_count,
        );
    }

    /// Dispatches one layer's speculative candidates to the background
    /// warmer: already-warm experts are skipped, the rest's storage ranges
    /// are planned on the decode thread (pure planning, no I/O) and handed
    /// to the warmer thread.
    pub(crate) fn dispatch_speculative_page_warming(
        &self,
        speculative_candidates: &SpeculativeRouteCandidates,
        expert_pager: &Qwen3_5ExpertPager,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let next_layer_index = speculative_candidates.next_layer_index;
        if speculative_candidates.expert_ids.is_empty() {
            return;
        }
        // Record the full prediction before filtering, so the hit metric
        // measures what the gate predicted, not what survived the warm filter.
        if let Some(prediction_slot) = self
            .speculative_route_predictions_by_layer
            .borrow_mut()
            .get_mut(next_layer_index)
        {
            *prediction_slot = Some(speculative_candidates.expert_ids.clone());
        }
        let missing_expert_ids: Vec<usize> = {
            let Some(retained_experts) = self.retained_experts.as_ref() else {
                return;
            };
            let retained_experts = retained_experts.borrow();
            speculative_candidates
                .expert_ids
                .iter()
                .copied()
                .filter(|expert_id| !retained_experts.is_expert_warm(next_layer_index, *expert_id))
                .collect()
        };
        let skipped_candidate_count = speculative_candidates
            .expert_ids
            .len()
            .saturating_sub(missing_expert_ids.len());
        performance_attribution.record_counter(
            PerformanceCounter::ExpertRouteSpeculativeWarmSkipCount,
            u64::try_from(skipped_candidate_count).unwrap_or(u64::MAX),
        );
        if missing_expert_ids.is_empty() {
            return;
        }
        let warm_request = match self.plan_speculative_warm_ranges(
            expert_pager,
            next_layer_index,
            &missing_expert_ids,
        ) {
            Ok(warm_request) => warm_request,
            Err(planning_error) => {
                tracing::debug!(
                    error = %planning_error,
                    layer_index = next_layer_index,
                    "speculative warming planning failed; skipping this layer"
                );
                return;
            }
        };
        performance_attribution.record_counter(
            PerformanceCounter::ExpertRouteSpeculativeCandidateCount,
            u64::try_from(missing_expert_ids.len()).unwrap_or(u64::MAX),
        );
        performance_attribution.record_counter(
            PerformanceCounter::ExpertRouteSpeculativeWarmedByteCount,
            warm_request.warmed_byte_count,
        );
        let mut warmer_slot = self.speculative_page_warmer.borrow_mut();
        if warmer_slot.is_none() {
            *warmer_slot =
                crate::qwen3_5_moe::expert_paging::speculative_page_warmer::SpeculativePageWarmer::try_spawn();
        }
        if let Some(warmer) = warmer_slot.as_ref() {
            let dropped_request_count = warmer.take_dropped_request_count();
            if dropped_request_count > 0 {
                performance_attribution.record_counter(
                    PerformanceCounter::ExpertRouteSpeculativeWarmerDropCount,
                    dropped_request_count,
                );
            }
            let completed_request_count = warmer.completed_request_count();
            if completed_request_count > 0 {
                performance_attribution.record_counter(
                    PerformanceCounter::ExpertRouteSpeculativeWarmerCompletedCount,
                    completed_request_count,
                );
            }
            warmer.try_warm(warm_request.request);
        }
    }

    /// Plans the byte ranges one layer's candidates occupy in their storage
    /// files. Pure planning over startup-validated geometry: no model-payload
    /// I/O happens here.
    fn plan_speculative_warm_ranges(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        next_layer_index: usize,
        missing_expert_ids: &[usize],
    ) -> Result<PlannedSpeculativeWarmRanges, Qwen3_5ExecutionError> {
        let layer_plan = expert_pager.layer_plan(next_layer_index)?;
        let page_manifest = if let Some(expert_file_paths) =
            expert_pager.streaming_expert_file_paths(next_layer_index)
        {
            build_streaming_expert_page_manifest(
                layer_plan,
                expert_file_paths,
                missing_expert_ids,
            )
            .map_err(|planning_error| {
                tracing::debug!(
                    error = %planning_error,
                    layer_index = next_layer_index,
                    "speculative warm planning failed"
                );
                Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative warm planning failed for the streaming pack source",
                }
            })?
        } else {
            build_quantized_expert_page_manifest_from_plan(layer_plan, missing_expert_ids).map_err(
                |planning_error| {
                    tracing::debug!(
                        error = %planning_error,
                        layer_index = next_layer_index,
                        "speculative warm planning failed"
                    );
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "speculative warm planning failed for the shard source",
                    }
                },
            )?
        };
        let mut file_ranges: Vec<(PathBuf, Vec<(u64, usize)>)> = Vec::new();
        let mut warmed_byte_count = 0_u64;
        for source_manifest in &page_manifest.source_manifests {
            let range_list = source_manifest
                .source_intervals
                .iter()
                .map(|source_interval| {
                    warmed_byte_count +=
                        u64::try_from(source_interval.source_byte_count).unwrap_or(u64::MAX);
                    (
                        source_interval.source_file_offset,
                        source_interval.source_byte_count,
                    )
                })
                .collect::<Vec<(u64, usize)>>();
            file_ranges.push((source_manifest.source_file.clone(), range_list));
        }
        Ok(PlannedSpeculativeWarmRanges {
            request: SpeculativeWarmRequest { file_ranges },
            warmed_byte_count,
        })
    }
}

/// Planning outcome for one layer's speculative warming: the request for the
/// background thread plus the byte count for attribution.
struct PlannedSpeculativeWarmRanges {
    request: SpeculativeWarmRequest,
    warmed_byte_count: u64,
}

#[cfg(test)]
mod speculative_gate_tests {
    use super::*;

    #[test]
    fn candidate_selection_ranks_by_score_and_widens_past_the_strict_top_k() {
        let router_logits = vec![0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4];
        let candidates = Qwen3_5Model::select_speculative_candidate_expert_ids(&router_logits, 2);
        assert_eq!(
            candidates,
            vec![1, 6, 4, 2],
            "candidates must be the top four experts by descending gate score"
        );
    }

    #[test]
    fn candidate_selection_never_exceeds_the_expert_count() {
        let router_logits = vec![0.3, 0.1, 0.2];
        let candidates = Qwen3_5Model::select_speculative_candidate_expert_ids(&router_logits, 2);
        assert_eq!(
            candidates.len(),
            3,
            "doubling two candidates cannot exceed three experts"
        );
    }

    #[test]
    fn hit_counting_scores_true_experts_against_the_candidate_set() {
        let (hit_count, evaluated_expert_count) =
            Qwen3_5Model::count_speculative_route_hits(&[3, 7, 1], &[7, 1, 9]);
        assert_eq!(
            hit_count, 2,
            "experts 7 and 1 were predicted; expert 9 was not"
        );
        assert_eq!(evaluated_expert_count, 3);
    }

    #[test]
    fn hit_counting_reports_zero_when_no_prediction_exists() {
        let (hit_count, evaluated_expert_count) =
            Qwen3_5Model::count_speculative_route_hits(&[], &[7, 1, 9]);
        assert_eq!(hit_count, 0);
        assert_eq!(evaluated_expert_count, 3);
    }
}
