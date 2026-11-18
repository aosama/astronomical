//! Hosting a DFlash2 drafter beside a streaming model, and deciding which
//! sessions draft with it.
//!
//! A model that streams from storage decodes in draft rounds too: the round
//! is the shared core's, and its target pass finds each row's experts through
//! the `astro_expert_cache` and the pager. The drafter itself never pages, so
//! what it holds is counted with the model's core weights, and the memory
//! governor plans expert capacity around it.

use crate::PerformanceAttribution;
use crate::qwen3_5_core::draft_tree_decode::engine_draft_runtime::{
    DraftTargetProjections, Qwen3_5SpeculativeDraftRuntime,
};
use crate::qwen3_5_core::draft_tree_decode::speculation_session_admission::{
    SpeculationSessionRequestProperties, should_admit_speculation_for_session,
    verification_token_choice_for_strategy,
};
use crate::qwen3_5_core::text::Qwen3_5InferenceRequest;
use crate::qwen3_5_streaming::model::Qwen3_5Model;

use super::Qwen3_5InferenceExecution;

impl Qwen3_5InferenceExecution {
    /// Loads the drafter that pairs with `model`, when one is installed and
    /// this GPU runs the tree kernels. A drafter that cannot load is logged
    /// and the model decodes one token a step, as without one.
    pub(super) fn host_speculative_draft_runtime(
        &mut self,
        model: &mut Qwen3_5Model,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) {
        // A dense model's layers stream from storage one forward at a time,
        // and the tree pass reads every layer's weights from memory.
        if model.dense_layer_streamer.is_some() {
            return;
        }
        let active_memory_bytes_before_drafter = active_memory_bytes(model);
        self.speculative_draft_runtime = match Qwen3_5SpeculativeDraftRuntime::load_for_target(
            model,
            DraftTargetProjections::KeepLoadedLayout,
            self.memory_limits.active_memory_limit_bytes(),
            &self.model_directory,
            model_loading_performance_attribution,
        ) {
            Ok(speculative_draft_runtime) => speculative_draft_runtime,
            Err(draft_loading_error) => {
                tracing::warn!(
                    %draft_loading_error,
                    "the installed DFlash2 drafter could not be loaded; decoding stays classic"
                );
                None
            }
        };
        if self.speculative_draft_runtime.is_some() {
            let hosted_drafter_payload_bytes =
                active_memory_bytes(model).saturating_sub(active_memory_bytes_before_drafter);
            model.count_hosted_drafter(hosted_drafter_payload_bytes);
            tracing::info!(
                hosted_drafter_payload_bytes,
                "the streaming engine hosts a DFlash2 drafter"
            );
        }
    }

    /// Whether a new session decodes in draft rounds: a drafter is hosted
    /// and the request's semantics are ones a round preserves.
    pub(super) fn admits_drafting_for(
        &self,
        inference_request: &Qwen3_5InferenceRequest,
        sampling_strategy: &crate::Qwen3_5SamplingStrategy,
        visual_embeddings_are_active: bool,
    ) -> bool {
        self.speculative_draft_runtime.is_some()
            && should_admit_speculation_for_session(&SpeculationSessionRequestProperties {
                sampling_strategy: verification_token_choice_for_strategy(sampling_strategy),
                structured_generation_is_active: inference_request
                    .structured_generation()
                    .is_some(),
                forced_thinking_transition_is_pending: false,
                visual_embeddings_are_active,
                repetition_penalty_is_active: inference_request.repetition_penalty_thousandths()
                    != 1_000,
            })
    }
}

/// MLX's active memory, or zero when the snapshot cannot be read: the
/// difference across the drafter load is then zero and nothing is charged.
fn active_memory_bytes(model: &Qwen3_5Model) -> u64 {
    model
        .runtime()
        .memory_snapshot()
        .map_or(0, |memory_snapshot| {
            u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(0)
        })
}
