//! Whether one forward fits, and what it needed once it ran (issue #2221).
//!
//! This engine keeps every weight in memory, so the memory governor has
//! nothing to give back on its behalf. The engine asks it one question
//! before each forward: do the weights, the stored state after the forward,
//! the forward's MLX working buffers and the margin fit the limit? A forward
//! that does not fit is never made smaller; the request moves to the
//! streaming engine. After the forward the engine reports what MLX
//! measured, which is where the MLX working-buffer figure comes from.

use memory_policy::ledger::{
    HeldMemory, LedgerAccounts, LedgerPhase, MeasuredStep, MemoryGovernor,
};

use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use crate::qwen3_5_resident::model::request_errors::invalid_request_error;
use crate::{
    InferenceEngineError, ResidentForwardAdmissionDecision, ResidentForwardAdmissionRejection,
    StreamingEngineRequirement,
};

use crate::qwen3_5_core::shared_execution::forward_memory_governance::{
    Qwen3_5ForwardKind, Qwen3_5ForwardMemoryGovernance,
};

use super::Qwen3_5EngineState;
use super::completed_forward_memory::{
    CompletedForwardMemoryObservation, capture_completed_forward_memory_observation,
};
use super::qwen3_5_runtime_error;
use super::resident_memory_governor::nothing_to_give_back;

/// One forward the engine is about to run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::qwen3_5_resident) struct ForwardShape {
    phase: LedgerPhase,
    token_count: usize,
    is_prompt_chunk: bool,
}

impl ForwardShape {
    /// A chunk of prompt tokens run through every layer at once.
    ///
    /// A chunk of one token, such as the last token of a prompt, is the
    /// same work as a reply step and is measured as one. The ledger answers
    /// for a chunk with the figure of the smallest measured chunk at least
    /// as large, so a one-token chunk would otherwise be charged what a
    /// whole chunk needs: a 9B model at a 5 GiB limit was refused its last
    /// prompt token for 280 MB it did not need.
    pub(in crate::qwen3_5_resident) fn prompt_chunk(token_count: usize) -> Self {
        let phase = if token_count == 1 {
            LedgerPhase::WritingReply
        } else {
            LedgerPhase::ReadingPrompt {
                span_token_count: u32::try_from(token_count).unwrap_or(u32::MAX),
            }
        };
        Self {
            phase,
            token_count,
            is_prompt_chunk: true,
        }
    }

    /// One token run through every layer while a reply is written.
    pub(in crate::qwen3_5_resident) const fn reply_token() -> Self {
        Self {
            phase: LedgerPhase::WritingReply,
            token_count: 1,
            is_prompt_chunk: false,
        }
    }

    pub(in crate::qwen3_5_resident) const fn phase(self) -> LedgerPhase {
        self.phase
    }

    const fn prompt_chunk_token_count(self) -> Option<usize> {
        if self.is_prompt_chunk {
            Some(self.token_count)
        } else {
            None
        }
    }
}

/// A forward that was found to fit, kept until the forward is measured.
#[derive(Clone, Copy, Debug)]
pub struct AdmittedForward {
    forward_shape: ForwardShape,
    active_bytes_before_forward: u64,
}

pub(in crate::qwen3_5_resident) enum ForwardAdmissionError {
    /// The forward does not fit; the caller moves the request to the
    /// streaming engine.
    Rejected {
        rejection: ResidentForwardAdmissionRejection,
        reason: String,
    },
    Engine(InferenceEngineError),
}

impl From<InferenceEngineError> for ForwardAdmissionError {
    fn from(inference_engine_error: InferenceEngineError) -> Self {
        Self::Engine(inference_engine_error)
    }
}

impl From<ForwardAdmissionError> for InferenceEngineError {
    fn from(admission_error: ForwardAdmissionError) -> Self {
        match admission_error {
            // A prompt chunk intercepts the typed rejection before this
            // conversion and moves the request to the streaming engine. A
            // reply step or a token fed back into a request has no such
            // caller, so there the rejection is an invalid request.
            ForwardAdmissionError::Rejected { reason, .. } => invalid_request_error(reason),
            ForwardAdmissionError::Engine(inference_engine_error) => inference_engine_error,
        }
    }
}

impl Qwen3_5EngineState {
    /// Asks the memory governor whether `forward_shape` fits, with the
    /// stored state as it will be after the forward.
    ///
    /// `temporary_workspace_bytes` is room the forward needs only while it
    /// runs and whose size is known exactly, such as astro SSD prompt cache blocks
    /// being captured. It counts against the forward's peak, not against
    /// what stays afterwards.
    pub(in crate::qwen3_5_resident) fn admit_forward(
        &mut self,
        forward_shape: ForwardShape,
        request_decoder_state: &RequestDecoderStateStack,
        additional_stored_state_growth_bytes: usize,
        temporary_workspace_bytes: usize,
    ) -> Result<AdmittedForward, ForwardAdmissionError> {
        self.tell_neural_engine_preparation_the_gpu_is_in_use();
        let model = self.core.loaded_model()?;
        let stored_state_growth_bytes = request_decoder_state
            .projected_persistent_state_growth_bytes(
                model.astro_decoder_state_cache_layout(),
                forward_shape.token_count,
            )
            .map_err(qwen3_5_runtime_error)?
            .saturating_add(additional_stored_state_growth_bytes);
        let active_bytes_before_forward = model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?
            .active_memory_bytes() as u64;
        if let Some(memory_governor) = model.memory_governor.borrow_mut().as_mut() {
            if memory_governor.phase() != forward_shape.phase {
                nothing_to_give_back(memory_governor.enter_phase(forward_shape.phase));
            }
            let forward_fit = ForwardFit::ask(
                memory_governor,
                forward_shape,
                held_beyond_weights_bytes(model, active_bytes_before_forward)
                    .saturating_add(stored_state_growth_bytes as u64),
                temporary_workspace_bytes as u64,
            );
            if let Some(rejection) = forward_fit.decision().rejection() {
                return Err(ForwardAdmissionError::Rejected {
                    reason: forward_fit.rejection_reason(rejection),
                    rejection,
                });
            }
            if let Some(prompt_chunk_token_count) = forward_shape.prompt_chunk_token_count() {
                model.decide_neural_engine_for_chunk(
                    prompt_chunk_token_count,
                    forward_fit.set_aside_at_peak_bytes(),
                    forward_fit.accounts_after_forward.limit_bytes,
                );
            }
        }
        model
            .runtime()
            .reset_peak_memory()
            .map_err(qwen3_5_runtime_error)?;
        Ok(AdmittedForward {
            forward_shape,
            active_bytes_before_forward,
        })
    }
}

/// What MLX holds beyond the model's weights: the request's stored state
/// and anything else alive at this boundary. Taking it from MLX rather than
/// from the engine's own figures keeps memory nobody accounts for inside
/// the question.
fn held_beyond_weights_bytes(model: &Qwen3_5ResidentModel, active_bytes: u64) -> u64 {
    active_bytes.saturating_sub(fixed_weight_bytes(model))
}

/// Every weight of this model. This engine keeps them all, so they are the
/// ledger's fixed weights and nothing is streamable.
pub(super) fn fixed_weight_bytes(model: &Qwen3_5ResidentModel) -> u64 {
    model
        .held_model_core_payload_bytes()
        .saturating_add(model.held_expert_payload_bytes())
}

/// The governor's answer for one forward.
struct ForwardFit {
    forward_shape: ForwardShape,
    accounts_after_forward: LedgerAccounts,
    peak_shortfall_bytes: u64,
    temporary_workspace_bytes: u64,
}

impl ForwardFit {
    fn ask(
        memory_governor: &MemoryGovernor,
        forward_shape: ForwardShape,
        stored_state_after_forward_bytes: u64,
        temporary_workspace_bytes: u64,
    ) -> Self {
        let accounts_after_forward =
            memory_governor.accounts_while(forward_shape.phase, stored_state_after_forward_bytes);
        let peak_shortfall_bytes = memory_governor
            .accounts_while(
                forward_shape.phase,
                stored_state_after_forward_bytes.saturating_add(temporary_workspace_bytes),
            )
            .shortfall_bytes;
        Self {
            forward_shape,
            accounts_after_forward,
            peak_shortfall_bytes,
            temporary_workspace_bytes,
        }
    }

    fn decision(&self) -> ResidentForwardAdmissionDecision {
        ResidentForwardAdmissionDecision::for_measured_shortfalls(
            self.accounts_after_forward.stable_shortfall_bytes(),
            self.peak_shortfall_bytes,
            self.forward_shape.prompt_chunk_token_count(),
        )
    }

    /// What the forward holds at its highest point.
    fn set_aside_at_peak_bytes(&self) -> u64 {
        self.accounts_after_forward
            .set_aside_bytes()
            .saturating_add(self.temporary_workspace_bytes)
    }

    fn rejection_reason(&self, rejection: ResidentForwardAdmissionRejection) -> String {
        let ResidentForwardAdmissionRejection::RequireStreamingEngine { cause } = rejection;
        let limit_bytes = self.accounts_after_forward.limit_bytes;
        match cause {
            StreamingEngineRequirement::StableShortfall { shortfall_bytes } => format!(
                "the weights and the stored state after this forward exceed the {limit_bytes} byte limit by {shortfall_bytes} bytes"
            ),
            StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                shortfall_bytes,
                prefill_chunk_token_count,
            } => format!(
                "a prompt chunk of {prefill_chunk_token_count} tokens needs {} bytes of MLX working buffers, {shortfall_bytes} bytes more than the {limit_bytes} byte limit leaves",
                self.accounts_after_forward.mlx_working_buffer_bytes
            ),
            StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk {
                shortfall_bytes,
            } => format!(
                "one reply step needs {} bytes of MLX working buffers, {shortfall_bytes} bytes more than the {limit_bytes} byte limit leaves",
                self.accounts_after_forward.mlx_working_buffer_bytes
            ),
            // The causes below are raised elsewhere: before any forward for
            // the request's context, and after a forward that failed.
            StreamingEngineRequirement::RequestContextShortfall { shortfall_bytes }
            | StreamingEngineRequirement::ForwardAllocationShortfall { shortfall_bytes } => {
                format!("the {limit_bytes} byte limit is exceeded by {shortfall_bytes} bytes")
            }
            StreamingEngineRequirement::ForwardGraphicsProcessorMemoryExhausted => {
                "a forward exhausted graphics-processor memory".to_owned()
            }
        }
    }
}

/// Samples MLX after a forward and tells the memory governor what the
/// forward needed.
///
/// `stored_state_bytes` is the request's stored state by the engine's own
/// count, so the governor can compare its accounts with what MLX reports.
/// The returned sample is the one the governor saw, so memory telemetry
/// describes the same instant.
pub(in crate::qwen3_5_resident) fn report_completed_forward(
    model: &Qwen3_5ResidentModel,
    admitted_forward: AdmittedForward,
    stored_state_bytes: u64,
) -> Result<CompletedForwardMemoryObservation, InferenceEngineError> {
    let observation = capture_completed_forward_memory_observation(model)?;
    let mut memory_governor = model.memory_governor.borrow_mut();
    let Some(memory_governor) = memory_governor.as_mut() else {
        return Ok(observation);
    };
    nothing_to_give_back(memory_governor.settle_context(stored_state_bytes));
    let step_verdict = memory_governor.complete_step(MeasuredStep {
        phase: admitted_forward.forward_shape.phase,
        held_memory: HeldMemory {
            context_bytes: stored_state_bytes,
            kept_weight_bytes: 0,
            stream_slot_bytes: 0,
        },
        in_flight_weight_bytes: 0,
        active_bytes_before_step: admitted_forward.active_bytes_before_forward,
        active_bytes_after_step: observation.mlx_memory_snapshot.active_memory_bytes() as u64,
        peak_bytes_during_step: observation.mlx_memory_snapshot.peak_memory_bytes() as u64,
    });
    nothing_to_give_back(step_verdict.capacity);
    Ok(observation)
}

#[cfg(test)]
#[path = "forward_memory_tests.rs"]
mod tests;

/// The resident engine asks its memory governor before each forward and may
/// refuse it; afterwards it reports what the forward used.
impl Qwen3_5ForwardMemoryGovernance for Qwen3_5EngineState {
    type ForwardInProgress = AdmittedForward;

    fn begin_governed_forward(
        &mut self,
        forward_kind: Qwen3_5ForwardKind,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<AdmittedForward, InferenceEngineError> {
        let forward_shape = match forward_kind {
            Qwen3_5ForwardKind::PromptChunk { token_count } => {
                ForwardShape::prompt_chunk(token_count)
            }
            Qwen3_5ForwardKind::ReplyToken => ForwardShape::reply_token(),
        };
        Ok(self.admit_forward(forward_shape, request_decoder_state, 0, 0)?)
    }

    fn complete_governed_forward(
        &self,
        admitted_forward: AdmittedForward,
        stored_state_bytes: u64,
        _performance_attribution: &mut crate::PerformanceAttribution,
    ) -> Result<runtime_integration::MlxMemorySnapshot, InferenceEngineError> {
        let model = self.core.loaded_model()?;
        Ok(
            report_completed_forward(model, admitted_forward, stored_state_bytes)?
                .mlx_memory_snapshot,
        )
    }
}
