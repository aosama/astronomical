//! This engine's memory governor (issue #2221).
//!
//! The governor in the policy crate owns the ledger. This engine keeps
//! every weight in memory, so its ledger has no streamable weights and the
//! capacity the governor answers with is always zero: there is nothing to
//! give back. The engine uses the governor for the question it asks before
//! each forward (`forward_memory`), for the MLX working buffers it measures, and
//! for comparing its accounts with what MLX reports.

use memory_policy::ledger::{KeptWeightCapacity, LedgerPhase, MemoryGovernor, ModelMemoryShape};

use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;

use super::Qwen3_5EngineState;
use super::forward_memory::fixed_weight_bytes;

/// The capacity a governor event answers with is the room for kept
/// streamable weights. This engine has none, so no event can ask it to give
/// anything back; naming that here keeps every call site honest about why
/// the answer is dropped.
pub(super) fn nothing_to_give_back(_capacity_for_streamable_weights: KeptWeightCapacity) {}

impl Qwen3_5EngineState {
    /// Opens the governor for the model just loaded: every weight is fixed,
    /// and the MLX working buffers of one prompt chunk and one reply token is
    /// measured now.
    ///
    /// A probe the limit cannot hold does not fail the load. The working
    /// memory stays unmeasured, a forward is then admitted when the weights
    /// and its stored state fit, and one that runs out of memory moves the
    /// request to the streaming engine as it always has.
    pub(in crate::qwen3_5_resident::inference_execution) fn open_memory_governor(
        &self,
        model: &Qwen3_5ResidentModel,
    ) {
        let limit_bytes = self.memory_limits.active_memory_limit_bytes() as u64;
        let model_shape = ModelMemoryShape {
            core_weight_bytes: fixed_weight_bytes(model),
            streamable_weight_bytes: 0,
            reading_prompt_stream_slot_bytes: 0,
            writing_reply_stream_slot_bytes: 0,
        };
        let Ok(mut memory_governor) = MemoryGovernor::new(limit_bytes, model_shape) else {
            tracing::warn!("the memory governor could not open under a zero limit");
            return;
        };
        let prompt_chunk_token_count = self
            .prompt_processing_chunk_sizer
            .prompt_processing_operation_bound_tokens();
        let probe_started_at = std::time::Instant::now();
        match model.measure_forward_mlx_working_buffers(
            prompt_chunk_token_count,
            self.full_attention_kv_state_growth_tokens,
        ) {
            Ok(probed_mlx_working_buffers) => {
                tracing::info!(
                    probe_elapsed_millis = probe_started_at.elapsed().as_millis(),
                    prompt_chunk_token_count,
                    prompt_chunk_mlx_working_buffer_bytes =
                        probed_mlx_working_buffers.prompt_chunk_mlx_working_buffer_bytes,
                    reply_token_mlx_working_buffer_bytes =
                        probed_mlx_working_buffers.reply_token_mlx_working_buffer_bytes,
                    "measured one forward's MLX working buffers at load"
                );
                nothing_to_give_back(memory_governor.record_probe(
                    reading_prompt_phase(prompt_chunk_token_count),
                    probed_mlx_working_buffers.prompt_chunk_mlx_working_buffer_bytes,
                ));
                nothing_to_give_back(memory_governor.record_probe(
                    LedgerPhase::WritingReply,
                    probed_mlx_working_buffers.reply_token_mlx_working_buffer_bytes,
                ));
            }
            Err(probe_error) => {
                tracing::warn!(
                    error = %probe_error,
                    "could not measure a forward's MLX working buffers at load"
                );
                if let Err(cleanup_error) = model
                    .runtime()
                    .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
                {
                    tracing::warn!(error = %cleanup_error, "could not clear MLX after the probe");
                }
            }
        }
        *model.memory_governor.borrow_mut() = Some(memory_governor);
    }

    /// A request starts with `stored_state_bytes` already in memory, such as
    /// a prefix kept from the last turn.
    pub(in crate::qwen3_5_resident::inference_execution) fn govern_request_start(
        &self,
        stored_state_bytes: u64,
    ) {
        let prompt_chunk_phase = self.serving_prompt_chunk_phase();
        if let Some(model) = self.model.as_ref()
            && let Some(memory_governor) = model.memory_governor.borrow_mut().as_mut()
        {
            nothing_to_give_back(
                memory_governor.start_request(prompt_chunk_phase, stored_state_bytes),
            );
        }
    }

    /// The request is over; a prefix kept for the next turn stays on the
    /// stored-state account.
    pub(in crate::qwen3_5_resident::inference_execution) fn govern_request_end(&self) {
        let retained_stored_state_bytes = self.live_retained_context_state_payload_bytes();
        if let Some(model) = self.model.as_ref()
            && let Some(memory_governor) = model.memory_governor.borrow_mut().as_mut()
        {
            nothing_to_give_back(memory_governor.finish_request(retained_stored_state_bytes));
        }
    }

    /// The MLX working buffers set aside for a prompt chunk of the size this
    /// engine serves; `None` while nothing has measured one.
    pub(in crate::qwen3_5_resident::inference_execution) fn measured_prompt_chunk_mlx_working_buffer_bytes(
        &self,
    ) -> Option<u64> {
        let model = self.model.as_ref()?;
        let memory_governor = model.memory_governor.borrow();
        let prompt_chunk_accounts = memory_governor
            .as_ref()?
            .accounts_while(self.serving_prompt_chunk_phase(), 0);
        prompt_chunk_accounts
            .mlx_working_buffers_are_measured
            .then_some(prompt_chunk_accounts.mlx_working_buffer_bytes)
    }

    /// The phase of a prompt chunk of the size this engine serves.
    pub(super) fn serving_prompt_chunk_phase(&self) -> LedgerPhase {
        reading_prompt_phase(
            self.prompt_processing_chunk_sizer
                .prompt_processing_operation_bound_tokens(),
        )
    }
}

impl Qwen3_5ResidentModel {
    /// The limit moved. A limit the governor cannot take (zero) is refused
    /// by the runtime before this point.
    pub(in crate::qwen3_5_resident::inference_execution) fn govern_limit_change(
        &self,
        limit_bytes: u64,
    ) {
        if let Some(memory_governor) = self.memory_governor.borrow_mut().as_mut()
            && let Ok(capacity) = memory_governor.change_limit(limit_bytes)
        {
            nothing_to_give_back(capacity);
        }
    }
}

fn reading_prompt_phase(prompt_chunk_token_count: usize) -> LedgerPhase {
    LedgerPhase::ReadingPrompt {
        span_token_count: u32::try_from(prompt_chunk_token_count).unwrap_or(u32::MAX),
    }
}
