//! Thin orchestrator for a single prompt-processing chunk.
//!
//! Calls the three phases in order:
//! 1. `plan_prompt_prefill_chunk` — pure decision logic (no side effects).
//! 2. `execute_prefill_memory_admission` — adaptive RAM growth.
//! 3. `dispatch_prefill_forward` — GPU forward dispatch for visual, history,
//!    or plain-text input.
//!
//! Collects results into `PromptPrefillChunkOutcome` for the caller in
//! `prefill_advance.rs`.

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::prefill_forward_dispatch::PrefillForwardError;
use super::prompt_prefill_errors;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{fatal_engine_error, qwen3_5_runtime_error};

use crate::{PerformanceCounter, Qwen3_5PersistentPromptCacheBoundaryCheckpoint};

/// Outcome of executing one prompt-processing chunk.
pub(super) struct PromptPrefillChunkOutcome {
    pub(super) active_memory_bytes_before_growth: usize,
    pub(super) retained_expert_payload_bytes_before_growth: u64,
    pub(super) streamed_expert_page_bytes_before_growth: u64,
    pub(super) forward_chunk_elapsed_millis: u64,
    pub(super) adaptive_ram_growth_context: crate::AdaptiveRamGrowthContext,
    pub(super) exact_temporary_workspace_bytes: usize,
    pub(super) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

impl Qwen3_5EngineState {
    /// Execute one prompt-processing chunk: plan → admit → forward.
    pub(super) fn execute_prompt_prefill_chunk(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<PromptPrefillChunkOutcome, PromptPrefillChunkAttemptError> {
        // Phase 1: pure decision logic — no side effects.
        let plan = self.plan_prompt_prefill_chunk(active_request, prefill_start, prefill_end)?;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::PrefillChunkCount, 1);

        // Phase 2: adaptive RAM growth admission.
        let admission_outcome = self.execute_prefill_memory_admission(
            active_request,
            plan.adaptive_ram_growth_context,
            0,
            plan.exact_temporary_workspace_bytes,
            plan.direct_publication_workspace_bytes,
        )?;

        // Phase 3: dispatch the target model's prompt forward.
        //
        // Colibri's bounded staging, applied to our one-shot graph: the
        // chunk's tape holds one union page per streaming layer while the
        // graph builds, so it needs a fixed staging reserve — eight page
        // slots, colibri's READ_DEPTH analog. Ceiling the admitted peak left
        // unclaimed funds that reserve first; only the remainder is borrowed
        // from the warm cache, which then refills from warming after the
        // chunk. The reserve is a page-slot count, never a per-layer tape
        // model: the 2026-10-24 per-layer tape model charged admission for
        // every streamed layer's payload and made admission evict the very
        // tables residency had just seated.
        let staging_page_slot_count: u64 = 8;
        let staging_yield_bytes = {
            let model = self.model.as_ref().ok_or_else(|| {
                PromptPrefillChunkAttemptError::Engine(fatal_engine_error(
                    "Qwen3.5 engine lost its loaded model",
                ))
            })?;
            let staging_page_slot_bytes = model
                .expert_page_reservation_bytes_for_forward(1)
                .map_err(qwen3_5_runtime_error)
                .map_err(PromptPrefillChunkAttemptError::Engine)?;
            let staging_reserve_bytes =
                staging_page_slot_bytes.saturating_mul(staging_page_slot_count);
            let resident_cache_bytes = model
                .expert_weight_memory_cache_statistics()
                .resident_payload_byte_count;
            let staging_yield_bytes = super::prefill_staging_yield::staging_yield_bytes(
                staging_reserve_bytes,
                admission_outcome.unreserved_headroom_bytes,
                resident_cache_bytes,
            );
            if staging_yield_bytes == 0 {
                0
            } else {
                let cache_yielded = model
                    .limit_expert_retention_for_request_memory_pressure(
                        usize::try_from(staging_yield_bytes).unwrap_or(usize::MAX),
                    )
                    .map_err(qwen3_5_runtime_error)
                    .map_err(PromptPrefillChunkAttemptError::Engine)?;
                if cache_yielded {
                    tracing::info!(
                        staging_reserve_bytes,
                        unreserved_headroom_bytes = admission_outcome.unreserved_headroom_bytes,
                        staging_yield_bytes,
                        "warm cache yielded staging slots for the prefill chunk tape"
                    );
                }
                staging_yield_bytes
            }
        };
        let forward_started_at = std::time::Instant::now();
        let dispatch_outcome_result =
            self.dispatch_prefill_forward(active_request, prefill_start, prefill_end, &plan);
        // The staging yield is a per-chunk loan: lift the pressure ceiling
        // immediately whatever the dispatch outcome was, so inter-chunk
        // warming can refill the cache.
        if staging_yield_bytes > 0
            && let Some(model) = self.model.as_ref()
        {
            model.resume_expert_retention_after_request_memory_pressure();
        }
        let dispatch_outcome = match dispatch_outcome_result {
            Ok(dispatch_outcome) => dispatch_outcome,
            Err(PrefillForwardError::Execution(execution_error)) => {
                return Err(prompt_prefill_errors::prefill_execution_error(
                    execution_error,
                ));
            }
        };
        let forward_elapsed = forward_started_at.elapsed();
        if forward_elapsed > std::time::Duration::from_secs(5) {
            tracing::info!(
                prefill_start,
                prefill_end,
                prefill_token_count = plan.prefill_token_count,
                forward_elapsed_millis = forward_elapsed.as_millis(),
                "slow prefill chunk forward"
            );
        }

        let mut boundary_checkpoints = dispatch_outcome.boundary_checkpoints;
        super::prompt_prefill_counters::record_persistent_prompt_cache_boundary_checkpoint(
            active_request,
            plan.prefill_token_count,
            &plan.all_completed_prefill_chunk_tokens,
            &mut boundary_checkpoints,
        )?;

        Ok(PromptPrefillChunkOutcome {
            active_memory_bytes_before_growth: admission_outcome.active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth: admission_outcome
                .retained_expert_payload_bytes_before_growth,
            streamed_expert_page_bytes_before_growth: admission_outcome
                .streamed_expert_page_bytes_before_growth,
            forward_chunk_elapsed_millis: forward_elapsed.as_millis() as u64,
            adaptive_ram_growth_context: plan
                .adaptive_ram_growth_context
                .with_execution_profile(crate::AdaptiveRamGrowthExecutionProfile::Paged),
            exact_temporary_workspace_bytes: plan.exact_temporary_workspace_bytes,
            boundary_checkpoints,
        })
    }
}
