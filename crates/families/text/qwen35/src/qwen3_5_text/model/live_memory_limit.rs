//! Coordinates the live MLX ceiling with the streaming pager's retained-page budget.
//!
//! Keeping both limits in one transition prevents the retained `astro_expert_cache` from exceeding
//! the user's current memory grant. A change may land between two steps of a running request
//! (issue #2681); the floor then also keeps what that request holds and its next step's buffers.

use memory_policy::live_memory_limit::{
    InFlightRequestMemory, needs_at_least_message, request_aware_ceiling_floor_bytes,
};
use runtime_integration::MlxMemoryLimits;

use crate::{
    InferenceEngineError, MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements,
    geometry_safe_minimum_mlx_memory_ceiling_bytes,
};

use crate::qwen3_5_core::model::live_memory_observation::{
    LiveMemoryObservation, observe_live_memory, post_reclamation_active_memory_bytes,
    requested_mlx_memory_limits_when_changed,
};

use super::Qwen3_5Model;
use super::prefill_partial_layer_read::should_clear_prefill_completion_refusals;

impl Qwen3_5Model {
    pub fn minimum_mlx_memory_ceiling_bytes(&self) -> Result<u64, InferenceEngineError> {
        Ok(geometry_safe_minimum_mlx_memory_ceiling_bytes(
            &self.memory_facts().model_geometry(),
        ))
    }

    /// Moves the model to a new ceiling, between requests or between two
    /// steps of a running one (`request_is_running`).
    pub fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
        request_is_running: bool,
    ) -> Result<(u64, MlxMemoryLimits), InferenceEngineError> {
        let LiveMemoryObservation {
            current_mlx_memory_limits,
            current_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes: idle_floor_bytes,
            current_active_memory_bytes,
        } = observe_live_memory(&self.runtime, self.minimum_mlx_memory_ceiling_bytes()?)?;
        let (minimum_mlx_memory_ceiling_bytes, ceiling_change_decision) = self
            .decide_live_ceiling_change(
                requested_mlx_memory_ceiling_bytes,
                request_is_running,
                current_mlx_memory_ceiling_bytes,
                idle_floor_bytes,
                current_active_memory_bytes,
            )?;
        let Some(requested_mlx_memory_limits) = requested_mlx_memory_limits_when_changed(
            requested_mlx_memory_ceiling_bytes,
            current_mlx_memory_limits,
        )?
        else {
            return Ok((minimum_mlx_memory_ceiling_bytes, current_mlx_memory_limits));
        };
        let is_lowering_mlx_memory_ceiling = matches!(
            ceiling_change_decision,
            MemoryCeilingChangeDecision::Lower { .. }
        );
        // The ceiling a failed change rolls back to: the pager's own when it
        // has one, otherwise the current one, which a dense model's budget
        // and kept layers return to.
        let previous_memory_ceiling_bytes = self
            .expert_pager
            .as_ref()
            .map_or(current_mlx_memory_ceiling_bytes, |expert_pager| {
                expert_pager.configured_mlx_memory_ceiling_bytes()
            });
        if is_lowering_mlx_memory_ceiling {
            // Native eviction comes first because the old MLX ceiling still
            // permits the bookkeeping and synchronization needed to release
            // pages; the helper enforces that order and rolls the pager
            // ceiling back if reclaimed memory still exceeds the request.
            self.lower_ceiling_with_expert_reclamation(
                requested_mlx_memory_ceiling_bytes,
                previous_memory_ceiling_bytes,
                request_is_running,
            )?;
        }

        self.install_runtime_memory_limit_with_rollback(
            requested_mlx_memory_limits,
            previous_memory_ceiling_bytes,
        )?;
        if !is_lowering_mlx_memory_ceiling
            && let Err(policy_update_error) = self
                .update_expert_residency_for_live_mlx_memory_limit(
                    requested_mlx_memory_ceiling_bytes,
                )
        {
            return Err(self.restore_failed_raised_memory_limit(
                current_mlx_memory_limits,
                previous_memory_ceiling_bytes,
                policy_update_error,
            ));
        }
        self.clear_prefill_completion_refusals_when_raise_sticks(
            current_mlx_memory_ceiling_bytes,
            requested_mlx_memory_ceiling_bytes,
        );
        Ok((
            minimum_mlx_memory_ceiling_bytes,
            requested_mlx_memory_limits,
        ))
    }

    /// The floor the model can move to now and what the change asks of it, or
    /// the refusal of a ceiling below that floor.
    fn decide_live_ceiling_change(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
        request_is_running: bool,
        current_mlx_memory_ceiling_bytes: u64,
        idle_floor_bytes: u64,
        current_active_memory_bytes: u64,
    ) -> Result<(u64, MemoryCeilingChangeDecision), InferenceEngineError> {
        // Kept dense layers are given back like retained experts.
        let retained_paged_expert_payload_bytes = self
            .astro_expert_cache_statistics()
            .resident_payload_byte_count
            .saturating_add(self.dense_kept_layer_payload_bytes());
        let minimum_mlx_memory_ceiling_bytes = self.ceiling_floor_bytes(
            idle_floor_bytes,
            request_is_running,
            current_active_memory_bytes,
            retained_paged_expert_payload_bytes,
        );
        let ceiling_change_decision = MemoryCeilingChangeRequirements {
            current_ceiling_bytes: current_mlx_memory_ceiling_bytes,
            requested_ceiling_bytes: requested_mlx_memory_ceiling_bytes,
            minimum_safe_ceiling_bytes: minimum_mlx_memory_ceiling_bytes,
            current_active_memory_bytes,
            retained_paged_expert_payload_bytes,
            complete_experts_are_resident: false,
            complete_residency_required_headroom_bytes: 0,
        }
        .decide();
        if let MemoryCeilingChangeDecision::Reject { .. } = ceiling_change_decision {
            return Err(below_floor_rejection(
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                request_is_running,
            ));
        }
        Ok((minimum_mlx_memory_ceiling_bytes, ceiling_change_decision))
    }

    /// A higher ceiling can seat a layer the previous one refused. Clearing
    /// only after the raise sticks lets the next prefill try completion once.
    fn clear_prefill_completion_refusals_when_raise_sticks(
        &mut self,
        current_mlx_memory_ceiling_bytes: u64,
        requested_mlx_memory_ceiling_bytes: u64,
    ) {
        if should_clear_prefill_completion_refusals(
            current_mlx_memory_ceiling_bytes,
            requested_mlx_memory_ceiling_bytes,
        ) {
            self.prefill_layer_completion_refused.borrow_mut().clear();
        }
    }

    /// Lowers the ceiling by first reclaiming retained experts under the old,
    /// larger limit, then proving the reclaimed memory fits the new one.
    /// Native eviction comes first because the old MLX ceiling still permits
    /// the bookkeeping and synchronization needed to release pages;
    /// installing the smaller runtime limit first could make the reclamation
    /// operation reject its own temporary work. If active memory still
    /// exceeds the request after reclamation, the pager ceiling is restored
    /// and the whole transition fails without ever touching the runtime.
    ///
    /// Memory left above the request is refused, not fatal: the policy is
    /// already back at the old ceiling and the model keeps serving under it.
    fn lower_ceiling_with_expert_reclamation(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
        previous_memory_ceiling_bytes: u64,
        request_is_running: bool,
    ) -> Result<(), InferenceEngineError> {
        self.prepare_expert_residency_for_lower_mlx_memory_limit(
            requested_mlx_memory_ceiling_bytes,
        )?;
        let post_reclamation_active_memory_bytes =
            post_reclamation_active_memory_bytes(&self.runtime)?;
        if post_reclamation_active_memory_bytes > requested_mlx_memory_ceiling_bytes {
            self.restore_memory_ceiling_policy(previous_memory_ceiling_bytes)?;
            return Err(below_floor_rejection(
                requested_mlx_memory_ceiling_bytes,
                post_reclamation_active_memory_bytes,
                request_is_running,
            ));
        }
        Ok(())
    }

    /// The smallest ceiling the model can move to now. A running request also
    /// keeps the active memory kept weights do not account for, which it
    /// cannot give back, and the MLX working buffers its next step allocates,
    /// which the memory governor has measured.
    fn ceiling_floor_bytes(
        &self,
        idle_floor_bytes: u64,
        request_is_running: bool,
        current_active_memory_bytes: u64,
        reclaimable_bytes: u64,
    ) -> u64 {
        let in_flight_request_memory = if request_is_running {
            Some(InFlightRequestMemory {
                non_evictable_active_bytes: current_active_memory_bytes
                    .saturating_sub(reclaimable_bytes),
                next_step_working_buffer_bytes: self.measured_mlx_working_buffer_bytes(),
            })
        } else {
            None
        };
        request_aware_ceiling_floor_bytes(idle_floor_bytes, in_flight_request_memory)
    }

    /// The MLX working buffers the memory governor has measured a step to
    /// need; zero before any step was measured.
    fn measured_mlx_working_buffer_bytes(&self) -> u64 {
        let Ok(memory_governor) = self.governed_memory.governor.try_borrow() else {
            return 0;
        };
        match memory_governor.as_ref() {
            Some(memory_governor) => memory_governor.accounts().mlx_working_buffer_bytes,
            None => 0,
        }
    }

    /// Installs the new runtime limits, rolling the pager ceiling back if the
    /// runtime refuses them. A restore failure propagates as-is (it is its
    /// own fatal condition); the refused update error is wrapped as the
    /// runtime error the caller reports.
    fn install_runtime_memory_limit_with_rollback(
        &mut self,
        requested_mlx_memory_limits: MlxMemoryLimits,
        previous_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        if let Err(runtime_update_error) = self
            .runtime
            .update_memory_limits(requested_mlx_memory_limits)
        {
            self.restore_memory_ceiling_policy(previous_memory_ceiling_bytes)?;
            return Err(text_generation::engine_errors::fatal_runtime_error(
                runtime_update_error,
            ));
        }
        Ok(())
    }

    fn prepare_expert_residency_for_lower_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        self.memory_facts
            .borrow_mut()
            .update_mlx_active_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes)
            .map_err(|model_memory_facts_error| {
                text_generation::engine_errors::fatal_engine_error(
                    model_memory_facts_error.to_string(),
                )
            })?;
        // The governor follows the installed limit, and the astro expert cache or
        // a dense model's kept layers give back what no longer fits.
        self.govern_limit_change();
        let Some(expert_pager) = self.expert_pager.as_mut() else {
            return Ok(());
        };
        expert_pager.update_configured_mlx_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes);
        Ok(())
    }

    fn update_expert_residency_for_live_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        self.apply_live_ceiling_to_budget_and_dense_layers(requested_mlx_memory_ceiling_bytes)?;
        let Some(expert_pager) = self.expert_pager.as_mut() else {
            return Ok(());
        };
        expert_pager.update_configured_mlx_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes);
        Ok(())
    }

    /// Moves the memory facts and the memory governor to the new ceiling, then
    /// reads back the dense layers the governor now has room for.
    fn apply_live_ceiling_to_budget_and_dense_layers(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        self.memory_facts
            .borrow_mut()
            .update_mlx_active_memory_ceiling_bytes(requested_mlx_memory_ceiling_bytes)
            .map_err(|model_memory_facts_error| {
                text_generation::engine_errors::fatal_engine_error(
                    model_memory_facts_error.to_string(),
                )
            })?;
        self.govern_limit_change();
        Ok(self.keep_dense_layers_the_governor_has_room_for()?)
    }

    /// Returns the budget, the pager and a dense model's kept layers to the
    /// ceiling a failed change started from.
    fn restore_memory_ceiling_policy(
        &mut self,
        previous_memory_ceiling_bytes: u64,
    ) -> Result<(), InferenceEngineError> {
        self.update_expert_residency_for_live_mlx_memory_limit(previous_memory_ceiling_bytes)
    }

    fn restore_failed_raised_memory_limit(
        &mut self,
        previous_mlx_memory_limits: MlxMemoryLimits,
        previous_memory_ceiling_bytes: u64,
        transition_error: InferenceEngineError,
    ) -> InferenceEngineError {
        let policy_restore_result =
            self.restore_memory_ceiling_policy(previous_memory_ceiling_bytes);
        let runtime_restore_result = self
            .runtime
            .update_memory_limits(previous_mlx_memory_limits);
        match (policy_restore_result, runtime_restore_result) {
            (Ok(()), Ok(())) => transition_error,
            (policy_restore_result, runtime_restore_result) => {
                text_generation::engine_errors::fatal_engine_error(format!(
                    "raised MLX memory transition failed and rollback was incomplete: transition_error={transition_error}; policy_restore_error={:?}; runtime_restore_error={:?}",
                    policy_restore_result.err(),
                    runtime_restore_result.err(),
                ))
            }
        }
    }
}

/// The answer to a ceiling below the floor. While a request runs it names the
/// smallest ceiling in GB, which the memory control shows beside its red slider.
fn below_floor_rejection(
    requested_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
    request_is_running: bool,
) -> InferenceEngineError {
    let needs_at_least = needs_at_least_message(minimum_mlx_memory_ceiling_bytes);
    let reason = if request_is_running {
        format!("{needs_at_least} while this request runs")
    } else {
        format!("{needs_at_least} for the loaded model")
    };
    InferenceEngineError::MlxMemoryLimitRejected {
        requested_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        reason,
    }
}
