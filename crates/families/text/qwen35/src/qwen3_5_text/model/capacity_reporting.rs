//! What the engine says about each capacity it applies to the
//! `astro_expert_cache`: which account took room back when experts are given
//! back, and, while a reply is recorded, the capacity moment the replay lab
//! serves the reply under.

use memory_policy::ledger::{KeptWeightCapacity, MemoryGovernor};

use super::Qwen3_5Model;

impl Qwen3_5Model {
    /// Says which account took the room back when the `astro_expert_cache`
    /// gives experts back. The capacity is the limit less every need, so
    /// comparing two of these lines names the need that grew; without them a
    /// table dropped and read again a few tokens later cannot be explained.
    pub(super) fn report_released_experts(
        &self,
        capacity: KeptWeightCapacity,
        released_expert_bytes: u64,
    ) {
        let accounts = self
            .governed_memory
            .governor
            .try_borrow()
            .ok()
            .and_then(|memory_governor| memory_governor.as_ref().map(MemoryGovernor::accounts));
        let Some(accounts) = accounts else {
            tracing::info!(
                kept_weight_capacity_bytes = capacity.bytes(),
                released_expert_bytes,
                "the astro_expert_cache gave back what its new capacity no longer holds"
            );
            return;
        };
        tracing::info!(
            kept_weight_capacity_bytes = capacity.bytes(),
            released_expert_bytes,
            limit_bytes = accounts.limit_bytes,
            core_weight_bytes = accounts.core_weight_bytes,
            context_bytes = accounts.context_bytes,
            mlx_working_buffer_bytes = accounts.mlx_working_buffer_bytes,
            mlx_buffer_cache_bytes = accounts.mlx_buffer_cache_bytes,
            system_wired_reserve_bytes = accounts.system_wired_reserve_bytes,
            stream_slot_bytes = accounts.stream_slot_bytes,
            lease_bytes = accounts.lease_bytes,
            margin_bytes = accounts.margin_bytes,
            "the astro_expert_cache gave back what its new capacity no longer holds"
        );
    }

    /// Keeps this capacity for a reply being recorded, with what the reserve
    /// for macOS's wired line was computed from.
    pub(super) fn note_capacity_moment(
        &self,
        capacity: KeptWeightCapacity,
        released_expert_bytes: u64,
    ) {
        let Ok(memory_governor) = self.governed_memory.governor.try_borrow() else {
            return;
        };
        let Some(memory_governor) = memory_governor.as_ref() else {
            return;
        };
        let accounts = memory_governor.accounts();
        let capacity_moment = memory_policy::astro_expert_cache_model::CapacityMoment {
            reply_token_index: 0,
            kept_expert_capacity_bytes: capacity.bytes(),
            system_wired_reserve_bytes: accounts.system_wired_reserve_bytes,
            wired_outside_mlx_bytes: memory_governor
                .latest_wired_outside_mlx_bytes()
                .unwrap_or(0),
            wired_reading_milliseconds: memory_governor
                .latest_wired_reading_milliseconds()
                .unwrap_or(0),
            limit_bytes: accounts.limit_bytes,
            system_wire_limit_bytes: match memory_governor.latest_system_wired_reading() {
                Some((_, system_wire_limit_bytes)) => system_wire_limit_bytes,
                None => 0,
            },
            released_expert_bytes,
        };
        if let Ok(mut route_observation) = self.route_observation.try_borrow_mut() {
            route_observation.note_capacity_moment(capacity_moment);
        }
    }
}
