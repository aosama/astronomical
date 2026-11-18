//! Serves a reply token's missing experts by reading them straight into the
//! layer's table, when the table can take them.
//!
//! This is the model's side of `astro_expert_cache::in_place_fetch`: it
//! decides whether the fetch applies to this layer and this token, hands the
//! `astro_expert_cache` the reader and the layer's file layout, and records
//! what was read. When it does not apply the caller serves the token the
//! copying way, exactly as before.

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::expert_streaming::RoutedExpertCoverage;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5ExpertPager;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{PerformanceAttribution, should_commit_mandatory_routed_page};

impl Qwen3_5Model {
    /// Reads the experts `route_coverage` misses into the layer's table and
    /// returns `true` when the whole route can now be served from it.
    ///
    /// `false` leaves the token to the copying path: no reader, a layer the
    /// reply's plan keeps nothing new on, no table yet, or no room for every
    /// missing expert without evicting one this token uses.
    pub(super) fn fetch_decode_misses_into_table(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_coverage: &RoutedExpertCoverage,
        production_default_paging: bool,
        performance_attribution: &PerformanceAttribution,
    ) -> Result<bool, Qwen3_5ExecutionError> {
        let (Some(slot_filler), Some(astro_expert_cache)) = (
            self.expert_slot_storage_filler.as_ref(),
            self.astro_expert_cache.as_ref(),
        ) else {
            return Ok(false);
        };
        if !self.fetches_decode_misses_into_tables.get() {
            return Ok(false);
        }
        let warm_slot_count = self.hot_expert_warm_slot_count();
        // The same rule that decides whether a streamed page's experts are
        // kept: a layer the plan leaves absent is not refilled behind it.
        let keeps_decode_experts = should_commit_mandatory_routed_page(
            1,
            production_default_paging,
            self.expert_layer_state(layer_index),
            false,
        );
        if warm_slot_count == 0 || !keeps_decode_experts {
            return Ok(false);
        }
        let layer_plan = expert_pager.layer_plan(layer_index)?;
        let fetch_started_at = std::time::Instant::now();
        let storage_read_count = astro_expert_cache
            .borrow_mut()
            .fetch_missing_experts_into_table(
                &self.runtime,
                layer_index,
                layer_plan,
                &route_coverage.missing_expert_ids,
                &route_coverage.retained_expert_ids,
                warm_slot_count.min(layer_plan.expert_capacity),
                slot_filler,
            )?;
        let Some(storage_read_count) = storage_read_count else {
            return Ok(false);
        };
        let fetch_elapsed = fetch_started_at.elapsed();
        let missing_expert_count = route_coverage.missing_expert_ids.len();
        astro_expert_cache
            .borrow_mut()
            .record_disk_load(missing_expert_count, 1);
        let fetched_payload_bytes =
            crate::qwen3_5_streaming::expert_streaming::retained_expert_layer_page_payload_bytes(
                layer_plan,
                missing_expert_count,
            )?;
        self.request_expert_storage_read_tally
            .record_streamed_page(fetched_payload_bytes);
        // The reads went around MLX's loader, which is where reads are
        // counted; record them so storage figures stay whole.
        expert_pager.record_direct_expert_reads(
            performance_attribution,
            u64::try_from(storage_read_count).unwrap_or(u64::MAX),
            fetched_payload_bytes,
            fetch_elapsed,
        );
        Ok(true)
    }
}
