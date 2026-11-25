//! Serves a reply token's missing experts by reading them straight into the
//! layer's table, when the table can take them.
//!
//! This is the model's side of `astro_expert_cache::direct_fetch`: it
//! decides whether the fetch applies to this layer and this token, hands the
//! `astro_expert_cache` the reader and the layer's file layout, and records
//! what was read. When it does not apply the caller serves the token the
//! copying way, exactly as before.

use crate::qwen3_5_streaming::expert_streaming::ExpertSlotStorageFiller;
use crate::qwen3_5_streaming::expert_streaming::RoutedExpertCoverage;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5ExpertPager;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::qwen3_5_streaming::model::direct_expert_fetch_tally::{
    DirectExpertFetchOutcome, DirectExpertFetchTally,
};
use crate::{PerformanceAttribution, should_commit_mandatory_routed_page};
use text_generation::execution_error::TextExecutionError;

/// What a model needs to read a reply token's missing experts straight into
/// their tables, and the record of how often it could.
#[derive(Debug)]
pub struct DirectExpertFetching {
    /// `None` for a model without streamed experts, or when its files could
    /// not be opened for direct reads.
    pub slot_filler: Option<ExpertSlotStorageFiller>,
    /// Off only while a measurement compares the copying path with the
    /// direct one on the same loaded model.
    pub is_enabled: std::cell::Cell<bool>,
    /// Which way each fetching layer's missing experts went, and why.
    pub tally: DirectExpertFetchTally,
}

impl DirectExpertFetching {
    /// Opens the reader for a model that streams experts. Without it every
    /// missing expert is served the copying way, which is slower and still
    /// correct, so a failure to open it is logged, not fatal.
    ///
    /// It borrows the pager, so a model under construction calls this before
    /// the pager moves into it.
    pub fn open(expert_pager: Option<&Qwen3_5ExpertPager>) -> Self {
        let slot_filler = match expert_pager {
            None => None,
            Some(expert_pager) => match ExpertSlotStorageFiller::open(expert_pager.layer_plans()) {
                Ok(slot_filler) => Some(slot_filler),
                Err(open_error) => {
                    tracing::warn!(
                        %open_error,
                        "experts will be copied into their tables: the direct reader did not open"
                    );
                    None
                }
            },
        };
        Self {
            slot_filler,
            is_enabled: std::cell::Cell::new(true),
            tally: DirectExpertFetchTally::default(),
        }
    }
}

impl Qwen3_5Model {
    /// Reads the experts `route_coverage` misses into the layer's table and
    /// returns `true` when the whole route can now be served from it.
    ///
    /// `false` leaves the token to the copying path: no reader, a layer the
    /// reply's plan keeps nothing new on or whose tables have no slots, no
    /// table yet, or no room for every missing expert without evicting one
    /// this token uses.
    pub(super) fn fetch_decode_misses_into_table(
        &self,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_coverage: &RoutedExpertCoverage,
        production_default_paging: bool,
        performance_attribution: &PerformanceAttribution,
    ) -> Result<bool, TextExecutionError> {
        let (Some(slot_filler), Some(astro_expert_cache)) = (
            self.direct_expert_fetching.slot_filler.as_ref(),
            self.astro_expert_cache.as_ref(),
        ) else {
            self.direct_expert_fetching
                .tally
                .record_outcome(DirectExpertFetchOutcome::NoDirectReader);
            return Ok(false);
        };
        if !self.reads_host_routes_straight_into_tables() {
            self.direct_expert_fetching
                .tally
                .record_outcome(DirectExpertFetchOutcome::NoDirectReader);
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
            self.direct_expert_fetching
                .tally
                .record_outcome(DirectExpertFetchOutcome::LayerKeepsNothing);
            return Ok(false);
        }
        let layer_plan = expert_pager.layer_plan(layer_index)?;
        let fetch_started_at = std::time::Instant::now();
        let table_fetch_report = astro_expert_cache
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
        let Some(table_fetch_report) = table_fetch_report else {
            self.direct_expert_fetching
                .tally
                .record_outcome(DirectExpertFetchOutcome::TableCouldNotTakeThem);
            return Ok(false);
        };
        let missing_expert_count = route_coverage.missing_expert_ids.len();
        astro_expert_cache
            .borrow_mut()
            .record_disk_load(missing_expert_count, 1);
        self.record_direct_table_fetch(
            layer_index,
            missing_expert_count,
            table_fetch_report,
            fetch_started_at.elapsed(),
            performance_attribution,
        )?;
        Ok(true)
    }

    /// Records one fetch of `missing_expert_count` experts into a layer's
    /// table, whichever thread made it: this one, or a lookup inside the
    /// graph whose report is read here after the token's evaluation.
    pub(super) fn record_direct_table_fetch(
        &self,
        layer_index: usize,
        missing_expert_count: usize,
        table_fetch_report: crate::qwen3_5_streaming::expert_streaming::TableFetchReport,
        fetch_elapsed: std::time::Duration,
        performance_attribution: &PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(());
        };
        let fetched_payload_bytes =
            crate::qwen3_5_streaming::expert_streaming::retained_expert_layer_page_payload_bytes(
                expert_pager.layer_plan(layer_index)?,
                missing_expert_count,
            )?;
        self.direct_expert_fetching
            .tally
            .record_outcome(DirectExpertFetchOutcome::FetchedIntoTable);
        self.direct_expert_fetching.tally.record_fetch(
            missing_expert_count,
            fetched_payload_bytes,
            fetch_elapsed,
            table_fetch_report.storage_read_elapsed,
        );
        self.request_expert_storage_read_tally
            .record_streamed_page(fetched_payload_bytes);
        // The reads went around MLX's loader, which is where reads are
        // counted; record them so storage figures stay whole.
        expert_pager.record_direct_expert_reads(
            performance_attribution,
            u64::try_from(table_fetch_report.storage_read_count).unwrap_or(u64::MAX),
            fetched_payload_bytes,
            fetch_elapsed,
        );
        Ok(())
    }
}
