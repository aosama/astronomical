//! Recording what the lookups of an in-graph forward found and read, once
//! the token's evaluation has returned to the model's own thread, where the
//! tallies and the attribution live.

use super::Qwen3_5Model;
use super::in_graph_route_lookup::InGraphLayerOutcome;
use super::in_graph_table_serving::InGraphLayerServing;
use crate::qwen3_5_text::expert_streaming::TableReadThrough;
use crate::{PerformanceAttribution, PerformanceCounter};
use text_generation::execution_error::TextExecutionError;

impl Qwen3_5Model {
    /// Records what the lookups of a served forward found and read, on the
    /// model's thread, where the tallies and the attribution live.
    pub(super) fn record_in_graph_layer_servings(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let layer_servings =
            std::mem::take(&mut *self.in_graph_table_serving().layer_servings.borrow_mut());
        let mut counts = self.in_graph_table_serving().counts.get();
        for layer_serving in &layer_servings {
            let InGraphLayerOutcome::Served(table_read_through) =
                std::mem::take(&mut *layer_serving.outcome.lock())
            else {
                continue;
            };
            counts.served_layer_count += 1;
            self.reply_measurement.stage_tally.record_lookup(
                table_read_through.lookup_elapsed,
                table_read_through
                    .table_fetch_report
                    .map_or(std::time::Duration::ZERO, |table_fetch_report| {
                        table_fetch_report.storage_read_elapsed
                    }),
            );
            counts.fetching_layer_count +=
                u64::from(table_read_through.table_fetch_report.is_some());
            if let Err(accounting_error) = self.record_in_graph_layer_serving(
                layer_serving,
                &table_read_through,
                performance_attribution,
            ) {
                // The token was served; only its figures are incomplete.
                tracing::warn!(
                    layer_index = layer_serving.layer_index,
                    %accounting_error,
                    "an in-graph lookup's reads could not be recorded"
                );
            }
        }
        self.in_graph_table_serving().counts.set(counts);
    }

    fn record_in_graph_layer_serving(
        &self,
        layer_serving: &InGraphLayerServing,
        table_read_through: &TableReadThrough,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        let assignment_count = u64::try_from(layer_serving.assignment_count).unwrap_or(u64::MAX);
        self.record_route_coverage_counts(
            layer_serving.layer_index,
            u64::try_from(table_read_through.retained_expert_count).unwrap_or(u64::MAX),
            u64::try_from(table_read_through.missing_expert_count).unwrap_or(u64::MAX),
            performance_attribution,
        );
        for (prefetch_counter, prefetch_count) in [
            (
                PerformanceCounter::PreviousTokenPrefetchHitCount,
                table_read_through.prefetch_hit_count,
            ),
            (
                PerformanceCounter::PreviousTokenPrefetchMissCount,
                table_read_through.prefetch_miss_count,
            ),
        ] {
            if prefetch_count > 0 {
                performance_attribution.record_counter(prefetch_counter, prefetch_count);
            }
        }
        let Some(table_fetch_report) = table_read_through.table_fetch_report else {
            performance_attribution.record_counter(
                PerformanceCounter::RetainedRouteAssignmentHitCount,
                assignment_count,
            );
            // Only the avoided read: the coverage was recorded above, and
            // recording it again counted every held layer-token twice.
            self.record_avoided_resident_route(
                layer_serving.assignment_count,
                layer_serving.table_payload_bytes,
                layer_serving.expert_capacity,
                performance_attribution,
            );
            return Ok(());
        };
        performance_attribution.record_counter(
            PerformanceCounter::RetainedRouteAssignmentMissCount,
            assignment_count,
        );
        self.record_direct_table_fetch(
            layer_serving.layer_index,
            table_read_through.missing_expert_count,
            table_fetch_report,
            table_read_through.fetch_elapsed,
            performance_attribution,
        )
    }
}
