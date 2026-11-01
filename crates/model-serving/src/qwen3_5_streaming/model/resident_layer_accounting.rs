//! Honest accounting for forwards served from seated complete layers.
//!
//! Two numbers were inflated here (issue #1715):
//!
//! 1. **Avoided storage bytes.** A resident layer's forward used to credit the
//!    whole layer payload as "storage reads avoided". The honest counterfactual
//!    is what a no-cache server would have read for *this route*: the payload of
//!    the distinct routed experts (a decode token's routed set; a chunk's routed
//!    union). A route touching every expert equals exactly one layer payload and
//!    no route can exceed it.
//! 2. **Route hit rate.** Decode tokens served by fully resident layers are
//!    hits too; they were previously invisible to the observatory hit rate,
//!    which only saw paged-layer classifications.
//!
//! Every avoided value is computed by the pure `no_cache_route_source_bytes`
//! (or measured directly from a routed-union page) so the page can be trusted
//! as a measuring instrument.

use mlx_c_rust::MlxArray;

use crate::memory::{RouteCoverageClass, no_cache_route_source_bytes};
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{PerformanceAttribution, PerformanceCounter};

/// How one packed full hit from a seated complete layer is accounted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResidentPackedFullHitAccounting {
    /// One decode token: its top-k experts are distinct, so its demand is
    /// queued and its avoided read credited immediately.
    CreditDecodeTokenNow,
    /// A multi-token chunk: its routed union is only known after the
    /// post-evaluation host copy, so the route is queued exactly once through
    /// the resident-route deferral, which records both its demand and its
    /// avoided bytes. Queuing it through the plain demand deferral as well
    /// would count the chunk's demand twice and skew retention.
    DeferChunkRouteOnce,
}

/// Pure: the accounting a packed full hit of `token_count` routed tokens gets.
#[must_use]
pub(super) const fn resident_packed_full_hit_accounting(
    token_count: i32,
) -> ResidentPackedFullHitAccounting {
    if token_count == 1 {
        ResidentPackedFullHitAccounting::CreditDecodeTokenNow
    } else {
        ResidentPackedFullHitAccounting::DeferChunkRouteOnce
    }
}

impl Qwen3_5Model {
    /// Accounts one hit served from a routed subset of a seated layer: a
    /// decode token without the fused kernel is a covered hit, and the subset
    /// page holds exactly the routed union (this token's set or the chunk's
    /// union), so its payload is the no-cache baseline for this forward.
    pub(super) fn account_resident_subset_hit(
        &self,
        token_count: i32,
        distinct_routed_expert_count: usize,
        subset_payload_byte_count: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        if token_count == 1 {
            self.record_resident_layer_decode_coverage(
                distinct_routed_expert_count,
                performance_attribution,
            );
        }
        self.record_avoided_resident_page(subset_payload_byte_count, performance_attribution);
    }

    /// Queues the demand of one packed full hit exactly once and credits its
    /// avoided storage read.
    ///
    /// A decode token's top-k experts are distinct, so its assignment count is
    /// its no-cache read and can be credited now. A chunk's routed union is
    /// only known after a host copy this fast path exists to avoid, so the
    /// resident-route deferral records both the chunk's demand and its avoided
    /// bytes from the one post-evaluation copy. Queuing the chunk through both
    /// deferrals would count its demand twice and skew retention.
    pub(super) fn account_resident_packed_full_hit(
        &self,
        layer_index: usize,
        expert_capacity: usize,
        token_count: i32,
        selected_indices: &MlxArray,
        complete_layer_payload_bytes: u64,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        match resident_packed_full_hit_accounting(token_count) {
            ResidentPackedFullHitAccounting::CreditDecodeTokenNow => {
                self.defer_expert_demand(layer_index, expert_capacity, selected_indices);
                self.record_resident_layer_decode_hit(
                    selected_indices.element_count(),
                    complete_layer_payload_bytes,
                    expert_capacity,
                    performance_attribution,
                );
            }
            ResidentPackedFullHitAccounting::DeferChunkRouteOnce => {
                self.defer_resident_layer_route(
                    layer_index,
                    expert_capacity,
                    selected_indices,
                    complete_layer_payload_bytes,
                );
            }
        }
    }

    /// Records one decode layer-token served wholly from a seated complete
    /// layer whose packed page multiplies without a transient decode: a fully
    /// covered hit for the hit rate, plus the no-cache read it avoided (the
    /// token's routed experts only, not the layer payload).
    pub(super) fn record_resident_layer_decode_hit(
        &self,
        distinct_routed_expert_count: usize,
        complete_layer_payload_bytes: u64,
        expert_capacity: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        self.record_resident_layer_decode_coverage(
            distinct_routed_expert_count,
            performance_attribution,
        );
        self.record_avoided_resident_route(
            distinct_routed_expert_count,
            complete_layer_payload_bytes,
            expert_capacity,
            performance_attribution,
        );
    }

    /// Records only the coverage classification of a resident-layer decode
    /// layer-token, for paths that account their avoided bytes separately.
    pub(super) fn record_resident_layer_decode_coverage(
        &self,
        distinct_routed_expert_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let retained_assignment_count =
            u64::try_from(distinct_routed_expert_count).unwrap_or(u64::MAX);
        // The same classification counters the paged path records, so the
        // attribution reports and the session counters see resident layers
        // and paged layers through one definition of "covered".
        performance_attribution
            .record_counter(PerformanceCounter::HotExpertRouteFullyCoveredCount, 1);
        performance_attribution.record_counter(
            PerformanceCounter::HotExpertRouteRetainedAssignmentCount,
            retained_assignment_count,
        );
        if let Some(expert_pager) = self.expert_pager.as_ref() {
            expert_pager.record_session_route_coverage(
                retained_assignment_count,
                0,
                RouteCoverageClass::FullyCovered,
            );
        }
    }

    /// Records the no-cache read a resident route avoided, in both the opt-in
    /// attribution counter and the always-on session counter, so the two can
    /// never disagree about the same forward. Callers that already hold the
    /// routed-union page's payload record that value directly instead.
    pub(super) fn record_avoided_resident_route(
        &self,
        distinct_routed_expert_count: usize,
        complete_layer_payload_bytes: u64,
        expert_capacity: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let avoided_source_bytes = no_cache_route_source_bytes(
            complete_layer_payload_bytes,
            expert_capacity,
            distinct_routed_expert_count,
        );
        self.record_avoided_resident_page(avoided_source_bytes, performance_attribution);
    }
}

#[cfg(test)]
#[path = "resident_layer_accounting_tests.rs"]
mod resident_layer_accounting_tests;
