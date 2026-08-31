//! Laguna-owned expert residency status: resident, complete-layer, or routed.

use std::cell::{Cell, Ref, RefCell};

use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::ExpertResidencyTelemetry;
use crate::expert_paging::{
    ExpertWeightMemoryCacheStatistics, ExpertWeightPage, RetainedExpertPageCache,
    RetainedExpertReclamation,
};
use crate::laguna::normalization::{LagunaFeedForwardDescriptor, LagunaTargetContract};
use crate::laguna::paging::{LagunaExpertPagingPlan, LagunaExpertWeightPage};
use crate::memory::{
    ExpertResidencyPhase, PhaseAwareExpertResidencyPlan, RequestExpertResidency,
    publish_request_stable_residency_plan, should_commit_mandatory_complete_layer,
    should_commit_mandatory_routed_page,
};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::error::LagunaExecutionError;
use super::expert_coverage::{
    resident_complete_payload_bytes, resident_sparse_layer_count, sparse_layer_count,
    sparse_layer_counts,
};
use super::weights::LagunaNativeWeights;

/// Last sparse-expert grain executed by the model.
///
/// `expert_count` carries the number of routed experts the grain materialized so
/// residency telemetry can report the same grain as the payload figures.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::laguna) enum LagunaLastExpertForward {
    #[default]
    None,
    StreamedCompleteLayer {
        layer_count: u32,
        expert_count: u32,
        payload_bytes: u64,
    },
    StreamedRoutedPage {
        layer_count: u32,
        expert_count: u32,
        payload_bytes: u64,
    },
}

/// Phase-aware plan plus last-forward grain used by Laguna status.
pub(super) struct LagunaExpertResidencyState {
    paging_plan: Option<LagunaExpertPagingPlan>,
    active_plan: RefCell<Option<PhaseAwareExpertResidencyPlan>>,
    request_residency: RefCell<Option<RequestExpertResidency>>,
    last_forward: RefCell<LagunaLastExpertForward>,
    retained_layers: RefCell<Option<RetainedExpertPageCache<LagunaExpertWeightPage>>>,
    retained_expert_ceiling_bytes: Cell<u64>,
}

impl LagunaExpertResidencyState {
    pub(super) fn new() -> Self {
        Self {
            paging_plan: None,
            active_plan: RefCell::new(None),
            request_residency: RefCell::new(None),
            last_forward: RefCell::new(LagunaLastExpertForward::None),
            retained_layers: RefCell::new(None),
            retained_expert_ceiling_bytes: Cell::new(0),
        }
    }

    pub(super) fn attach_paging_plan(&mut self, paging_plan: LagunaExpertPagingPlan) {
        let sparse_layer_count = paging_plan.sparse_layers().len();
        self.paging_plan = Some(paging_plan);
        let mut retained_layers = RetainedExpertPageCache::new(sparse_layer_count);
        retained_layers.update_maximum_resident_payload_bytes(0);
        self.retained_layers.replace(Some(retained_layers));
        self.retained_expert_ceiling_bytes.set(0);
    }

    pub(super) fn set_retained_expert_ceiling(
        &self,
        retained_expert_ceiling_bytes: u64,
    ) -> Result<(), LagunaExecutionError> {
        let mut retained_layers = self.retained_layers.borrow_mut();
        let Some(retained_layers) = retained_layers.as_mut() else {
            return Err(LagunaExecutionError::invalid_geometry(
                "a retained-expert ceiling requires an attached paging plan",
            ));
        };
        retained_layers.update_maximum_resident_payload_bytes(retained_expert_ceiling_bytes);
        self.retained_expert_ceiling_bytes
            .set(retained_expert_ceiling_bytes);
        self.last_forward.replace(LagunaLastExpertForward::None);
        Ok(())
    }

    pub(super) fn paging_plan(&self) -> Option<&LagunaExpertPagingPlan> {
        self.paging_plan.as_ref()
    }

    pub(super) fn record_forward(&self, last_forward: LagunaLastExpertForward) {
        *self.last_forward.borrow_mut() = last_forward;
    }

    pub(super) fn record_disk_load(&self, expert_count: usize, batch_count: usize) {
        if let Some(retained_layers) = self.retained_layers.borrow_mut().as_mut() {
            retained_layers.record_disk_load(expert_count, batch_count);
        }
    }

    pub(super) fn refresh_explicit_phase_plan(&self, phase: ExpertResidencyPhase) {
        let Some(paging_plan) = self.paging_plan.as_ref() else {
            self.active_plan.replace(None);
            return;
        };
        let expert_capacity = self
            .paging_plan
            .as_ref()
            .and_then(|paging_plan| {
                paging_plan
                    .layer_geometries()
                    .ok()
                    .and_then(|geometries| geometries.first().map(|g| g.expert_capacity))
            })
            .unwrap_or(0);
        let current_residencies = self
            .retained_layers
            .borrow()
            .as_ref()
            .map(|retained_layers| retained_layers.topology_snapshot(expert_capacity))
            .unwrap_or_default();
        match paging_plan.plan_phase_aware_residency(
            phase,
            self.retained_expert_ceiling_bytes.get(),
            &current_residencies,
        ) {
            Ok(candidate_plan) => {
                let layer_geometries = paging_plan
                    .layer_geometries()
                    .unwrap_or_else(|_| Vec::new());
                let (next_request_residency, active_plan) = publish_request_stable_residency_plan(
                    phase,
                    self.request_residency.borrow().as_ref(),
                    candidate_plan,
                    &current_residencies,
                    0,
                    &layer_geometries,
                );
                self.request_residency.replace(next_request_residency);
                self.active_plan.replace(Some(active_plan));
                if phase == ExpertResidencyPhase::GenerationPreparation
                    && let Some(retained_layers) = self.retained_layers.borrow_mut().as_mut()
                {
                    retained_layers.clear_expert_demand();
                }
            }
            Err(_) => {
                self.active_plan.replace(None);
            }
        }
    }

    pub(super) fn reclaim_for_request_pressure(
        &self,
        required_reclamation_bytes: u64,
    ) -> RetainedExpertReclamation {
        let mut retained_layers = self.retained_layers.borrow_mut();
        let Some(retained_layers) = retained_layers.as_mut() else {
            return RetainedExpertReclamation::default();
        };
        let reclamation = retained_layers.reclaim_for_request_pressure(required_reclamation_bytes);
        let admitted_payload_ceiling = retained_layers.statistics().resident_payload_byte_count;
        retained_layers.limit_for_request_pressure_to_maximum(admitted_payload_ceiling);
        reclamation
    }

    pub(super) fn resume_after_request_pressure(&self) {
        if let Some(retained_layers) = self.retained_layers.borrow_mut().as_mut() {
            retained_layers.resume_after_request_pressure();
        }
    }

    pub(super) fn record_expert_demand(
        &self,
        paging_slot_index: usize,
        expert_capacity: usize,
        selected_expert_ids: &[usize],
    ) {
        if let Some(retained_layers) = self.retained_layers.borrow_mut().as_mut() {
            retained_layers.record_expert_demand(
                paging_slot_index,
                expert_capacity,
                selected_expert_ids,
            );
        }
    }

    /// Returns whether a retained complete layer can serve this paging slot.
    pub(super) fn has_retained_complete_layer(&self, paging_slot_index: usize) -> bool {
        self.retained_layers
            .borrow()
            .as_ref()
            .and_then(|retained_layers| retained_layers.retained_layer(paging_slot_index))
            .is_some_and(|retained_page| retained_page.manifest().contains_all_experts())
    }

    /// Returns whether a retained page already covers every selected expert.
    pub(super) fn retained_page_covers_experts(
        &self,
        paging_slot_index: usize,
        selected_expert_ids: &[usize],
    ) -> bool {
        self.retained_layers
            .borrow()
            .as_ref()
            .and_then(|retained_layers| retained_layers.retained_layer(paging_slot_index))
            .is_some_and(|retained_page| {
                retained_page
                    .manifest()
                    .contains_every_expert(selected_expert_ids)
            })
    }

    /// Executes gathered SwiGLU on the retained complete layer for one paging slot.
    pub(super) fn with_retained_complete_layer<Output, Execute>(
        &self,
        paging_slot_index: usize,
        execute_on_page: Execute,
    ) -> Result<Output, LagunaExecutionError>
    where
        Execute: FnOnce(&LagunaExpertWeightPage) -> Result<Output, LagunaExecutionError>,
    {
        let retained_layers = self.retained_layers.borrow();
        let retained_page = retained_layers
            .as_ref()
            .and_then(|retained_layers| retained_layers.retained_layer(paging_slot_index))
            .ok_or_else(|| {
                LagunaExecutionError::invalid_geometry(
                    "a retained complete Laguna layer was expected for this paging slot",
                )
            })?;
        execute_on_page(retained_page)
    }

    /// Offers a just-loaded complete layer to retained ownership when the plan allows it.
    pub(super) fn try_commit_complete_layer(
        &self,
        paging_slot_index: usize,
        expert_capacity: usize,
        expert_page: LagunaExpertWeightPage,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), LagunaExecutionError> {
        let residency_target = self
            .active_plan()
            .as_ref()
            .and_then(|active_plan| active_plan.layer_targets.get(paging_slot_index).copied());
        if !should_commit_mandatory_complete_layer(2, true, residency_target) {
            return Ok(());
        }
        let mut retained_layers = self.retained_layers.borrow_mut();
        let Some(retained_layers) = retained_layers.as_mut() else {
            return Ok(());
        };
        if !retained_layers.can_commit_materialized_page(
            paging_slot_index,
            expert_page.resident_payload_byte_count(),
        ) {
            return Ok(());
        }
        let _commit = performance_attribution.measure_operation(
            PerformanceOperation::ExpertResidencyCommit,
            |_| {
                retained_layers.commit_materialized_complete_layer(
                    paging_slot_index,
                    expert_capacity,
                    expert_page,
                )
            },
        )?;
        Ok(())
    }

    /// Offers a just-loaded routed page to retained ownership when the plan allows it.
    pub(super) fn try_commit_routed_page(
        &self,
        paging_slot_index: usize,
        expert_capacity: usize,
        expert_ids: Vec<usize>,
        expert_page: LagunaExpertWeightPage,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), LagunaExecutionError> {
        let residency_target = self
            .active_plan()
            .as_ref()
            .and_then(|active_plan| active_plan.layer_targets.get(paging_slot_index).copied());
        let layer_has_no_retained_page =
            self.retained_layers
                .borrow()
                .as_ref()
                .is_none_or(|retained_layers| {
                    retained_layers.retained_layer(paging_slot_index).is_none()
                });
        if !should_commit_mandatory_routed_page(
            1,
            true,
            residency_target,
            layer_has_no_retained_page,
        ) {
            return Ok(());
        }
        let mut retained_layers = self.retained_layers.borrow_mut();
        let Some(retained_layers) = retained_layers.as_mut() else {
            return Ok(());
        };
        if !retained_layers.can_commit_materialized_page(
            paging_slot_index,
            expert_page.resident_payload_byte_count(),
        ) {
            return Ok(());
        }
        let _commit = performance_attribution.measure_operation(
            PerformanceOperation::ExpertResidencyCommit,
            |_| {
                retained_layers.commit_materialized_routed_page(
                    paging_slot_index,
                    expert_capacity,
                    expert_ids,
                    expert_page,
                )
            },
        )?;
        Ok(())
    }

    pub(super) fn active_plan(&self) -> Ref<'_, Option<PhaseAwareExpertResidencyPlan>> {
        self.active_plan.borrow()
    }

    pub(super) fn retained_expert_ceiling_bytes(&self) -> u64 {
        self.retained_expert_ceiling_bytes.get()
    }

    pub(super) fn expert_memory_mode(
        &self,
        contract: &LagunaTargetContract,
        weights: &LagunaNativeWeights,
    ) -> ExpertMemoryMode {
        let (sparse_layer_count, resident_layer_count) = sparse_layer_counts(contract, weights);
        let retained_complete_layer_count = self
            .retained_layers
            .borrow()
            .as_ref()
            .map(|retained_layers| retained_layers.statistics().complete_layer_count)
            .unwrap_or(0);
        if sparse_layer_count == 0
            || resident_layer_count.saturating_add(retained_complete_layer_count)
                == sparse_layer_count
        {
            return ExpertMemoryMode::Resident;
        }
        let retained_payload_bytes = self
            .retained_layers
            .borrow()
            .as_ref()
            .map(|retained_layers| retained_layers.statistics().resident_payload_byte_count)
            .unwrap_or(0);
        if retained_payload_bytes > 0 {
            return ExpertMemoryMode::Hybrid;
        }
        if resident_layer_count == 0 && self.paging_plan.is_some() {
            return ExpertMemoryMode::Paged;
        }
        ExpertMemoryMode::Hybrid
    }

    pub(super) fn expert_residency_telemetry(
        &self,
        contract: &LagunaTargetContract,
        weights: &LagunaNativeWeights,
    ) -> ExpertResidencyTelemetry {
        let statistics = self.expert_weight_memory_cache_statistics(contract, weights);
        let total_layer_count = u32::try_from(sparse_layer_count(contract)).unwrap_or(u32::MAX);
        let last_forward = *self.last_forward.borrow();
        let retained_expert_count =
            if self.expert_memory_mode(contract, weights) == ExpertMemoryMode::Resident {
                // Natively resident experts live in the bound weights, which a
                // fully resident model carries without any paging plan; the
                // roster therefore comes from the normalized geometry.
                contract
                    .layers()
                    .iter()
                    .filter_map(|layer_descriptor| match layer_descriptor.feed_forward() {
                        LagunaFeedForwardDescriptor::Moe(moe_descriptor) => {
                            Some(u64::from(moe_descriptor.expert_count()))
                        }
                        LagunaFeedForwardDescriptor::Dense(_) => None,
                    })
                    .sum::<u64>()
            } else {
                let retained_cache_expert_count: u64 = self
                    .retained_layers
                    .borrow()
                    .as_ref()
                    .map_or(0, RetainedExpertPageCache::resident_expert_count)
                    as u64;
                if retained_cache_expert_count > 0 {
                    retained_cache_expert_count
                } else {
                    match last_forward {
                        // A streamed page that nothing retained is still the
                        // resident expert payload this forward materialized,
                        // which is the same grain the payload bytes report.
                        LagunaLastExpertForward::StreamedCompleteLayer { expert_count, .. }
                        | LagunaLastExpertForward::StreamedRoutedPage { expert_count, .. } => {
                            u64::from(expert_count)
                        }
                        LagunaLastExpertForward::None => 0,
                    }
                }
            };
        ExpertResidencyTelemetry {
            total_layer_count,
            resident_expert_count: u32::try_from(retained_expert_count).unwrap_or(u32::MAX),
            resident_expert_payload_bytes: statistics.resident_payload_byte_count,
        }
    }

    pub(super) fn expert_weight_memory_cache_statistics(
        &self,
        contract: &LagunaTargetContract,
        weights: &LagunaNativeWeights,
    ) -> ExpertWeightMemoryCacheStatistics {
        let mode = self.expert_memory_mode(contract, weights);
        let last_forward = *self.last_forward.borrow();
        let cache_statistics = self
            .retained_layers
            .borrow()
            .as_ref()
            .map(RetainedExpertPageCache::statistics)
            .unwrap_or_default();
        let (
            complete_layer_count,
            complete_layer_payload_byte_count,
            partial_layer_count,
            partial_layer_payload_byte_count,
        ) = match (mode, last_forward) {
            (ExpertMemoryMode::Resident, _) => {
                let complete_layer_count = resident_sparse_layer_count(contract, weights);
                let native_complete_layer_payload_byte_count = match self.paging_plan.as_ref() {
                    Some(plan) => {
                        resident_complete_payload_bytes(plan, contract, weights).unwrap_or(0)
                    }
                    // Without a paging plan every routed payload is the bound
                    // weight ownership itself.
                    None => weights.resident_routed_expert_payload_bytes(),
                };
                (
                    complete_layer_count.saturating_add(cache_statistics.complete_layer_count),
                    native_complete_layer_payload_byte_count
                        .saturating_add(cache_statistics.complete_layer_payload_byte_count),
                    0,
                    0,
                )
            }
            (ExpertMemoryMode::Hybrid, _) => (
                cache_statistics.complete_layer_count,
                cache_statistics.complete_layer_payload_byte_count,
                cache_statistics.partial_layer_count,
                cache_statistics.partial_layer_payload_byte_count,
            ),
            (
                _,
                LagunaLastExpertForward::StreamedCompleteLayer {
                    layer_count,
                    payload_bytes,
                    ..
                },
            ) => (layer_count as usize, payload_bytes, 0, 0),
            (
                _,
                LagunaLastExpertForward::StreamedRoutedPage {
                    layer_count,
                    payload_bytes,
                    ..
                },
            ) => (0, 0, layer_count as usize, payload_bytes),
            _ => (0, 0, 0, 0),
        };
        ExpertWeightMemoryCacheStatistics {
            entry_count: complete_layer_count.saturating_add(partial_layer_count),
            resident_payload_byte_count: complete_layer_payload_byte_count
                .saturating_add(partial_layer_payload_byte_count),
            maximum_resident_payload_byte_count: cache_statistics
                .maximum_resident_payload_byte_count
                .max(
                    complete_layer_payload_byte_count
                        .saturating_add(partial_layer_payload_byte_count),
                ),
            eviction_count: cache_statistics.eviction_count,
            disk_page_load_count: cache_statistics.disk_page_load_count,
            disk_batch_load_count: cache_statistics.disk_batch_load_count,
            complete_layer_count,
            complete_layer_payload_byte_count,
            partial_layer_count,
            partial_layer_payload_byte_count,
            mandatory_read_promotion_count: cache_statistics.mandatory_read_promotion_count,
            complete_layer_eviction_count: cache_statistics.complete_layer_eviction_count,
            partial_layer_eviction_count: cache_statistics.partial_layer_eviction_count,
        }
    }
}
