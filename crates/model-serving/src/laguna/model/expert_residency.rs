//! Laguna-owned expert residency status: resident, complete-layer, or routed.

use std::cell::{Cell, Ref, RefCell};

use crate::expert_paging::{ExpertWeightPage, RetainedExpertPageCache, RetainedExpertReclamation};
use crate::laguna::paging::{LagunaExpertPagingPlan, LagunaExpertWeightPage, LagunaResidentExpert};
use crate::memory::DecodeExpertCache;
use crate::memory::{
    ExpertResidencyPlan, MemoryPhase, RequestExpertResidency,
    publish_request_stable_residency_plan, should_commit_mandatory_complete_layer,
    should_commit_mandatory_routed_page,
};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::error::LagunaExecutionError;

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
    pub(super) paging_plan: Option<LagunaExpertPagingPlan>,
    active_plan: RefCell<Option<ExpertResidencyPlan>>,
    request_residency: RefCell<Option<RequestExpertResidency>>,
    pub(super) last_forward: RefCell<LagunaLastExpertForward>,
    pub(super) retained_layers: RefCell<Option<RetainedExpertPageCache<LagunaExpertWeightPage>>>,
    retained_expert_ceiling_bytes: Cell<u64>,
    pub(super) decode_cache: RefCell<Option<DecodeExpertCache<LagunaResidentExpert>>>,
    pub(super) decode_gather_pages: RefCell<Vec<Option<LagunaExpertWeightPage>>>,
    pub(super) decode_gather_ids: RefCell<Vec<Vec<usize>>>,
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
            decode_cache: RefCell::new(None),
            decode_gather_pages: RefCell::new(Vec::new()),
            decode_gather_ids: RefCell::new(Vec::new()),
        }
    }

    pub(super) fn attach_paging_plan(&mut self, paging_plan: LagunaExpertPagingPlan) {
        let sparse_layer_count = paging_plan.sparse_layers().len();
        self.paging_plan = Some(paging_plan);
        let mut retained_layers = RetainedExpertPageCache::new(sparse_layer_count);
        retained_layers.update_maximum_resident_payload_bytes(0);
        self.retained_layers.replace(Some(retained_layers));
        self.retained_expert_ceiling_bytes.set(0);
        let mut decode_cache = DecodeExpertCache::new(sparse_layer_count);
        decode_cache.set_ceiling(0);
        self.decode_cache.replace(Some(decode_cache));
        self.decode_gather_pages
            .replace((0..sparse_layer_count).map(|_| None).collect());
        self.decode_gather_ids
            .replace(vec![Vec::new(); sparse_layer_count]);
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
        if let Some(decode_cache) = self.decode_cache.borrow_mut().as_mut() {
            decode_cache.set_ceiling(retained_expert_ceiling_bytes);
        }
        self.invalidate_decode_gather_views();
        self.last_forward.replace(LagunaLastExpertForward::None);
        Ok(())
    }

    pub(super) fn invalidate_decode_gather_views(&self) {
        for gather_page in self.decode_gather_pages.borrow_mut().iter_mut() {
            *gather_page = None;
        }
        for gather_ids in self.decode_gather_ids.borrow_mut().iter_mut() {
            gather_ids.clear();
        }
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

    pub(super) fn refresh_explicit_phase_plan(&self, phase: MemoryPhase) {
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
                if phase == MemoryPhase::GenerationPreparation
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

    pub(super) fn take_prefill_page(
        &self,
        paging_slot_index: usize,
    ) -> Option<LagunaExpertWeightPage> {
        self.retained_layers
            .borrow_mut()
            .as_mut()
            .and_then(|retained_layers| retained_layers.take_retained_layer(paging_slot_index))
    }

    pub(super) fn reclaim_for_request_pressure(
        &self,
        required_reclamation_bytes: u64,
    ) -> RetainedExpertReclamation {
        if let Some(decode_cache) = self.decode_cache.borrow_mut().as_mut() {
            let decode_payload = decode_cache.total_payload_bytes();
            if decode_payload > 0 {
                let remaining = decode_payload.saturating_sub(required_reclamation_bytes);
                decode_cache.set_ceiling(remaining);
                self.invalidate_decode_gather_views();
            }
        }
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

    pub(super) fn active_plan(&self) -> Ref<'_, Option<ExpertResidencyPlan>> {
        self.active_plan.borrow()
    }

    pub(super) fn retained_expert_ceiling_bytes(&self) -> u64 {
        self.retained_expert_ceiling_bytes.get()
    }
}
