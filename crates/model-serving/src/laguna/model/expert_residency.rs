//! Laguna-owned expert residency status: resident, complete-layer, or routed.

use std::cell::{Cell, Ref, RefCell};

use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::ExpertResidencyTelemetry;
use crate::expert_paging::{
    ExpertLayerResidencyTarget, ExpertResidencyPhase, ExpertWeightMemoryCacheStatistics,
    ExpertWeightPage, PhaseAwareExpertResidencyPlan, RetainedExpertLayerCache,
};
use crate::laguna::normalization::{LagunaFeedForwardDescriptor, LagunaTargetContract};
use crate::laguna::paging::{LagunaExpertPagingPlan, LagunaExpertWeightPage};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::error::LagunaExecutionError;
use super::weights::LagunaNativeWeights;

/// Last sparse-expert grain executed by the model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::laguna) enum LagunaLastExpertForward {
    #[default]
    None,
    StreamedCompleteLayer {
        layer_count: u32,
        payload_bytes: u64,
    },
    StreamedRoutedPage {
        layer_count: u32,
        payload_bytes: u64,
    },
}

/// Phase-aware plan plus last-forward grain used by Laguna status.
pub(super) struct LagunaExpertResidencyState {
    paging_plan: Option<LagunaExpertPagingPlan>,
    active_plan: RefCell<Option<PhaseAwareExpertResidencyPlan>>,
    last_forward: RefCell<LagunaLastExpertForward>,
    disk_page_load_count: Cell<u64>,
    retained_layers: RefCell<Option<RetainedExpertLayerCache<LagunaExpertWeightPage>>>,
    retained_expert_ceiling_bytes: Cell<u64>,
}

impl LagunaExpertResidencyState {
    pub(super) fn new() -> Self {
        Self {
            paging_plan: None,
            active_plan: RefCell::new(None),
            last_forward: RefCell::new(LagunaLastExpertForward::None),
            disk_page_load_count: Cell::new(0),
            retained_layers: RefCell::new(None),
            retained_expert_ceiling_bytes: Cell::new(0),
        }
    }

    pub(super) fn attach_paging_plan(&mut self, paging_plan: LagunaExpertPagingPlan) {
        let sparse_layer_count = paging_plan.sparse_layers().len();
        self.paging_plan = Some(paging_plan);
        let mut retained_layers = RetainedExpertLayerCache::new(sparse_layer_count);
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

    pub(super) fn record_disk_page_load(&self) {
        self.disk_page_load_count
            .set(self.disk_page_load_count.get().saturating_add(1));
    }

    pub(super) fn refresh_phase_plan(&self, token_count: usize) {
        let Some(paging_plan) = self.paging_plan.as_ref() else {
            self.active_plan.replace(None);
            return;
        };
        let phase = if token_count > 1 {
            ExpertResidencyPhase::Prefill
        } else {
            ExpertResidencyPhase::Decode
        };
        let current_residencies = self
            .retained_layers
            .borrow()
            .as_ref()
            .map(|retained_layers| retained_layers.topology_snapshot())
            .unwrap_or_default();
        match paging_plan.plan_phase_aware_residency(
            phase,
            self.retained_expert_ceiling_bytes.get(),
            &current_residencies,
        ) {
            Ok(active_plan) => {
                self.active_plan.replace(Some(active_plan));
            }
            Err(_) => {
                self.active_plan.replace(None);
            }
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
        let should_promote = self.active_plan().as_ref().is_some_and(|active_plan| {
            active_plan.layer_targets.get(paging_slot_index)
                == Some(&ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead)
        });
        if !should_promote {
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
        let should_admit = self.active_plan().as_ref().is_some_and(|active_plan| {
            matches!(
                active_plan.layer_targets.get(paging_slot_index),
                Some(
                    ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
                        | ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
                )
            )
        });
        if !should_admit {
            return Ok(());
        }
        let mut retained_layers = self.retained_layers.borrow_mut();
        let Some(retained_layers) = retained_layers.as_mut() else {
            return Ok(());
        };
        if retained_layers.retained_layer(paging_slot_index).is_some() {
            return Ok(());
        }
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
        if sparse_layer_count == 0 || resident_layer_count == sparse_layer_count {
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
        ExpertResidencyTelemetry {
            total_layer_count,
            complete_layer_count: u32::try_from(statistics.complete_layer_count)
                .unwrap_or(u32::MAX),
            complete_layer_payload_bytes: statistics.complete_layer_payload_byte_count,
            partial_layer_count: u32::try_from(statistics.partial_layer_count).unwrap_or(u32::MAX),
            partial_layer_payload_bytes: statistics.partial_layer_payload_byte_count,
        }
    }

    pub(super) fn expert_weight_memory_cache_statistics(
        &self,
        contract: &LagunaTargetContract,
        weights: &LagunaNativeWeights,
    ) -> ExpertWeightMemoryCacheStatistics {
        let mode = self.expert_memory_mode(contract, weights);
        let last_forward = *self.last_forward.borrow();
        let (
            complete_layer_count,
            complete_layer_payload_byte_count,
            partial_layer_count,
            partial_layer_payload_byte_count,
        ) = match (mode, last_forward) {
            (ExpertMemoryMode::Resident, _) => {
                let complete_layer_count = resident_sparse_layer_count(contract, weights);
                let complete_layer_payload_byte_count = self
                    .paging_plan
                    .as_ref()
                    .and_then(|plan| resident_complete_payload_bytes(plan, contract, weights))
                    .unwrap_or(0);
                (
                    complete_layer_count,
                    complete_layer_payload_byte_count,
                    0,
                    0,
                )
            }
            (ExpertMemoryMode::Hybrid, _) => {
                let cache_statistics = self
                    .retained_layers
                    .borrow()
                    .as_ref()
                    .map(RetainedExpertLayerCache::statistics)
                    .unwrap_or_default();
                (
                    cache_statistics.complete_layer_count,
                    cache_statistics.complete_layer_payload_byte_count,
                    cache_statistics.partial_layer_count,
                    cache_statistics.partial_layer_payload_byte_count,
                )
            }
            (
                _,
                LagunaLastExpertForward::StreamedCompleteLayer {
                    layer_count,
                    payload_bytes,
                },
            ) => (layer_count as usize, payload_bytes, 0, 0),
            (
                _,
                LagunaLastExpertForward::StreamedRoutedPage {
                    layer_count,
                    payload_bytes,
                },
            ) => (0, 0, layer_count as usize, payload_bytes),
            _ => (0, 0, 0, 0),
        };
        ExpertWeightMemoryCacheStatistics {
            entry_count: complete_layer_count.saturating_add(partial_layer_count),
            resident_payload_byte_count: complete_layer_payload_byte_count
                .saturating_add(partial_layer_payload_byte_count),
            maximum_resident_payload_byte_count: complete_layer_payload_byte_count
                .saturating_add(partial_layer_payload_byte_count),
            eviction_count: 0,
            disk_page_load_count: self.disk_page_load_count.get(),
            disk_batch_load_count: self.disk_page_load_count.get(),
            complete_layer_count,
            complete_layer_payload_byte_count,
            partial_layer_count,
            partial_layer_payload_byte_count,
            mandatory_read_promotion_count: 0,
            complete_layer_eviction_count: 0,
            partial_layer_eviction_count: 0,
        }
    }
}

/// Confirms every sparse layer either owns stacked experts or appears in the plan.
pub(super) fn validate_sparse_coverage(
    contract: &LagunaTargetContract,
    weights: &LagunaNativeWeights,
    paging_plan: Option<&LagunaExpertPagingPlan>,
) -> Result<(), LagunaExecutionError> {
    for layer_descriptor in contract.layers() {
        if !matches!(
            layer_descriptor.feed_forward(),
            LagunaFeedForwardDescriptor::Moe(_)
        ) {
            continue;
        }
        let layer_index = layer_descriptor.layer_index();
        if weights.has_routed_experts(layer_index) {
            continue;
        }
        let Some(paging_plan) = paging_plan else {
            return Err(LagunaExecutionError::invalid_geometry(
                "a sparse Laguna layer without resident routed experts requires a paging plan",
            ));
        };
        if paging_plan.sparse_layer_for_decoder(layer_index).is_none() {
            return Err(LagunaExecutionError::invalid_geometry(
                "a sparse Laguna layer is missing from the paging plan",
            ));
        }
    }
    Ok(())
}

fn sparse_layer_count(contract: &LagunaTargetContract) -> usize {
    contract
        .layers()
        .iter()
        .filter(|layer_descriptor| {
            matches!(
                layer_descriptor.feed_forward(),
                LagunaFeedForwardDescriptor::Moe(_)
            )
        })
        .count()
}

fn sparse_layer_counts(
    contract: &LagunaTargetContract,
    weights: &LagunaNativeWeights,
) -> (usize, usize) {
    let mut sparse_count = 0_usize;
    let mut resident_count = 0_usize;
    for layer_descriptor in contract.layers() {
        if !matches!(
            layer_descriptor.feed_forward(),
            LagunaFeedForwardDescriptor::Moe(_)
        ) {
            continue;
        }
        sparse_count = sparse_count.saturating_add(1);
        if weights.has_routed_experts(layer_descriptor.layer_index()) {
            resident_count = resident_count.saturating_add(1);
        }
    }
    (sparse_count, resident_count)
}

fn resident_sparse_layer_count(
    contract: &LagunaTargetContract,
    weights: &LagunaNativeWeights,
) -> usize {
    sparse_layer_counts(contract, weights).1
}

fn resident_complete_payload_bytes(
    paging_plan: &LagunaExpertPagingPlan,
    contract: &LagunaTargetContract,
    weights: &LagunaNativeWeights,
) -> Option<u64> {
    let mut complete_payload_bytes = 0_u64;
    for layer_descriptor in contract.layers() {
        if !matches!(
            layer_descriptor.feed_forward(),
            LagunaFeedForwardDescriptor::Moe(_)
        ) || !weights.has_routed_experts(layer_descriptor.layer_index())
        {
            continue;
        }
        let sparse_layer = paging_plan.sparse_layer_for_decoder(layer_descriptor.layer_index())?;
        complete_payload_bytes = complete_payload_bytes
            .checked_add(sparse_layer.complete_layer_payload_byte_count().ok()?)?;
    }
    Some(complete_payload_bytes)
}
