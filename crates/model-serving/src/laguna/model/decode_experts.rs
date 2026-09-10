//! Decode-path expert residency: handoff from prefill pages, admit, stack.

use astronomical_runtime_integration::MlxRuntime;

use crate::laguna::paging::{LagunaExpertWeightPage, LagunaResidentExpert};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::error::LagunaExecutionError;
use super::expert_residency::LagunaExpertResidencyState;

/// Moves one prefill-retained page into the decode cache as individual experts.
pub(super) fn handoff_prefill_page_into_decode_cache(
    residency: &LagunaExpertResidencyState,
    runtime: &MlxRuntime,
    paging_slot_index: usize,
    bytes_per_expert: u64,
) -> Result<(), LagunaExecutionError> {
    let Some(prefill_page) = residency.take_prefill_page(paging_slot_index) else {
        return Ok(());
    };
    let handed_off_ids = prefill_page.manifest().expert_ids.clone();
    let resident_experts = prefill_page.split_resident_experts(runtime, bytes_per_expert)?;
    if let Some(decode_cache) = residency.decode_cache.borrow_mut().as_mut() {
        for (expert_id, expert_weight) in resident_experts {
            decode_cache.admit(paging_slot_index, expert_id, expert_weight);
        }
        let protected = decode_cache
            .resident_ids(paging_slot_index)
            .into_iter()
            .map(|expert_id| (paging_slot_index, expert_id))
            .collect::<Vec<_>>();
        let _ = decode_cache.enforce_ceiling(&protected);
    }
    store_gather_view(residency, paging_slot_index, handed_off_ids, prefill_page);
    Ok(())
}

/// Admits missing experts and drops a stale gather view when membership changes.
pub(super) fn admit_decode_experts(
    residency: &LagunaExpertResidencyState,
    paging_slot_index: usize,
    routed_ids: &[usize],
    loaded_experts: Vec<(usize, LagunaResidentExpert)>,
) {
    let protected = routed_ids
        .iter()
        .map(|expert_id| (paging_slot_index, *expert_id))
        .collect::<Vec<_>>();
    let admitted_count = loaded_experts.len();
    let mut vacated_slots = Vec::new();
    if let Some(decode_cache) = residency.decode_cache.borrow_mut().as_mut() {
        if admitted_count > 0 {
            decode_cache.record_disk_load(admitted_count, 1);
        }
        for (expert_id, expert_weight) in loaded_experts {
            decode_cache.admit(paging_slot_index, expert_id, expert_weight);
        }
        decode_cache.record_demand(paging_slot_index, routed_ids);
        vacated_slots = decode_cache.enforce_ceiling(&protected);
    }
    if admitted_count > 0 && !vacated_slots.contains(&paging_slot_index) {
        vacated_slots.push(paging_slot_index);
    }
    for vacated_slot in vacated_slots {
        clear_gather_view(residency, vacated_slot);
    }
}

/// Rebuilds the gather view when the resident expert set changed.
pub(super) fn ensure_stacked_decode_page(
    residency: &LagunaExpertResidencyState,
    runtime: &MlxRuntime,
    paging_slot_index: usize,
    expert_capacity: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(), LagunaExecutionError> {
    let resident_ids = residency
        .decode_cache
        .borrow()
        .as_ref()
        .map(|decode_cache| decode_cache.resident_ids(paging_slot_index))
        .unwrap_or_default();
    let gather_ids_match = residency
        .decode_gather_ids
        .borrow()
        .get(paging_slot_index)
        .is_some_and(|gather_ids| *gather_ids == resident_ids);
    if !gather_ids_match || gather_page_missing(residency, paging_slot_index) {
        let stacked_page = rebuild_stacked_page(
            residency,
            runtime,
            paging_slot_index,
            expert_capacity,
            performance_attribution,
        )?;
        store_gather_view(residency, paging_slot_index, resident_ids, stacked_page);
    }
    Ok(())
}

/// Borrows the current gatherable stacked page for one sparse slot.
pub(super) fn with_stacked_decode_page<Output>(
    residency: &LagunaExpertResidencyState,
    paging_slot_index: usize,
    execute_on_page: impl FnOnce(&LagunaExpertWeightPage) -> Result<Output, LagunaExecutionError>,
) -> Result<Output, LagunaExecutionError> {
    let gather_pages = residency.decode_gather_pages.borrow();
    let stacked_page = gather_pages
        .get(paging_slot_index)
        .and_then(|gather_page| gather_page.as_ref())
        .ok_or_else(|| {
            LagunaExecutionError::invalid_geometry(
                "decode stacking produced no gatherable expert page",
            )
        })?;
    execute_on_page(stacked_page)
}

fn gather_page_missing(residency: &LagunaExpertResidencyState, paging_slot_index: usize) -> bool {
    residency
        .decode_gather_pages
        .borrow()
        .get(paging_slot_index)
        .is_none_or(Option::is_none)
}

fn rebuild_stacked_page(
    residency: &LagunaExpertResidencyState,
    runtime: &MlxRuntime,
    paging_slot_index: usize,
    expert_capacity: usize,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<LagunaExpertWeightPage, LagunaExecutionError> {
    performance_attribution.measure_operation(
        PerformanceOperation::PagedMoeGraphConstruction,
        |_| {
            let decode_cache = residency.decode_cache.borrow();
            let Some(decode_cache) = decode_cache.as_ref() else {
                return Err(LagunaExecutionError::invalid_geometry(
                    "decode stacking requires an attached decode expert cache",
                ));
            };
            let resident_weights = decode_cache.resident_weights(paging_slot_index);
            LagunaExpertWeightPage::stack_resident_experts(
                runtime,
                expert_capacity,
                &resident_weights,
            )
            .map_err(|_| {
                LagunaExecutionError::invalid_geometry(
                    "decode stacking failed to concatenate resident experts",
                )
            })
        },
    )
}

fn store_gather_view(
    residency: &LagunaExpertResidencyState,
    paging_slot_index: usize,
    expert_ids: Vec<usize>,
    stacked_page: LagunaExpertWeightPage,
) {
    if let Some(gather_ids) = residency
        .decode_gather_ids
        .borrow_mut()
        .get_mut(paging_slot_index)
    {
        *gather_ids = expert_ids;
    }
    if let Some(gather_page) = residency
        .decode_gather_pages
        .borrow_mut()
        .get_mut(paging_slot_index)
    {
        *gather_page = Some(stacked_page);
    }
}

fn clear_gather_view(residency: &LagunaExpertResidencyState, paging_slot_index: usize) {
    if let Some(gather_ids) = residency
        .decode_gather_ids
        .borrow_mut()
        .get_mut(paging_slot_index)
    {
        gather_ids.clear();
    }
    if let Some(gather_page) = residency
        .decode_gather_pages
        .borrow_mut()
        .get_mut(paging_slot_index)
    {
        *gather_page = None;
    }
}
