//! Compact Machine Learning framework for Apple silicon arrays for one split expert route.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use crate::expert_paging::{ExpertPageRoutePartition, QuantizedExpertPageManifest};

const BUILD_SPLIT_PAGE_ROUTE_OPERATION: &str = "build Qwen3.5-MoE split expert page route";
const REMAP_EXPERT_PAGE_SLOTS_OPERATION: &str = "remap Qwen3.5-MoE expert page slots";

/// Assignment arrays consumed by retained-page and missing-page expert projections.
pub struct Qwen3_5MoESplitPageRoute {
    pub retained_page_slot_indices: MlxArray,
    pub retained_scores: MlxArray,
    pub missing_page_slot_indices: MlxArray,
    pub missing_scores: MlxArray,
}

impl Qwen3_5MoESplitPageRoute {
    /// Builds disjoint compact arrays without copying selected route values to the host.
    pub fn build(
        runtime: &MlxRuntime,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        route_partition: &ExpertPageRoutePartition,
        retained_page_manifest: &QuantizedExpertPageManifest,
        missing_page_manifest: &QuantizedExpertPageManifest,
    ) -> Result<Self, MlxRuntimeError> {
        let assignment_count =
            validate_split_page_route(selected_indices, selected_scores, route_partition)?;
        let retained_assignment_positions =
            assignment_position_array(runtime, &route_partition.retained_assignment_positions)?;
        let missing_assignment_positions =
            assignment_position_array(runtime, &route_partition.missing_assignment_positions)?;
        let flattened_selected_indices = runtime.reshape(selected_indices, &[assignment_count])?;
        let flattened_selected_scores = runtime.reshape(selected_scores, &[assignment_count])?;
        let retained_expert_ids = runtime.take_axis(
            &flattened_selected_indices,
            &retained_assignment_positions,
            0,
        )?;
        let retained_scores = runtime.take_axis(
            &flattened_selected_scores,
            &retained_assignment_positions,
            0,
        )?;
        let missing_expert_ids = runtime.take_axis(
            &flattened_selected_indices,
            &missing_assignment_positions,
            0,
        )?;
        let missing_scores =
            runtime.take_axis(&flattened_selected_scores, &missing_assignment_positions, 0)?;
        let retained_page_slot_indices = qwen3_5_moe_remap_expert_page_slots(
            runtime,
            &retained_expert_ids,
            &route_partition.retained_expert_ids,
            retained_page_manifest,
        )?;
        let missing_page_slot_indices = qwen3_5_moe_remap_expert_page_slots(
            runtime,
            &missing_expert_ids,
            &route_partition.missing_expert_ids,
            missing_page_manifest,
        )?;
        Ok(Self {
            retained_page_slot_indices,
            retained_scores,
            missing_page_slot_indices,
            missing_scores,
        })
    }
}

pub(super) fn qwen3_5_moe_remap_expert_page_slots(
    runtime: &MlxRuntime,
    selected_indices: &MlxArray,
    sorted_unique_expert_ids: &[usize],
    page_manifest: &QuantizedExpertPageManifest,
) -> Result<MlxArray, MlxRuntimeError> {
    if sorted_unique_expert_ids.iter().any(|expert_id| {
        page_manifest
            .page_slot_by_global_expert_id
            .get(*expert_id)
            .is_none_or(|page_slot| *page_slot == u32::MAX)
    }) {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: REMAP_EXPERT_PAGE_SLOTS_OPERATION,
            description: "a routed expert is absent from the streamed page manifest".to_owned(),
        });
    }
    let expert_capacity = i32::try_from(page_manifest.page_slot_by_global_expert_id.len())
        .map_err(|_| MlxRuntimeError::RuntimeOperation {
            operation: REMAP_EXPERT_PAGE_SLOTS_OPERATION,
            description: "expert capacity exceeds the MLX shape range".to_owned(),
        })?;
    let page_slots = runtime.array_from_u32(
        &page_manifest.page_slot_by_global_expert_id,
        &[expert_capacity],
    )?;
    runtime.take_axis(&page_slots, selected_indices, 0)
}

fn validate_split_page_route(
    selected_indices: &MlxArray,
    selected_scores: &MlxArray,
    route_partition: &ExpertPageRoutePartition,
) -> Result<i32, MlxRuntimeError> {
    let assignment_count = selected_indices.element_count();
    if assignment_count == 0
        || selected_scores.element_count() != assignment_count
        || route_partition.retained_assignment_positions.is_empty()
        || route_partition.missing_assignment_positions.is_empty()
        || route_partition.retained_assignment_positions.len()
            + route_partition.missing_assignment_positions.len()
            != assignment_count
    {
        return Err(split_page_route_error(
            "selected indices, scores, and non-empty route sides must cover the same assignments",
        ));
    }

    // Validate exact disjoint coverage before constructing lazy take operations.
    // This prevents a malformed partition from executing one top-K assignment twice.
    let mut is_assignment_position_covered = vec![false; assignment_count];
    for assignment_position in route_partition
        .retained_assignment_positions
        .iter()
        .chain(&route_partition.missing_assignment_positions)
    {
        let Some(is_covered) = is_assignment_position_covered.get_mut(*assignment_position) else {
            return Err(split_page_route_error(
                "a route assignment position exceeds the selected assignment count",
            ));
        };
        if *is_covered {
            return Err(split_page_route_error(
                "a route assignment position appears more than once",
            ));
        }
        *is_covered = true;
    }
    if is_assignment_position_covered
        .iter()
        .any(|is_covered| !is_covered)
    {
        return Err(split_page_route_error(
            "the split route does not cover every selected assignment",
        ));
    }

    i32::try_from(assignment_count)
        .map_err(|_| split_page_route_error("expert assignment count exceeds the MLX shape range"))
}

fn assignment_position_array(
    runtime: &MlxRuntime,
    assignment_positions: &[usize],
) -> Result<MlxArray, MlxRuntimeError> {
    let assignment_positions = assignment_positions
        .iter()
        .copied()
        .map(|assignment_position| {
            u32::try_from(assignment_position)
                .map_err(|_| split_page_route_error("expert assignment position exceeds u32"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let compact_assignment_count = i32::try_from(assignment_positions.len()).map_err(|_| {
        split_page_route_error("compact assignment count exceeds the MLX shape range")
    })?;
    runtime.array_from_u32(&assignment_positions, &[compact_assignment_count])
}

fn split_page_route_error(description: &str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: BUILD_SPLIT_PAGE_ROUTE_OPERATION,
        description: description.to_owned(),
    }
}
