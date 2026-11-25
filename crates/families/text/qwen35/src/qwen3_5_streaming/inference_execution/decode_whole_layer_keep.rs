//! How many layers a reply keeps whole, and giving up the rest.
//!
//! A layer held whole keeps its route on the graphics processor. Every other
//! layer serves its experts from a table, which costs a lookup per token:
//! inside the token's graph when the model looks routes up there, otherwise
//! a stop that ends the evaluation. Holding a layer whole costs all of its
//! experts, though, and replays of recorded replies at half the model's size
//! (issue #2338) showed that keeping every layer the prompt left whole starved
//! the other layers' tables to 16 experts and missed 71 to 109 experts per
//! token, against 18 to 34 with only a few layers whole and fair tables for
//! the rest. `decode_whole_layer_keep_count` sets the number, protecting a
//! table size that scales with the layer's expert count; the layers kept
//! are the ones the requests routed to most.

use ipc_protocol::RequestId;

use crate::RetainedExpertPageClass;
use crate::qwen3_5_streaming::model::Qwen3_5Model;

/// How many complete layers this reply may keep whole under the capacity
/// the memory governor gave the `astro_expert_cache`.
pub(super) fn whole_layer_keep_count(model: &Qwen3_5Model) -> usize {
    let Some(expert_pager) = model.expert_pager.as_ref() else {
        return 0;
    };
    let Some(first_layer_plan) = expert_pager.layer_plans().first() else {
        return 0;
    };
    crate::decode_whole_layer_keep_count(
        model.kept_expert_capacity_bytes(),
        mean_expert_payload_bytes(expert_pager),
        first_layer_plan.expert_capacity,
        expert_pager.layer_count(),
    )
}

/// One expert's payload bytes averaged over every sparse layer.
///
/// A model quantized with mixed precision stores some layers' experts at 8
/// bits and others at 4: the 4-bit OptiQ 35B-A3B's first layer holds 3.19 MB
/// experts against a mean of 1.82 MB. Pricing every table by the first layer
/// sized them to 62 experts under an 11 GB limit and left about 3.2 GB of the
/// `astro_expert_cache` capacity unused (issue #2338). The mean prices the
/// room the tables share as it is spent.
pub(super) fn mean_expert_payload_bytes(
    expert_pager: &crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5ExpertPager,
) -> u64 {
    let mut total_payload_bytes = 0_u64;
    let mut total_expert_count = 0_u64;
    for layer_plan in expert_pager.layer_plans() {
        let Ok(layer_payload_bytes) =
            crate::qwen3_5_streaming::expert_streaming::retained_expert_layer_page_payload_bytes(
                layer_plan,
                layer_plan.expert_capacity,
            )
        else {
            continue;
        };
        total_payload_bytes = total_payload_bytes.saturating_add(layer_payload_bytes);
        total_expert_count = total_expert_count
            .saturating_add(u64::try_from(layer_plan.expert_capacity).unwrap_or(0));
    }
    total_payload_bytes
        .checked_div(total_expert_count)
        .unwrap_or(0)
}

/// The planned complete layers to seat, cut to what the keep count leaves
/// after the layers already held whole: seating a layer only to give it up
/// would read it from storage for nothing.
pub(super) fn cap_seating_to_whole_layer_keep_count(
    model: &Qwen3_5Model,
    complete_layer_indexes_to_seat: &[usize],
) -> Vec<usize> {
    let held_whole_count = held_whole_layer_count(model);
    let seatable_count = whole_layer_keep_count(model).saturating_sub(held_whole_count);
    complete_layer_indexes_to_seat
        .iter()
        .copied()
        .take(seatable_count)
        .collect()
}

/// Gives up the complete layers beyond the keep count, the least routed to
/// first, so their room goes to the other layers' tables.
pub(super) fn release_complete_layers_beyond_keep_count(
    request_id: RequestId,
    model: &Qwen3_5Model,
) {
    let keep_count = whole_layer_keep_count(model);
    let layer_residencies = model.current_expert_layer_residencies();
    let released_layer_indexes =
        crate::complete_layers_beyond_keep_count(&layer_residencies, keep_count);
    let mut released_layer_count = 0;
    for layer_index in released_layer_indexes {
        if model.remove_retained_expert_layer(layer_index) {
            released_layer_count += 1;
        }
    }
    if released_layer_count > 0 {
        tracing::info!(
            request_id = request_id.value(),
            keep_count,
            released_layer_count,
            "gave up complete layers beyond the reply's whole-layer keep count"
        );
    }
}

fn held_whole_layer_count(model: &Qwen3_5Model) -> usize {
    let mut held_whole_count = 0;
    for layer_residency in model.current_expert_layer_residencies() {
        if layer_residency.class == RetainedExpertPageClass::StableCompleteLayer {
            held_whole_count += 1;
        }
    }
    held_whole_count
}
