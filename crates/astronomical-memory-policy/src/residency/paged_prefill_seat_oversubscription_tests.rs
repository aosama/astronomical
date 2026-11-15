//! A reproduction test for the 2026-10-15 paged-prefill seat-oversubscription
//! incident (Ornith-1.5-35B-A3B 8-bit, 36 GB ceiling, 39-token prompt stuck at
//! 0% for 74 s; 91 s prefill at 0.43 tokens/s).
//!
//! Every constant below is taken verbatim from the Development worker log of
//! that session so the tests reproduce the policy arithmetic that produced the
//! incident, not a simplified stand-in:
//!
//! - 38 sparse decoder layers, each a complete 855,638,016-byte expert page
//!   (256 experts x 3,342,336 bytes, 8 experts routed per token).
//! - Composed Prefill leftover expert budget 28,974,982,400 bytes (36 GB
//!   ceiling minus model core 2,602,465,536, context reserve 1 GB, activation
//!   headroom 2,566,914,048, and one 855,638,016 stream slot).
//! - First capacity failure: active 35,208,958,986 bytes at failure, an
//!   855,638,016-byte complete-layer allocation rejected against the
//!   36,000,000,000-byte ceiling, 32,514,244,608 bytes of retained experts
//!   (38 seated complete layers, one whole-table eviction granularity), and a
//!   computed 64,597,002-byte reclamation target.
//!
//! What is pinned here is the plan the session opened with. The two other
//! mechanisms of the incident, a reclamation target smaller than the only
//! eviction granularity and a refresh floor that legalized any overshoot,
//! left the code with the memory governor (issue #2221): a step that hits
//! the limit now leases room and is run again, and the kept experts' one
//! capacity is always the remainder.

use crate::residency::ExpertLayerGeometry;
use crate::residency::{ExpertLayerState, plan_expert_residency};

/// Sparse decoder layers of the incident artifact (40-layer model, 2 dense).
const SPARSE_LAYER_COUNT: usize = 38;
/// One complete expert page of the incident artifact, bytes.
const COMPLETE_LAYER_PAYLOAD_BYTES: u64 = 855_638_016;
/// One expert of the incident artifact, bytes (8-bit affine, 2,048 hidden).
const EXPERT_PAYLOAD_BYTES: u64 = 3_342_336;
/// Routed experts per decoder layer of the incident artifact.
const EXPERT_CAPACITY: usize = 256;
/// Experts routed per token of the incident artifact.
const EXPERTS_PER_TOKEN: usize = 8;
/// Composed Prefill leftover expert budget from the incident worker log.
const COMPOSED_LEFTOVER_BUDGET_BYTES: u64 = 28_974_982_400;

fn incident_geometry() -> Vec<ExpertLayerGeometry> {
    (0..SPARSE_LAYER_COUNT)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: COMPLETE_LAYER_PAYLOAD_BYTES,
            expert_payload_bytes: EXPERT_PAYLOAD_BYTES,
            expert_capacity: EXPERT_CAPACITY,
            experts_per_token: EXPERTS_PER_TOKEN,
        })
        .collect()
}

#[test]
fn should_pin_exactly_the_affordable_complete_layers_for_the_incident_geometry() {
    // The opening plan pins exactly 33 complete layers: the routed floor for all
    // 38 layers plus 33 incremental complete payloads fits the composed budget,
    // and a 34th does not. This is the plan the incident session published
    // (`complete_layer_target_count=33 pinned_complete_layer_count=33`).
    let plan = plan_expert_residency(COMPOSED_LEFTOVER_BUDGET_BYTES, &incident_geometry(), &[])
        .expect("the incident geometry should plan against the incident budget");

    assert_eq!(plan.complete_layer_targets.len(), 33);
    assert_eq!(
        plan.complete_layer_targets,
        (0..33).collect::<Vec<_>>(),
        "uniform incremental cost must pin the lowest 33 decoder indexes"
    );
    assert_eq!(plan.layer_states[33], ExpertLayerState::Hot);
    assert_eq!(plan.layer_states[37], ExpertLayerState::Hot);
}
