//! Reproduction tests for the 2026-10-15 paged-prefill seat-oversubscription
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
//! The incident mechanism these tests pin, from the bottom of the call stack
//! upward:
//!
//! 1. The request-stable plan authorizes complete-layer adoption offers for
//!    streamed-role layers (commit gate) while budgeting zero new retained
//!    bytes for them: the offer side and the budget side disagree.
//! 2. The recovery formula computes a sub-table-granularity reclamation target
//!    (64,597,002 bytes) while the only eviction granularity is one whole
//!    855,638,016-byte table. At the time the paged profile also never retried
//!    the same chunk, so every failure halved the chunk and re-streamed at
//!    least one complete layer from SSD; the policy now authorizes exactly one
//!    unchanged retry once a whole-table eviction satisfied the target, and the
//!    test below pins that remediation.
//! 3. The budget-refresh floor legalizes any overshoot forever: once resident
//!    bytes exceed the composed budget, every later publication floors the
//!    cache ceiling at the overshoot, and the composed budget never reasserts
//!    itself for the rest of the request.

use crate::budget::adaptive_growth::AdaptiveRamGrowthExecutionProfile;
use crate::phase::MemoryPhase;
use crate::reclamation::whole_table_reclamation_target_bytes;
use crate::recovery::ForwardRecoveryPolicy;
use crate::residency::ExpertLayerGeometry;
use crate::residency::{
    ExpertLayerResidencyTarget, plan_expert_residency, publish_request_stable_residency_plan,
    retained_resident_ceiling_after_budget_refresh, should_commit_mandatory_complete_layer,
};

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
/// Configured active-memory ceiling of the incident session, bytes.
const ACTIVE_MEMORY_CEILING_BYTES: usize = 36_000_000_000;
/// Active memory sampled before the first capacity recovery, bytes.
const ACTIVE_MEMORY_BEFORE_RECLAMATION_BYTES: usize = 35_116_715_662;
/// Active memory at the first failed allocation, bytes.
const ACTIVE_MEMORY_AT_FAILURE_BYTES: usize = 35_208_958_986;
/// The rejected complete-layer allocation, bytes.
const ATTEMPTED_ALLOCATION_BYTES: usize = 855_638_016;
/// Retained expert payload at the first recovery (38 seated layers), bytes.
const RETAINED_PAYLOAD_AT_FIRST_RECOVERY_BYTES: usize = 32_514_244_608;
/// Retained expert payload after one whole-table eviction (37 layers), bytes.
const RETAINED_PAYLOAD_AFTER_ONE_TABLE_EVICTION_BYTES: usize = 31_658_606_592;
/// The sub-table reclamation target the recovery computed, bytes.
const FIRST_RECOVERY_RECLAMATION_TARGET_BYTES: usize = 64_597_002;
/// The fixed workspace the recovery reconstructed, bytes.
const FIRST_RECOVERY_FIXED_WORKSPACE_BYTES: usize = 947_881_340;

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
    let plan = plan_expert_residency(
        MemoryPhase::Prefill,
        COMPOSED_LEFTOVER_BUDGET_BYTES,
        &incident_geometry(),
        &[],
    )
    .expect("the incident geometry should plan against the incident budget");

    assert_eq!(plan.complete_layer_targets.len(), 33);
    assert_eq!(
        plan.complete_layer_targets,
        (0..33).collect::<Vec<_>>(),
        "uniform incremental cost must pin the lowest 33 decoder indexes"
    );
    assert_eq!(
        plan.layer_targets[33],
        ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
    );
    assert_eq!(
        plan.layer_targets[37],
        ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
    );
}

#[test]
fn should_not_authorize_admission_offers_beyond_the_composed_leftover_budget() {
    // Before the fix (issue #1278), the stabilized request plan mapped every
    // streamed-role layer to AdmitCompleteOnMandatoryRead while budgeting
    // zero new retained bytes: the commit gate then authorized complete-layer
    // adoption offers whose 855,638,016-byte pages the composed budget could
    // never seat. The plan must resolve a streamed layer to admit-on-read
    // only while the leftover budget can actually seat its complete page.
    let opening_plan = plan_expert_residency(
        MemoryPhase::Prefill,
        COMPOSED_LEFTOVER_BUDGET_BYTES,
        &incident_geometry(),
        &[],
    )
    .expect("the incident geometry should plan against the incident budget");
    let (request_residency, stabilized_plan) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        None,
        opening_plan,
        &[],
        0,
        &incident_geometry(),
    );

    let request_residency =
        request_residency.expect("Prefill publication must freeze a request contract");
    assert_eq!(request_residency.pinned_complete_layer_indexes().len(), 33);
    for (layer_index, layer_target) in stabilized_plan
        .layer_targets
        .iter()
        .enumerate()
        .take(SPARSE_LAYER_COUNT)
        .skip(33)
    {
        assert_eq!(
            *layer_target,
            ExpertLayerResidencyTarget::StreamOperationLocal,
            "streamed-role layer {layer_index} must stream operation-local when the leftover budget cannot seat its page"
        );
        assert!(
            !should_commit_mandatory_complete_layer(39, true, Some(*layer_target),),
            "the commit gate must refuse layer {layer_index}: 33 pinned claims leave {} bytes, one page is {}",
            COMPOSED_LEFTOVER_BUDGET_BYTES - 33 * COMPLETE_LAYER_PAYLOAD_BYTES,
            COMPLETE_LAYER_PAYLOAD_BYTES
        );
    }

    // A leftover that holds the pinned claims plus streamed pages keeps
    // admit-on-read for exactly as many streamed layers as it can seat: the
    // retention win for already-materialized pages survives the fix where
    // the budget makes it real.
    let richer_budget_bytes = 35 * COMPLETE_LAYER_PAYLOAD_BYTES;
    let richer_opening = plan_expert_residency(
        MemoryPhase::Prefill,
        richer_budget_bytes,
        &incident_geometry(),
        &[],
    )
    .expect("the richer budget should pin 35 complete layers");
    let (_richer_residency, richer_plan) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        None,
        richer_opening,
        &[],
        0,
        &incident_geometry(),
    );
    assert_eq!(
        richer_plan.complete_layer_targets.len(),
        34,
        "the routed floors for all 38 layers consume one layer of the richer budget"
    );
    assert_eq!(
        richer_plan.layer_targets[34],
        ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead,
        "the single leftover page after 34 pinned claims admits the first streamed layer"
    );
    for layer_target in richer_plan
        .layer_targets
        .iter()
        .take(SPARSE_LAYER_COUNT)
        .skip(35)
    {
        assert_eq!(
            *layer_target,
            ExpertLayerResidencyTarget::StreamOperationLocal,
            "every further streamed layer exceeds the leftover and must stream operation-local"
        );
    }
}

#[test]
fn should_reproduce_the_first_incident_recovery_target_and_authorize_one_retry() {
    // The recovery arithmetic of the incident's first failure, reproduced
    // exactly: the workspace formula reconstructs 947,881,340 bytes (the
    // 92,243,324 transient bytes alive at the failure plus the rejected
    // allocation) and the reclamation formula demands 64,597,002 bytes. The
    // target is 13x smaller than the smallest eviction granularity the cache
    // owns (one whole 855,638,016-byte table), so one table satisfies it, and
    // the unchanged chunk is retried once instead of halving and re-paying a
    // complete-layer SSD read.
    let fixed_workspace_bytes = ForwardRecoveryPolicy::fixed_workspace_bytes(
        ACTIVE_MEMORY_BEFORE_RECLAMATION_BYTES,
        ACTIVE_MEMORY_AT_FAILURE_BYTES,
        ATTEMPTED_ALLOCATION_BYTES,
        0,
    );
    assert_eq!(fixed_workspace_bytes, FIRST_RECOVERY_FIXED_WORKSPACE_BYTES);

    let required_reclamation_bytes = ForwardRecoveryPolicy::required_reclamation_bytes(
        ACTIVE_MEMORY_BEFORE_RECLAMATION_BYTES,
        RETAINED_PAYLOAD_AT_FIRST_RECOVERY_BYTES,
        ACTIVE_MEMORY_CEILING_BYTES,
        fixed_workspace_bytes,
    );
    assert_eq!(
        required_reclamation_bytes,
        FIRST_RECOVERY_RECLAMATION_TARGET_BYTES
    );
    assert!(
        u64::try_from(required_reclamation_bytes).unwrap_or(u64::MAX)
            < COMPLETE_LAYER_PAYLOAD_BYTES,
        "the recovery target is below the whole-table eviction granularity"
    );

    assert!(
        ForwardRecoveryPolicy::retry_is_authorized(
            false,
            RETAINED_PAYLOAD_AT_FIRST_RECOVERY_BYTES as u64,
            RETAINED_PAYLOAD_AFTER_ONE_TABLE_EVICTION_BYTES as u64,
            required_reclamation_bytes,
            AdaptiveRamGrowthExecutionProfile::Paged,
        ),
        "one whole-table eviction satisfies the target, so the unchanged chunk is retried once"
    );
    assert!(
        !ForwardRecoveryPolicy::retry_is_authorized(
            true,
            RETAINED_PAYLOAD_AT_FIRST_RECOVERY_BYTES as u64,
            RETAINED_PAYLOAD_AFTER_ONE_TABLE_EVICTION_BYTES as u64,
            required_reclamation_bytes,
            AdaptiveRamGrowthExecutionProfile::Paged,
        ),
        "a second failure on the retried chunk must not loop"
    );
}

#[test]
fn should_floor_the_cache_ceiling_at_the_overshoot_after_every_budget_refresh() {
    // The budget-refresh floor legalizes the incident's overshoot for the rest
    // of the request: with 38 complete layers resident (32,514,244,608 bytes)
    // against a 28,974,982,400-byte composed budget, every later publication
    // floors the cache ceiling at the resident payload, and after each
    // one-table eviction the floor still sits a full table above the composed
    // budget. The composed budget never reasserts itself while the overshoot
    // exists, which is how a single first-chunk overshoot becomes a
    // whole-request budget the composer never chose.
    assert_eq!(
        retained_resident_ceiling_after_budget_refresh(
            COMPOSED_LEFTOVER_BUDGET_BYTES,
            RETAINED_PAYLOAD_AT_FIRST_RECOVERY_BYTES as u64,
        ),
        RETAINED_PAYLOAD_AT_FIRST_RECOVERY_BYTES as u64
    );
    assert_eq!(
        retained_resident_ceiling_after_budget_refresh(
            COMPOSED_LEFTOVER_BUDGET_BYTES,
            RETAINED_PAYLOAD_AFTER_ONE_TABLE_EVICTION_BYTES as u64,
        ),
        RETAINED_PAYLOAD_AFTER_ONE_TABLE_EVICTION_BYTES as u64
    );
    assert!(
        RETAINED_PAYLOAD_AFTER_ONE_TABLE_EVICTION_BYTES as u64 > COMPOSED_LEFTOVER_BUDGET_BYTES,
        "after one whole-table eviction the floored ceiling still exceeds the composed budget"
    );
}

#[test]
fn should_round_the_recovery_target_up_to_whole_table_granularity() {
    // The eviction granularity is one complete table; a byte-exact target
    // leaves the next attempt's headroom dependent on where table boundaries
    // fall. The incident's 64,597,002-byte deficit must round up to one
    // whole 855,638,016-byte table, and a deficit of one and a fraction
    // tables must round to two.
    assert_eq!(
        whole_table_reclamation_target_bytes(
            FIRST_RECOVERY_RECLAMATION_TARGET_BYTES,
            COMPLETE_LAYER_PAYLOAD_BYTES as usize,
        ),
        COMPLETE_LAYER_PAYLOAD_BYTES as usize
    );
    assert_eq!(
        whole_table_reclamation_target_bytes(
            COMPLETE_LAYER_PAYLOAD_BYTES as usize + 1,
            COMPLETE_LAYER_PAYLOAD_BYTES as usize,
        ),
        2 * COMPLETE_LAYER_PAYLOAD_BYTES as usize
    );
    assert_eq!(
        whole_table_reclamation_target_bytes(
            2 * COMPLETE_LAYER_PAYLOAD_BYTES as usize,
            COMPLETE_LAYER_PAYLOAD_BYTES as usize,
        ),
        2 * COMPLETE_LAYER_PAYLOAD_BYTES as usize,
        "an exact whole-table multiple must stay unchanged"
    );
    assert_eq!(
        whole_table_reclamation_target_bytes(0, COMPLETE_LAYER_PAYLOAD_BYTES as usize),
        0,
        "zero deficit must stay zero"
    );
    assert_eq!(
        whole_table_reclamation_target_bytes(FIRST_RECOVERY_RECLAMATION_TARGET_BYTES, 0),
        FIRST_RECOVERY_RECLAMATION_TARGET_BYTES,
        "a dense model with no tables must keep the byte-exact target"
    );
}

#[test]
fn should_leave_the_evicted_layer_unseatable_under_both_documented_ceilings() {
    // The eviction-and-restream thrash the halving loop entered: once one table
    // is evicted under request pressure, neither documented ceiling restores the
    // evicted layer — the pressure override admits only
    // `resident_at_pressure_time - target` bytes, and the next budget-refresh
    // floor freezes the post-eviction payload. The evicted layer is therefore
    // re-read from SSD on every subsequent attempt with no path back into the
    // cache, which is the steady state the incident's five failed attempts and
    // the 69-second silent span executed.
    let pressure_override_bytes = RETAINED_PAYLOAD_AT_FIRST_RECOVERY_BYTES as u64
        - FIRST_RECOVERY_RECLAMATION_TARGET_BYTES as u64;
    let post_eviction_resident_bytes = RETAINED_PAYLOAD_AFTER_ONE_TABLE_EVICTION_BYTES as u64;
    assert!(
        post_eviction_resident_bytes + COMPLETE_LAYER_PAYLOAD_BYTES > pressure_override_bytes,
        "the pressure override refuses the evicted layer's re-adoption"
    );
    let floored_ceiling_bytes = retained_resident_ceiling_after_budget_refresh(
        COMPOSED_LEFTOVER_BUDGET_BYTES,
        post_eviction_resident_bytes,
    );
    assert_eq!(floored_ceiling_bytes, post_eviction_resident_bytes);
    assert!(
        post_eviction_resident_bytes + COMPLETE_LAYER_PAYLOAD_BYTES > floored_ceiling_bytes,
        "the budget-refresh floor also refuses the evicted layer's re-adoption"
    );
}
