//! Boundary, positive, and negative tests for the multi-token union
//! complete-layer adoption gate.

use crate::residency::ExpertLayerState;
use crate::residency::{
    should_commit_multi_token_union_as_complete_layer, should_seed_partial_layer_on_prefill_read,
};

#[test]
fn should_commit_a_multi_token_union_as_a_complete_layer_when_the_plan_funds_it() {
    // The 2026-10-23 Ornith under-utilization flight: a 2,048-token chunk
    // routes every expert of the layer, paying the complete-layer read cost,
    // and the plan funds the claim, so the union must be offered to retained
    // ownership instead of re-read from storage on every later chunk.
    assert!(should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        128,
        128,
    ));
    assert!(should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        128,
        128,
    ));
}

#[test]
fn should_never_commit_a_multi_token_union_that_is_not_the_whole_layer() {
    // A partial union is not a complete layer: adopting it as one would
    // claim the whole layer's payload budget for a fraction of the experts.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        96,
        128,
    ));
    // One-off and empty unions never adopt.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        0,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        128,
        0,
    ));
}

#[test]
fn should_not_commit_a_multi_token_union_for_unfunded_or_non_production_plans() {
    // Only a layer that should be kept whole authorizes the adoption offer.
    for unfunded_target in [ExpertLayerState::Absent, ExpertLayerState::Hot] {
        assert!(
            !should_commit_multi_token_union_as_complete_layer(
                2_048,
                true,
                unfunded_target,
                128,
                128,
            ),
            "target {unfunded_target:?} must not authorize the offer"
        );
    }
    // Diagnostic paging modes and single-token decode forwards never adopt.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        false,
        ExpertLayerState::Kept,
        128,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        1,
        true,
        ExpertLayerState::Kept,
        128,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        0,
        true,
        ExpertLayerState::Kept,
        128,
        128,
    ));
}

#[test]
fn should_seed_a_partial_prefill_union_for_a_funded_complete_layer_target() {
    let funded_target = ExpertLayerState::Kept;
    for routed_expert_count in [1, 127] {
        assert!(
            should_seed_partial_layer_on_prefill_read(
                2_048,
                true,
                funded_target,
                0,
                routed_expert_count,
                128,
            ),
            "a cold non-empty partial union must seed for {funded_target:?} \
             at {routed_expert_count} routed experts"
        );
    }
}

#[test]
fn should_not_seed_a_partial_prefill_union_outside_its_funded_boundary() {
    let funded_target = ExpertLayerState::Kept;
    for routed_expert_count in [0, 128, 129] {
        assert!(!should_seed_partial_layer_on_prefill_read(
            2_048,
            true,
            funded_target,
            0,
            routed_expert_count,
            128,
        ));
    }
    assert!(!should_seed_partial_layer_on_prefill_read(
        2_048,
        true,
        funded_target,
        0,
        0,
        0,
    ));
    // An existing partial table uses the missing-expert growth path instead.
    assert!(!should_seed_partial_layer_on_prefill_read(
        2_048,
        true,
        funded_target,
        1,
        64,
        128,
    ));
    // Decode tokens, diagnostic paths, and absent or partial-only plans do not
    // authorize a prefill completion seed.
    assert!(!should_seed_partial_layer_on_prefill_read(
        1,
        true,
        funded_target,
        0,
        64,
        128,
    ));
    assert!(!should_seed_partial_layer_on_prefill_read(
        0,
        true,
        funded_target,
        0,
        64,
        128,
    ));
    assert!(!should_seed_partial_layer_on_prefill_read(
        2_048,
        false,
        funded_target,
        0,
        64,
        128,
    ));
    for unfunded_target in [ExpertLayerState::Absent, ExpertLayerState::Hot] {
        assert!(
            !should_seed_partial_layer_on_prefill_read(2_048, true, unfunded_target, 0, 64, 128,),
            "target {unfunded_target:?} must not authorize the seed"
        );
    }
}

#[test]
fn should_complete_a_partial_layer_when_the_plan_funds_the_remaining_experts() {
    use crate::residency::ExpertLayerState;
    use crate::residency::should_complete_partial_layer_on_prefill_read;
    assert!(
        should_complete_partial_layer_on_prefill_read(true, ExpertLayerState::Kept, 65, 128),
        "a partly held layer that should be kept whole converges on the next read"
    );
}

#[test]
fn should_not_complete_an_empty_complete_or_unfunded_layer() {
    use crate::residency::ExpertLayerState;
    use crate::residency::should_complete_partial_layer_on_prefill_read;
    let funded = ExpertLayerState::Kept;
    // Cold-union seeding and complete-union adoption own the empty-table case.
    assert!(!should_complete_partial_layer_on_prefill_read(
        true, funded, 0, 128
    ));
    // A complete table has nothing left to read.
    assert!(!should_complete_partial_layer_on_prefill_read(
        true, funded, 128, 128
    ));
    assert!(!should_complete_partial_layer_on_prefill_read(
        true, funded, 200, 128
    ));
    // Unfunded or absent plans, diagnostic modes, and degenerate capacities
    // never authorize the extended read.
    assert!(!should_complete_partial_layer_on_prefill_read(
        true,
        ExpertLayerState::Absent,
        65,
        128
    ));
    assert!(!should_complete_partial_layer_on_prefill_read(
        false, funded, 65, 128
    ));
    assert!(!should_complete_partial_layer_on_prefill_read(
        true, funded, 65, 0
    ));
}
