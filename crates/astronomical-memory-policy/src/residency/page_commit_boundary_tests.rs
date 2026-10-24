//! Boundary, positive, and negative tests for the multi-token union
//! complete-layer adoption gate.

use crate::residency::ExpertLayerResidencyTarget;
use crate::residency::should_commit_multi_token_union_as_complete_layer;

#[test]
fn should_commit_a_multi_token_union_as_a_complete_layer_when_the_plan_funds_it() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // The 2026-10-23 Ornith under-utilization flight: a 2,048-token chunk
    // routes every expert of the layer, paying the complete-layer read cost,
    // and the plan funds the claim, so the union must be offered to retained
    // ownership instead of re-read from storage on every later chunk.
    assert!(should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
        128,
        128,
    ));
    assert!(should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead),
        128,
        128,
    ));
}

#[test]
fn should_never_commit_a_multi_token_union_that_is_not_the_whole_layer() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // A partial union is not a complete layer: adopting it as one would
    // claim the whole layer's payload budget for a fraction of the experts.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
        96,
        128,
    ));
    // One-off and empty unions never adopt.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
        0,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
        128,
        0,
    ));
}

#[test]
fn should_not_commit_a_multi_token_union_for_unfunded_or_non_production_plans() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // Unfunded or streaming targets never authorize the adoption offer.
    for unfunded_target in [
        ExpertLayerResidencyTarget::StreamOperationLocal,
        ExpertLayerResidencyTarget::ReleasePartial,
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit,
        ExpertLayerResidencyTarget::PreserveComplete,
        ExpertLayerResidencyTarget::PreservePartial,
        ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
    ] {
        assert!(
            !should_commit_multi_token_union_as_complete_layer(
                2_048,
                true,
                Some(unfunded_target),
                128,
                128,
            ),
            "target {unfunded_target:?} must not authorize the offer"
        );
    }
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048, true, None, 128, 128,
    ));
    // Diagnostic paging modes and single-token decode forwards never adopt.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        false,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
        128,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        1,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
        128,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        0,
        true,
        Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead),
        128,
        128,
    ));
}
