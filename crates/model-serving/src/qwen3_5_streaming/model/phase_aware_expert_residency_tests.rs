//! Beside-source coverage for the pure residency-telemetry counters. The
//! ceiling refresh and release enactment run under the serial engine lane
//! against real pagers.

use super::{count_planned_partial_layers, count_planned_streamed_layers};
use crate::ExpertLayerResidencyTarget;

#[test]
fn should_count_partial_and_streamed_targets_by_their_dispositions() {
    let layer_targets = [
        ExpertLayerResidencyTarget::PreserveComplete,
        ExpertLayerResidencyTarget::PreservePartial,
        ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
        ExpertLayerResidencyTarget::StreamOperationLocal,
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit,
    ];

    assert_eq!(count_planned_partial_layers(&layer_targets), 2);
    assert_eq!(count_planned_streamed_layers(&layer_targets), 2);
}

#[test]
fn should_count_zero_targets_in_an_empty_plan() {
    assert_eq!(count_planned_partial_layers(&[]), 0);
    assert_eq!(count_planned_streamed_layers(&[]), 0);
}

#[test]
fn should_count_every_promotion_and_release_as_streamed() {
    let layer_targets = [
        ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead,
        ExpertLayerResidencyTarget::ReleasePartial,
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit,
    ];

    assert_eq!(count_planned_streamed_layers(&layer_targets), 3);
    assert_eq!(count_planned_partial_layers(&layer_targets), 0);
}
