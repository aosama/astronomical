//! Truth-table tests for the planned-expert-release phase gate: every
//! residency target across every lifecycle phase.

use crate::phase::MemoryPhase;
use crate::residency::{ExpertLayerResidencyTarget, should_enact_planned_expert_release};

const EVERY_PHASE: [MemoryPhase; 4] = [
    MemoryPhase::Prefill,
    MemoryPhase::GenerationPreparation,
    MemoryPhase::Decode,
    MemoryPhase::Idle,
];

const EVERY_TARGET: [ExpertLayerResidencyTarget; 8] = [
    ExpertLayerResidencyTarget::PreserveComplete,
    ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead,
    ExpertLayerResidencyTarget::PreservePartial,
    ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
    ExpertLayerResidencyTarget::StreamOperationLocal,
    ExpertLayerResidencyTarget::ReleasePartial,
    ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit,
    ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead,
];

#[test]
fn should_enact_a_partial_release_only_during_prefill_and_idle() {
    for phase in EVERY_PHASE {
        let expected_release_may_run = matches!(phase, MemoryPhase::Prefill | MemoryPhase::Idle);
        assert_eq!(
            should_enact_planned_expert_release(phase, ExpertLayerResidencyTarget::ReleasePartial),
            expected_release_may_run,
            "ReleasePartial in {phase:?}"
        );
    }
}

#[test]
fn should_enact_an_exact_deficit_complete_release_only_when_idle() {
    for phase in EVERY_PHASE {
        assert_eq!(
            should_enact_planned_expert_release(
                phase,
                ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit,
            ),
            phase == MemoryPhase::Idle,
            "ReleaseCompleteForExactDeficit in {phase:?}"
        );
    }
}

#[test]
fn should_never_enact_a_preserve_admit_promote_or_stream_target_in_any_phase() {
    for phase in EVERY_PHASE {
        for target in EVERY_TARGET {
            let target_names_a_release = matches!(
                target,
                ExpertLayerResidencyTarget::ReleasePartial
                    | ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit
            );
            if !target_names_a_release {
                assert!(
                    !should_enact_planned_expert_release(phase, target),
                    "non-release target {target:?} must never enact in {phase:?}"
                );
            }
        }
    }
}
