//! Whether a planned residency release may run in the current request phase.

use super::{ExpertLayerResidencyTarget, ExpertResidencyPhase};

/// Returns whether execution may drop the named layer now.
///
/// Prefill still needs complete layers that already fit. Releasing them would
/// force another SSD complete-layer read on the next chunk. Decode handoff
/// (`GenerationPreparation`) is the phase that shrinks to leftover generation
/// topology. Partial pages stay elastic in every phase.
#[must_use]
pub const fn should_enact_planned_expert_release(
    phase: ExpertResidencyPhase,
    target: ExpertLayerResidencyTarget,
) -> bool {
    match target {
        ExpertLayerResidencyTarget::ReleasePartial => {
            // Prefill may drop a cold routed page to finish the prompt. Generation
            // must keep the experts that prompt already streamed.
            matches!(
                phase,
                ExpertResidencyPhase::Prefill | ExpertResidencyPhase::Idle
            )
        }
        ExpertLayerResidencyTarget::ReleaseCompleteForExactDeficit => {
            matches!(phase, ExpertResidencyPhase::Idle)
        }
        ExpertLayerResidencyTarget::PreserveComplete
        | ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
        | ExpertLayerResidencyTarget::PreservePartial
        | ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
        | ExpertLayerResidencyTarget::StreamOperationLocal => false,
    }
}
