import Foundation

/// Whether a planned residency release may run in the current request phase.
///
/// Execution asks this policy before dropping the named layer; it never
/// invents a second answer. Prefill still needs complete layers that already
/// fit — releasing them would force another SSD complete-layer read on the
/// next chunk. Generation handoff (`generationPreparation`) is the phase
/// that shrinks to leftover generation topology. Partial pages stay elastic
/// in every phase.
public enum ExpertReleasePolicy {

    /**
     * Returns whether execution may drop the named layer now.
     *
     * - Parameters:
     *   - phase: Lifecycle position the release would run in.
     *   - target: The active plan's target for the layer.
     * - Returns: Whether the planned release may be enacted in this phase.
     */
    public static func shouldEnactPlannedExpertRelease(
        phase: MemoryPhase,
        target: ExpertLayerResidencyTarget
    ) -> Bool {
        switch target {
        case .releasePartial:
            // Prefill may drop a cold routed page to finish the prompt.
            // Generation must keep the experts that prompt already streamed.
            return phase == .prefill || phase == .idle
        case .releaseCompleteForExactDeficit:
            return phase == .idle
        case .preserveComplete, .promoteCompleteOnMandatoryRead, .preservePartial,
            .admitPartialOnMandatoryRouteRead, .streamOperationLocal:
            return false
        }
    }
}
