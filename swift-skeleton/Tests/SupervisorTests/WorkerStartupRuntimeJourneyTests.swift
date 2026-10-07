import Foundation

import Testing

import JourneyCategories

@testable import Supervisor

/**
 * Proves the first generation still completes when the InitializeWorker
 * acknowledgement is still in flight after Idle has made the worker ready,
 * migrating apps/supervisor/tests/hermetic/worker_startup_runtime.rs: the
 * fixture delays its startup runtime acknowledgement by 150 ms, so the
 * request waits behind the delayed policy without rejecting admission.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerStartupRuntimeJourneyTests {

    @Test
    func should_complete_generation_when_startup_runtime_acknowledgement_is_still_in_flight() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchIdleWorkerFixture(
                configurationGeneration: "delayed-startup-runtime-configuration")
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        let generationEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID,
                requestId: 1))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            generationEvents,
            generationLabel: "delayed startup runtime acknowledgement")

        _ = try supervisor.shutdown()
    }
}
