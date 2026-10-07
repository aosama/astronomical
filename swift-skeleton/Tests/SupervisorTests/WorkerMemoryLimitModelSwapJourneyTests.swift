import Foundation

import Testing

import JourneyCategories

@testable import Supervisor

/**
 * Proves that live memory policy remains coherent through the next lazy
 * model swap, migrating
 * apps/supervisor/tests/hermetic/worker_memory_limit_model_swap.rs: after a
 * live ceiling update changes the configuration generation, the next
 * generation still loads its model against the synchronized policy.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerMemoryLimitModelSwapJourneyTests {

    @Test
    func should_load_a_model_after_a_live_memory_update_changes_the_configuration_generation() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchIdleWorkerFixture()
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor
        let updatedConfigurationGeneration: String = "generation-after-live-memory-update"

        supervisor.stageMemoryConfigurationGeneration(updatedConfigurationGeneration)
        let memoryUpdateOutcome: MlxMemoryLimitUpdateOutcome = try supervisor.updateMlxMemoryLimit(
            32_000_000_000,
            configurationGeneration: updatedConfigurationGeneration)
        supervisor.recordMemoryConfigurationGeneration(
            updatedConfigurationGeneration,
            memoryUpdateOutcome)
        #expect(memoryUpdateOutcome == .applied)

        let generationEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID,
                requestId: 20))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            generationEvents,
            generationLabel: "after live memory update")
        #expect(supervisor.workerHealthSnapshot().status == .ready)
        _ = try supervisor.shutdown()
    }
}
