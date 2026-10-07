import Foundation

import Testing

import IpcProtocol
import JourneyCategories

@testable import Supervisor

/**
 * Acceptance journeys for queued generation across a lazy model swap,
 * migrating apps/supervisor/tests/hermetic/worker_model_swap.rs against the
 * real supervisor test worker: idle telemetry observed before the swap
 * acknowledgement must not lose the queued request, generation-scoped
 * events during the swap wait contain the worker, chat/image/chat swaps
 * carry the exact runtime policy, swap failure recovery, the staged health
 * snapshot across delayed policy acknowledgements, and the disconnect
 * tripwire that proves an image is never dispatched after its requester
 * went away.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerModelSwapJourneyTests {

    @Test
    func should_complete_a_queued_model_swap_when_idle_telemetry_arrives_before_acknowledgement() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchIdleWorkerFixture()
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        let firstGenerationOutcome: GenerationJourneyOutcome = WorkerModelSwapJourneyTests
            .startChatGenerationThread(
                supervisor,
                modelId: IdleWorkerJourneySupport.DELAYED_COMPLETION_MODEL_ID,
                requestId: 1)
        WorkerModelSwapJourneyTests.awaitOutstandingAdmissionTickets(supervisor, expectedCount: 1)
        let queuedGenerationOutcome: GenerationJourneyOutcome = WorkerModelSwapJourneyTests
            .startChatGenerationThread(
                supervisor,
                modelId: IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID,
                requestId: 2)

        let firstStreamEvents: Array<ChatGenerationStreamEvent>? =
            WorkerModelSwapJourneyTests.completedStreamEvents(
                from: WorkerModelSwapJourneyTests.awaitGenerationOutcome(
                    firstGenerationOutcome, journeyLabel: "first delayed completion"),
                journeyLabel: "first delayed completion")
        IdleWorkerJourneySupport.assertGenerationCompleted(
            firstStreamEvents ?? [],
            generationLabel: "first delayed completion")
        let queuedStreamEvents: Array<ChatGenerationStreamEvent>? =
            WorkerModelSwapJourneyTests.completedStreamEvents(
                from: WorkerModelSwapJourneyTests.awaitGenerationOutcome(
                    queuedGenerationOutcome, journeyLabel: "queued telemetry-before-swap"),
                journeyLabel: "queued telemetry-before-swap")
        IdleWorkerJourneySupport.assertGenerationCompleted(
            queuedStreamEvents ?? [],
            generationLabel: "queued telemetry-before-swap")

        let workerHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot()
        #expect(workerHealthSnapshot.status == .ready)
        #expect(workerHealthSnapshot.readyModelId == IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID)
        #expect(
            workerHealthSnapshot.readyModelCapabilities?.chat?.maxOutputTokens == 64,
            "loaded model capabilities should be acknowledged from the swap policy")
        #expect(
            workerHealthSnapshot.workerRuntimeFeatureConfiguration?.loadedModel?.modelId()
                == IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID)
        _ = try supervisor.shutdown()
    }

    @Test
    func should_reject_a_generation_scoped_event_while_waiting_for_model_swap() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchIdleWorkerFixture()
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        let generationStartOutcome: Result<Array<ChatGenerationStreamEvent>, Error>? =
            WorkerModelSwapJourneyTests.awaitGenerationOutcome(
                WorkerModelSwapJourneyTests.startChatGenerationThread(
                    supervisor,
                    modelId: IdleWorkerJourneySupport.GENERATION_EVENT_BEFORE_SWAP_MODEL_ID,
                    requestId: 3),
                journeyLabel: "generation-scoped event during swap wait")
        guard case let .failure(startError)? = generationStartOutcome else {
            Issue.record("the generation-scoped swap event should fail the generation start")
            return
        }
        #expect(
            (startError as? GenerationStartError) == GenerationStartError.workerUnavailable,
            "the rejected swap must surface as workerUnavailable, received \(startError)")
        #expect(supervisor.workerHealthSnapshot().status == .unavailable)
        _ = try supervisor.shutdown()
    }

    @Test
    func should_swap_chat_to_image_to_chat_with_exact_runtime_policy() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchIdleWorkerFixture()
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        let firstChatEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID,
                requestId: 10))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            firstChatEvents,
            generationLabel: "chat before image swap")

        let imageOutput: ImageGenerationOutput = try supervisor.startImageGeneration(
            IdleWorkerJourneySupport.imageGenerationCommand(
                modelId: IdleWorkerJourneySupport.IMAGE_MODEL_ID,
                requestId: 11))
        #expect(imageOutput.generatedImage.mimeType == "image/png")
        #expect(imageOutput.resultMetadata.widthPixels == 1_024)
        #expect(imageOutput.resultMetadata.seed == 7)
        #expect(
            supervisor.workerHealthSnapshot().workerRuntimeFeatureConfiguration?.loadedModel?.modelId()
                == IdleWorkerJourneySupport.IMAGE_MODEL_ID,
            "the image model's runtime policy should be the acknowledged policy")

        let finalChatEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID,
                requestId: 12))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            finalChatEvents,
            generationLabel: "chat after image swap")
        _ = try supervisor.shutdown()
    }

    @Test
    func should_recover_from_an_image_model_swap_failure_without_poisoning_the_worker() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchIdleWorkerFixture()
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        do {
            _ = try supervisor.startImageGeneration(
                IdleWorkerJourneySupport.imageGenerationCommand(
                    modelId: IdleWorkerJourneySupport.INVALID_IMAGE_MODEL_ID,
                    requestId: 13))
            Issue.record("the invalid image model should fail to load")
        } catch let generationStartError as GenerationStartError {
            guard case .modelLoadFailed = generationStartError else {
                Issue.record(
                    Comment(stringLiteral: "expected modelLoadFailed, received \(generationStartError)"))
                return
            }
        }

        let validImageOutput: ImageGenerationOutput = try supervisor.startImageGeneration(
            IdleWorkerJourneySupport.imageGenerationCommand(
                modelId: IdleWorkerJourneySupport.IMAGE_MODEL_ID,
                requestId: 14))
        #expect(validImageOutput.generatedImage.mimeType == "image/png")
        _ = try supervisor.shutdown()
    }

    @Test
    func should_publish_model_identity_and_runtime_policy_as_one_health_snapshot() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchIdleWorkerFixture()
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        let firstGenerationEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID,
                requestId: 15))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            firstGenerationEvents,
            generationLabel: "health snapshot before delayed policy")

        let swapOutcome: GenerationJourneyOutcome = WorkerModelSwapJourneyTests.startChatGenerationThread(
            supervisor,
            modelId: IdleWorkerJourneySupport.DELAYED_POLICY_ACK_MODEL_ID,
            requestId: 16)
        WorkerModelSwapJourneyTests.awaitOutstandingAdmissionTickets(supervisor, expectedCount: 1)
        // The delayed policy acknowledgement (200 ms) holds the staged swap
        // unpublished, so the snapshot still names the previous model.
        Thread.sleep(forTimeInterval: 0.075)
        let stagedHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot()
        #expect(stagedHealthSnapshot.readyModelId == IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID)
        #expect(
            stagedHealthSnapshot.workerRuntimeFeatureConfiguration?.loadedModel?.modelId()
                == IdleWorkerJourneySupport.TELEMETRY_BEFORE_SWAP_MODEL_ID)

        let swappedStreamEvents: Array<ChatGenerationStreamEvent>? =
            WorkerModelSwapJourneyTests.completedStreamEvents(
                from: WorkerModelSwapJourneyTests.awaitGenerationOutcome(
                    swapOutcome, journeyLabel: "health snapshot after delayed policy"),
                journeyLabel: "health snapshot after delayed policy")
        IdleWorkerJourneySupport.assertGenerationCompleted(
            swappedStreamEvents ?? [],
            generationLabel: "health snapshot after delayed policy")
        let committedHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot()
        #expect(
            committedHealthSnapshot.readyModelId == IdleWorkerJourneySupport.DELAYED_POLICY_ACK_MODEL_ID)
        #expect(
            committedHealthSnapshot.workerRuntimeFeatureConfiguration?.loadedModel?.modelId()
                == IdleWorkerJourneySupport.DELAYED_POLICY_ACK_MODEL_ID)
        _ = try supervisor.shutdown()
    }

    /**
     * The disconnect tripwire: the fixture worker exits the moment it
     * receives the "must-not-dispatch-after-disconnect" image command, so a
     * supervisor that dispatches after its requester went away both trips
     * the marker and loses the worker. The synchronous port's requester-
     * went-away surface is shutdown requested during the delayed policy
     * acknowledgement; the Rust journey's follow-up reuse is covered by the
     * queued-swap and staged-snapshot journeys above.
     */
    @Test
    func should_not_dispatch_an_image_after_disconnect_during_model_swap() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchIdleWorkerFixture()
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        let disconnectedImageOutcome: ImageGenerationJourneyOutcome = ImageGenerationJourneyOutcome(
            workerThread: Thread())
        let imageThread: Thread = Thread {
            do {
                disconnectedImageOutcome.record(output: try supervisor.startImageGeneration(
                    ImageGenerationCommand(
                        requestId: RequestId(rawRequestId: 17),
                        model: IdleWorkerJourneySupport.DELAYED_IMAGE_POLICY_ACK_MODEL_ID,
                        prompt: "must-not-dispatch-after-disconnect",
                        settings: IdleWorkerJourneySupport.imageGenerationSettings())))
            } catch {
                disconnectedImageOutcome.record(error: error)
            }
        }
        imageThread.name = "journey-disconnected-image"
        disconnectedImageOutcome.workerThread = imageThread
        imageThread.start()
        WorkerModelSwapJourneyTests.awaitOutstandingAdmissionTickets(supervisor, expectedCount: 1)
        Thread.sleep(forTimeInterval: 0.075)

        let terminationOutcome: WorkerTerminationOutcome = try supervisor.shutdown()
        let observedOutcome: Result<ImageGenerationOutput, Error>? =
            disconnectedImageOutcome.awaitOutcome(deadlineSeconds: 3, journeyLabel: "disconnected image")
        guard case let .failure(disconnectError)? = observedOutcome else {
            Issue.record("the disconnected image request should fail, not complete")
            return
        }
        #expect(
            (disconnectError as? GenerationStartError) == GenerationStartError.workerUnavailable,
            "the abandoned request must refuse before dispatch, received \(disconnectError)")
        #expect(
            FileManager.default.fileExists(atPath: harness.disconnectTripwireMarkerPath) == false,
            "the tripwire prompt must never reach the worker")
        #expect(terminationOutcome == .graceful(processExitSuccessful: true))
    }

    // MARK: Threaded journey helpers

    private static func startChatGenerationThread(
        _ supervisor: WorkerSupervisor,
        modelId: String,
        requestId: UInt64
    ) -> GenerationJourneyOutcome {
        let outcome: GenerationJourneyOutcome = GenerationJourneyOutcome(workerThread: Thread())
        let generationThread: Thread = Thread {
            do {
                outcome.record(streamEvents: try supervisor.startChatGeneration(
                    IdleWorkerJourneySupport.chatCommand(modelId: modelId, requestId: requestId)))
            } catch {
                outcome.record(error: error)
            }
        }
        generationThread.name = "journey-chat-\(requestId)"
        outcome.workerThread = generationThread
        generationThread.start()
        return outcome
    }

    private static func awaitGenerationOutcome(
        _ outcome: GenerationJourneyOutcome,
        journeyLabel: String
    ) -> Result<Array<ChatGenerationStreamEvent>, Error>? {
        let joinDeadline: Date = Date().addingTimeInterval(5)
        while (Date() < joinDeadline) {
            if let observedOutcome: Result<Array<ChatGenerationStreamEvent>, Error> = outcome.observedOutcome {
                return observedOutcome
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        Issue.record(Comment(stringLiteral: "\(journeyLabel): the generation never finished"))
        return nil
    }

    /// Flattens a joined generation outcome into its events, recording the
    /// journey failure when the generation ended in an error instead.
    private static func completedStreamEvents(
        from generationResult: Result<Array<ChatGenerationStreamEvent>, Error>?,
        journeyLabel: String
    ) -> Array<ChatGenerationStreamEvent>? {
        guard let generationResult: Result<Array<ChatGenerationStreamEvent>, Error> = generationResult else {
            return nil
        }
        switch (generationResult) {
        case let .success(streamEvents):
            return streamEvents
        case let .failure(generationError):
            Issue.record(Comment(stringLiteral: "\(journeyLabel): the generation failed: \(generationError)"))
            return nil
        }
    }

    private static func awaitOutstandingAdmissionTickets(
        _ supervisor: WorkerSupervisor,
        expectedCount: Int
    ) -> Void {
        let fillDeadline: Date = Date().addingTimeInterval(5)
        while (supervisor.outstandingAdmissionTicketCount < expectedCount) {
            if (Date() >= fillDeadline) {
                Issue.record(
                    Comment(stringLiteral: "the queue never reached \(expectedCount) outstanding requests"))
                return
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
    }
}
