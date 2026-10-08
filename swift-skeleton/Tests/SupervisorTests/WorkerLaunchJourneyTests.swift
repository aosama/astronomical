import Foundation

import Testing

import JourneyCategories

@testable import IpcProtocol
@testable import Supervisor

/**
 * Worker launch journeys, migrating
 * apps/supervisor/tests/hermetic/worker_launch.rs: the stream-closure
 * diagnostics carry the exit status and a bounded stderr tail, the first
 * generation request is what loads a model, idle memory limits apply
 * immediately while unacknowledged ones contain the worker, a rejected
 * model load leaves the worker available, oversized commands are refused
 * without disturbing the loaded model, unmapped models never reach the
 * worker, and a hanging model load stays bounded.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerLaunchJourneyTests {

    @Test
    func should_report_worker_exit_status_and_stderr_when_the_event_stream_closes() throws {
        let probeWorkerExecutablePath: String = try IdleWorkerJourneySupport.locateBuiltExecutable(
            executableName: IdleWorkerJourneySupport.STDERR_PROBE_EXECUTABLE_NAME)
        let probeWorkerProcess: WorkerProcess = try WorkerProcess.launch(
            workerExecutablePath: probeWorkerExecutablePath)
        defer {
            _ = try? probeWorkerProcess.close()
        }
        let eventPump: WorkerEventPump = WorkerEventPump(workerProcess: probeWorkerProcess)

        let readinessEvent: WorkerEvent? = try eventPump.nextEvent(within: 5)
        #expect(readinessEvent != nil, "the probe's single readiness event should be readable")

        var workerExitDiagnostic: String = ""
        do {
            _ = try eventPump.nextEvent(within: 5)
            Issue.record("closing the event stream should carry process diagnostics")
        } catch let workerControlError as WorkerControlError {
            workerExitDiagnostic = workerControlError.errorDescription ?? ""
        }
        #expect(
            workerExitDiagnostic.contains("worker process exited after closing its event stream"))
        #expect(workerExitDiagnostic.contains("exit code 0"))
        #expect(workerExitDiagnostic.contains("stderr-probe worker observed visible stderr"))
        #expect(
            workerExitDiagnostic.count < 9_000,
            "worker stderr diagnostics must remain bounded")
    }

    @Test
    func should_load_the_requested_model_only_after_the_first_generation_request() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchUnconfiguredIdleWorker(
                modelPolicyCatalog: [
                    IdleWorkerJourneySupport.REQUESTED_MODEL_ID:
                        IdleWorkerJourneySupport.runtimeModelPolicy(
                            IdleWorkerJourneySupport.REQUESTED_MODEL_ID,
                            modelDirectory: "/models/requested-model",
                            maximumOutputTokens: 128),
                ],
                modelLoadTimeout: 10)
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor
        #expect(supervisor.workerHealthSnapshot().readyModelId == nil)

        let generationEvents: Array<ChatGenerationStreamEvent> = try supervisor.startChatGeneration(
            IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.REQUESTED_MODEL_ID,
                requestId: 1))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            generationEvents,
            generationLabel: "first lazy model load")

        let workerHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot()
        #expect(workerHealthSnapshot.readyModelId == IdleWorkerJourneySupport.REQUESTED_MODEL_ID)
        #expect(workerHealthSnapshot.minimumMlxMemoryCeilingBytes == 3_000_000_000)
        #expect(
            workerHealthSnapshot.expertMemoryMode == ExpertMemoryMode.resident,
            "health must publish the expert mode selected before the replacement model became ready")
        _ = try supervisor.shutdown()
    }

    @Test
    func should_apply_a_memory_limit_immediately_when_worker_is_idle() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchUnconfiguredIdleWorker(
                modelPolicyCatalog: [:],
                modelLoadTimeout: 10)
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        let memoryUpdateOutcome: MlxMemoryLimitUpdateOutcome = try supervisor.updateMlxMemoryLimit(
            32_000_000_000,
            configurationGeneration: "memory-generation")
        #expect(memoryUpdateOutcome == .applied)

        let workerHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot()
        #expect(workerHealthSnapshot.effectiveMlxMemoryCeilingBytes == 32_000_000_000)
        #expect(workerHealthSnapshot.pendingMlxMemoryCeilingBytes == nil)
        #expect(workerHealthSnapshot.mlxMemoryLimitError == nil)
        #expect(workerHealthSnapshot.readyModelId == nil)
        #expect(workerHealthSnapshot.expertMemoryMode == nil)
        _ = try supervisor.shutdown()
    }

    @Test
    func should_contain_a_worker_that_does_not_acknowledge_a_memory_limit_update() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchUnconfiguredIdleWorker(
                modelPolicyCatalog: [:],
                modelLoadTimeout: 1)
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        // The fixture ignores the 30 GB ceiling command, so the bounded
        // acknowledgement wait must fail and contain the worker.
        do {
            _ = try supervisor.updateMlxMemoryLimit(
                30_000_000_000,
                configurationGeneration: "timeout-memory-generation")
            Issue.record("an unacknowledged memory update should fail")
        } catch let workerControlError as WorkerControlError {
            let updateDiagnostic: String = workerControlError.errorDescription ?? ""
            #expect(
                updateDiagnostic.contains("acknowledge the memory-ceiling change"),
                "the bounded wait must name the memory acknowledgement, received \(updateDiagnostic)")
        }
        #expect(supervisor.workerHealthSnapshot().status == .unavailable)
        _ = try supervisor.shutdown()
    }

    @Test
    func should_remain_available_after_the_first_requested_model_fails_to_load() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchUnconfiguredIdleWorker(
                modelPolicyCatalog: [
                    IdleWorkerJourneySupport.INVALID_MODEL_ID:
                        IdleWorkerJourneySupport.runtimeModelPolicy(
                            IdleWorkerJourneySupport.INVALID_MODEL_ID,
                            modelDirectory: "/models/invalid-model",
                            maximumOutputTokens: 128),
                    IdleWorkerJourneySupport.REQUESTED_MODEL_ID:
                        IdleWorkerJourneySupport.runtimeModelPolicy(
                            IdleWorkerJourneySupport.REQUESTED_MODEL_ID,
                            modelDirectory: "/models/requested-model",
                            maximumOutputTokens: 128),
                ],
                modelLoadTimeout: 10)
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        do {
            _ = try supervisor.startChatGeneration(IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.INVALID_MODEL_ID,
                requestId: 1))
            Issue.record("the invalid model should fail before generation")
        } catch let generationStartError as GenerationStartError {
            #expect(
                generationStartError == .modelLoadFailed(
                    modelLoadFailureReason: "model artifact validation failed: OptiQ metadata uses unsupported 2-bit quantization"))
        }
        let afterRejectionSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot()
        #expect(afterRejectionSnapshot.status == .ready)
        #expect(afterRejectionSnapshot.readyModelId == nil)
        #expect(afterRejectionSnapshot.effectiveMlxMemoryCeilingBytes == 40_000_000_000)

        let recoveredGenerationEvents: Array<ChatGenerationStreamEvent> =
            try supervisor.startChatGeneration(IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.REQUESTED_MODEL_ID,
                requestId: 1))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            recoveredGenerationEvents,
            generationLabel: "valid model after rejected load")
        _ = try supervisor.shutdown()
    }

    @Test
    func should_reject_an_oversized_generation_command_without_terminating_the_loaded_worker() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchUnconfiguredIdleWorker(
                modelPolicyCatalog: [
                    IdleWorkerJourneySupport.REQUESTED_MODEL_ID:
                        IdleWorkerJourneySupport.runtimeModelPolicy(
                            IdleWorkerJourneySupport.REQUESTED_MODEL_ID,
                            modelDirectory: "/models/requested-model",
                            maximumOutputTokens: 128),
                ],
                modelLoadTimeout: 10)
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        let oversizedImageBytes: Array<UInt8> = Array<UInt8>(
            repeating: 0,
            count: IpcFrameLimits.maximumIpcFrameBytes * 3 / 4)
        let oversizedGenerationCommand: ChatGenerationCommand = ChatGenerationCommand(
            requestId: RequestId(rawRequestId: 1),
            model: IdleWorkerJourneySupport.REQUESTED_MODEL_ID,
            messages: [.user(
                content: "Describe this image.",
                images: [ChatImageInput(mimeType: "image/png", decodedBytes: oversizedImageBytes)])],
            tools: [],
            toolChoice: .none,
            settings: ChatGenerationSettings(
                maxOutputTokens: 1,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            structuredGeneration: nil)

        do {
            _ = try supervisor.startChatGeneration(oversizedGenerationCommand)
            Issue.record("the oversized command should be rejected before dispatch")
        } catch let generationStartError as GenerationStartError {
            guard case let .requestTooLarge(actualIpcMessageBytes, maximumIpcMessageBytes) =
                generationStartError else {
                Issue.record(
                    Comment(stringLiteral: "expected requestTooLarge, received \(generationStartError)"))
                return
            }
            #expect(maximumIpcMessageBytes == IpcFrameLimits.maximumIpcFrameBytes)
            #expect(actualIpcMessageBytes > IpcFrameLimits.maximumIpcFrameBytes)
        }

        let followupGenerationEvents: Array<ChatGenerationStreamEvent> =
            try supervisor.startChatGeneration(IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.REQUESTED_MODEL_ID,
                requestId: 1))
        IdleWorkerJourneySupport.assertGenerationCompleted(
            followupGenerationEvents,
            generationLabel: "follow-up after oversized rejection")
        _ = try supervisor.shutdown()
    }

    @Test
    func should_reject_an_unmapped_model_without_forwarding_generation_to_the_idle_worker() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchUnconfiguredIdleWorker(
                modelPolicyCatalog: [:],
                modelLoadTimeout: 10)
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor

        do {
            _ = try supervisor.startChatGeneration(IdleWorkerJourneySupport.chatCommand(
                modelId: "astronomical/unknown-model",
                requestId: 1))
            Issue.record("an unmapped model should be rejected before dispatch")
        } catch let generationStartError as GenerationStartError {
            #expect(generationStartError == GenerationStartError.workerUnavailable)
        }
        let workerHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot()
        #expect(workerHealthSnapshot.status == .ready)
        #expect(workerHealthSnapshot.readyModelId == nil)
        _ = try supervisor.shutdown()
    }

    @Test
    func should_bound_the_time_waiting_for_a_requested_model_to_load() throws {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchUnconfiguredIdleWorker(
                modelPolicyCatalog: [
                    IdleWorkerJourneySupport.HANGING_MODEL_ID:
                        IdleWorkerJourneySupport.runtimeModelPolicy(
                            IdleWorkerJourneySupport.HANGING_MODEL_ID,
                            modelDirectory: "/models/hanging-model",
                            maximumOutputTokens: 128),
                ],
                modelLoadTimeout: 2)
        defer { harness.dispose() }
        let supervisor: WorkerSupervisor = harness.supervisor
        let hangingModelStartAt: Date = Date()
        do {
            _ = try supervisor.startChatGeneration(IdleWorkerJourneySupport.chatCommand(
                modelId: IdleWorkerJourneySupport.HANGING_MODEL_ID,
                requestId: 1))
            Issue.record("a hanging model load should fail within the bound")
        } catch let generationStartError as GenerationStartError {
            #expect(generationStartError == GenerationStartError.workerUnavailable)
        }
        let hangingModelElapsedSeconds: TimeInterval = Date().timeIntervalSince(hangingModelStartAt)
        #expect(
            hangingModelElapsedSeconds < 5,
            "the hanging model load must stay bounded, took \(hangingModelElapsedSeconds)s")
        #expect(supervisor.workerHealthSnapshot().status == .unavailable)
    }
}
