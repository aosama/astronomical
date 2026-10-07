import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Worker-boundary acceptance coverage for shared admission and image
 * finalization, migrating apps/supervisor/tests/hermetic/image_generation.rs
 * and image_memory_snapshot.rs: chat and image requests share one FIFO with
 * one capacity, a disconnected or stalled image is cancelled through the
 * bounded shared path, completed bytes stay private until cleanup
 * finalization, and every protocol breach is contained.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class ImageGenerationExecutionJourneyTests {

    @Test
    func should_share_fifo_order_and_capacity_between_chat_and_image_requests() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let activeChatHandle: ChatGenerationStreamHandle = try journey.supervisor.startChatGenerationStream(
            IdleWorkerJourneySupport.chatCommand(
                modelId: "astronomical/delayed-fragment-chat-fixture",
                requestId: 1),
            onEvent: { (streamEvent: ChatGenerationStreamEvent) -> Void in
                return;
            });
        let queuedImageOutcome: ImageGenerationJourneyOutcome =
            ImageExecutionJourneySupport.startImageOnThread(
                journey.supervisor,
                imageCommand: ImageExecutionJourneySupport.imageCommand(
                    requestId: 100,
                    prompt: "completed-image"));
        Thread.sleep(forTimeInterval: 0.05);
        #expect(queuedImageOutcome.observedOutcome == nil,
            "the queued image must wait for the active chat to release the worker");

        // Queued chats park inside admission on their own threads — the
        // stream call blocks the caller until its ticket is served.
        var queuedChatBoxes: Array<ChatStreamHandleBox> = Array<ChatStreamHandleBox>();
        for queuePosition in 1..<GenerationQueueDepth.maximumWaiterCount {
            let queuedChatBox: ChatStreamHandleBox = ChatStreamHandleBox();
            queuedChatBoxes.append(queuedChatBox);
            let queuedChatThread: Thread = Thread(block: { () -> Void in
                do {
                    let queuedChatHandle: ChatGenerationStreamHandle =
                        try journey.supervisor.startChatGenerationStream(
                            IdleWorkerJourneySupport.chatCommand(
                                modelId: "astronomical/test-worker",
                                requestId: UInt64(queuePosition + 1)),
                            onEvent: { (streamEvent: ChatGenerationStreamEvent) -> Void in
                                return;
                            });
                    queuedChatBox.store(queuedChatHandle);
                } catch {
                    // A queued chat that never reaches its turn observes the
                    // journey's shutdown; its outcome is unobserved, exactly
                    // like the Rust spawned tasks dropped at test end.
                }
            });
            queuedChatThread.name = "astronomical-image-journey-queued-chat";
            queuedChatThread.start();
        }
        Thread.sleep(forTimeInterval: 0.05);
        let capacityOutcome: Error? = ImageExecutionJourneySupport.captureError({ () throws -> ImageGenerationOutput in
            return try journey.supervisor.startImageGeneration(
                ImageExecutionJourneySupport.imageCommand(requestId: 110, prompt: "completed-image"));
        });
        #expect(capacityOutcome as? GenerationStartError == .capacityUnavailable,
            "the shared queue must reject beyond its waiter capacity");

        activeChatHandle.abandon();
        let queuedImageResult: Result<ImageGenerationOutput, Error>? =
            queuedImageOutcome.awaitOutcome(deadlineSeconds: 2, journeyLabel: "queued image admission");
        guard case .success = queuedImageResult else {
            Issue.record(Comment(stringLiteral: "the first queued modality must be the image: \(String(describing: queuedImageResult))"));
            return;
        }
        for queuedChatBox in queuedChatBoxes {
            queuedChatBox.abandon();
        }
    }

    @Test
    func should_cancel_a_disconnected_image_and_reuse_the_worker() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let abandonedByClient: ImageAbandonmentFlag = ImageAbandonmentFlag();
        let disconnectedImageOutcome: ImageGenerationJourneyOutcome =
            ImageExecutionJourneySupport.startImageOnThread(
                journey.supervisor,
                imageCommand: ImageExecutionJourneySupport.imageCommand(
                    requestId: 100,
                    prompt: "delayed-image-generation-fixture"),
                isClientAbandoned: { () -> Bool in
                    return abandonedByClient.isAbandoned;
                });
        Thread.sleep(forTimeInterval: 0.1);
        abandonedByClient.abandon();

        let cancelledOutcome: Result<ImageGenerationOutput, Error>? =
            disconnectedImageOutcome.awaitOutcome(
                deadlineSeconds: 2,
                journeyLabel: "disconnect cancellation should remain bounded");
        #expect(cancelledOutcome != nil);
        #expect(
            journey.supervisor.workerHealthSnapshot().status == WorkerHealthStatus.ready,
            "an acknowledged cancellation must keep the worker ready");
        let followupOutput: ImageGenerationOutput = try journey.supervisor.startImageGeneration(
            ImageExecutionJourneySupport.imageCommand(requestId: 101, prompt: "completed-image"));
        #expect(followupOutput.resultMetadata.steps == 4);
    }

    @Test
    func should_release_a_failed_image_only_after_finalization_and_reuse_the_worker() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let failedOutcome: Error? = ImageExecutionJourneySupport.captureError({ () throws -> ImageGenerationOutput in
            return try journey.supervisor.startImageGeneration(
                ImageExecutionJourneySupport.imageCommand(
                    requestId: 102,
                    prompt: "failed-image-generation-fixture"));
        });
        #expect(failedOutcome as? ImageGenerationExecutionError == .workerFailure(.cancelled),
            "the failed fixture must surface its cancelled failure");

        let followupOutput: ImageGenerationOutput = try journey.supervisor.startImageGeneration(
            ImageExecutionJourneySupport.imageCommand(requestId: 103, prompt: "completed-image"));
        #expect(followupOutput.resultMetadata.steps == 4);
    }

    @Test
    func should_hold_completed_image_bytes_until_cleanup_finalization() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let heldImageOutcome: ImageGenerationJourneyOutcome =
            ImageExecutionJourneySupport.startImageOnThread(
                journey.supervisor,
                imageCommand: ImageExecutionJourneySupport.imageCommand(
                    requestId: 104,
                    prompt: "completion-before-finalization-fixture"));
        Thread.sleep(forTimeInterval: 0.05);
        #expect(heldImageOutcome.observedOutcome == nil,
            "completed bytes must stay private until cleanup finalization");

        let releasedOutcome: Result<ImageGenerationOutput, Error>? =
            heldImageOutcome.awaitOutcome(
                deadlineSeconds: 1,
                journeyLabel: "finalization should release the result");
        guard case .success = releasedOutcome else {
            Issue.record(Comment(stringLiteral: "finalization must release the held image: \(String(describing: releasedOutcome))"));
            return;
        }
    }

    @Test
    func should_attribute_image_performance_only_after_finalization() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let heldImageOutcome: ImageGenerationJourneyOutcome =
            ImageExecutionJourneySupport.startImageOnThread(
                journey.supervisor,
                imageCommand: ImageExecutionJourneySupport.imageCommand(
                    requestId: 107,
                    prompt: "completion-before-finalization-fixture"));
        Thread.sleep(forTimeInterval: 0.05);
        let logBeforeFinalization: String = (try? String(
            contentsOf: URL(fileURLWithPath: journey.journeyDirectoryPath + "/performance.jsonl"),
            encoding: String.Encoding.utf8)) ?? "";
        #expect(logBeforeFinalization.isEmpty,
            "the image performance row must wait for finalization");

        let releasedOutcome: Result<ImageGenerationOutput, Error>? =
            heldImageOutcome.awaitOutcome(
                deadlineSeconds: 1,
                journeyLabel: "finalization should release the attributed image");
        guard case .success = releasedOutcome else {
            Issue.record(Comment(stringLiteral: "the attributed image must complete: \(String(describing: releasedOutcome))"));
            return;
        }

        let imagePerformanceRow: [String: Any] = try ImageExecutionJourneySupport.firstJsonRow(
            inFileAtPath: journey.journeyDirectoryPath + "/performance.jsonl");
        #expect(imagePerformanceRow["operation"] as? String == "image_generation");
        #expect(imagePerformanceRow["request_id"] as? Int == 107);
        #expect(imagePerformanceRow["completion_outcome"] as? String == "completed");
        #expect(imagePerformanceRow["worker_reported_elapsed_millis"] as? Int == 30);
        #expect(imagePerformanceRow["total_elapsed_millis"] is NSNumber);
        for attributionField in [
            "queue_wait_elapsed_millis", "swap_load_elapsed_millis",
            "execution_elapsed_millis", "finalization_elapsed_millis",
        ] {
            #expect(imagePerformanceRow[attributionField] is NSNumber,
                "the \(attributionField) attribution field must be present");
        }
        var attributedElapsedMillis: UInt64 = 0;
        for attributionField in [
            "queue_wait_elapsed_millis", "swap_load_elapsed_millis",
            "execution_elapsed_millis", "finalization_elapsed_millis",
        ] {
            let attributedFieldMillis: UInt64 =
                (imagePerformanceRow[attributionField] as? NSNumber)?.uint64Value ?? 0;
            attributedElapsedMillis += attributedFieldMillis;
        }
        let totalElapsedMillis: UInt64 = (imagePerformanceRow["total_elapsed_millis"] as? NSNumber)?.uint64Value ?? 0;
        #expect(totalElapsedMillis >= attributedElapsedMillis);
        let encodedImageBytes: UInt64 = (imagePerformanceRow["encoded_image_bytes"] as? NSNumber)?.uint64Value ?? 0;
        #expect(encodedImageBytes > 0);
    }

    @Test
    func should_contain_an_image_request_without_a_bounded_cancellation_acknowledgement() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker(
                cancellationAcknowledgementTimeout: 0.1);
        defer { journey.dispose() }

        let abandonedByClient: ImageAbandonmentFlag = ImageAbandonmentFlag();
        let unacknowledgedImageOutcome: ImageGenerationJourneyOutcome =
            ImageExecutionJourneySupport.startImageOnThread(
                journey.supervisor,
                imageCommand: ImageExecutionJourneySupport.imageCommand(
                    requestId: 105,
                    prompt: "unacknowledged-image-cancellation-fixture"),
                isClientAbandoned: { () -> Bool in
                    return abandonedByClient.isAbandoned;
                });
        Thread.sleep(forTimeInterval: 0.1);
        abandonedByClient.abandon();
        _ = unacknowledgedImageOutcome.awaitOutcome(
            deadlineSeconds: 5,
            journeyLabel: "unacknowledged image containment");

        let replacementReady: Bool = ImageExecutionJourneySupport.waitUntilTrue({ () -> Bool in
            return journey.supervisor.workerHealthSnapshot().status == WorkerHealthStatus.ready;
        });
        #expect(replacementReady, "the replacement worker should serve again");
        let followupOutput: ImageGenerationOutput = try journey.supervisor.startImageGeneration(
            ImageExecutionJourneySupport.imageCommand(requestId: 108, prompt: "completed-image"));
        #expect(followupOutput.resultMetadata.steps == 4);
    }

    @Test
    func should_cancel_image_execution_and_progress_stalls_with_the_shared_bounded_path() throws {
        let boundedJourneys: Array<(UInt64, String, ImageGenerationTimeouts)> = [
            (109, "delayed-image-generation-fixture",
                ImageGenerationTimeouts(executionTimeoutSeconds: 1, progressStallTimeoutSeconds: 3)),
            (110, "progress-stall-image-fixture",
                ImageGenerationTimeouts(executionTimeoutSeconds: 3, progressStallTimeoutSeconds: 1)),
            (118, "duplicate-progress-stall-image-fixture",
                ImageGenerationTimeouts(executionTimeoutSeconds: 3, progressStallTimeoutSeconds: 1)),
        ];
        for (requestId, prompt, timeouts) in boundedJourneys {
            let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
                try ImageExecutionJourneySupport.launchImageWorker();
            defer { journey.dispose() }

            let boundedOutcome: Error? = ImageExecutionJourneySupport.captureError({ () throws -> ImageGenerationOutput in
                return try journey.supervisor.startImageGeneration(
                    ImageExecutionJourneySupport.imageCommand(requestId: requestId, prompt: prompt),
                    timeouts: timeouts,
                    isClientAbandoned: nil);
            });
            #expect(boundedOutcome as? ImageGenerationExecutionError == .deadlineExceeded,
                "the bounded image must cancel through the shared deadline path");

            let followupOutput: ImageGenerationOutput = try journey.supervisor.startImageGeneration(
                ImageExecutionJourneySupport.imageCommand(
                    requestId: requestId + 100,
                    prompt: "completed-image"));
            #expect(followupOutput.resultMetadata.steps == 4,
                "deadline cancellation must preserve worker reuse");
        }
    }

    @Test
    func should_refresh_the_stall_deadline_for_every_monotonic_image_progress_event() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let progressingOutcome: Result<ImageGenerationOutput, Error>? =
            ImageExecutionJourneySupport.awaitImageOutcome(
                journey.supervisor,
                imageCommand: ImageExecutionJourneySupport.imageCommand(
                    requestId: 114,
                    prompt: "elapsed-progress-refresh-image-fixture"),
                timeouts: ImageGenerationTimeouts(executionTimeoutSeconds: 3, progressStallTimeoutSeconds: 1),
                journeyLabel: "monotonic progress should keep the request alive");
        guard case .success = progressingOutcome else {
            Issue.record(Comment(stringLiteral: "the progressing image failed: \(String(describing: progressingOutcome))"));
            return;
        }
    }

    @Test
    func should_contain_an_image_completion_with_mismatched_guidance() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let containedOutcome: Error? = ImageExecutionJourneySupport.captureError({ () throws -> ImageGenerationOutput in
            return try journey.supervisor.startImageGeneration(
                ImageExecutionJourneySupport.imageCommand(
                    requestId: 115,
                    prompt: "guidance-mismatch-image-fixture"));
        });
        #expect(containedOutcome as? ImageGenerationExecutionError == .workerUnavailable,
            "a completion with mismatched guidance must be contained");
        #expect(
            journey.supervisor.workerHealthSnapshot().status == WorkerHealthStatus.unavailable,
            "the contained worker must report unavailable");
        _ = try? journey.supervisor.shutdown();
    }

    @Test
    func should_contain_malformed_or_dimension_mismatched_png_completions() throws {
        for (requestId, prompt) in [
            (UInt64(116), "malformed-png-image-fixture"),
            (UInt64(117), "png-dimension-mismatch-image-fixture"),
        ] {
            let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
                try ImageExecutionJourneySupport.launchImageWorker();
            defer { journey.dispose() }

            let containedOutcome: Error? = ImageExecutionJourneySupport.captureError({ () throws -> ImageGenerationOutput in
                return try journey.supervisor.startImageGeneration(
                    ImageExecutionJourneySupport.imageCommand(requestId: requestId, prompt: prompt));
            });
            #expect(containedOutcome as? ImageGenerationExecutionError == .workerUnavailable,
                "an invalid PNG completion must be contained");
            #expect(
                journey.supervisor.workerHealthSnapshot().status == WorkerHealthStatus.unavailable,
                "the contained worker must report unavailable");
            _ = try? journey.supervisor.shutdown();
        }
    }

    @Test
    func should_contain_every_non_monotonic_image_progress_dimension() throws {
        for (requestId, prompt) in [
            (UInt64(111), "phase-regression-image-fixture"),
            (UInt64(112), "step-regression-image-fixture"),
            (UInt64(113), "elapsed-regression-image-fixture"),
        ] {
            let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
                try ImageExecutionJourneySupport.launchImageWorker();
            defer { journey.dispose() }

            let containedOutcome: Error? = ImageExecutionJourneySupport.captureError({ () throws -> ImageGenerationOutput in
                return try journey.supervisor.startImageGeneration(
                    ImageExecutionJourneySupport.imageCommand(requestId: requestId, prompt: prompt));
            });
            #expect(containedOutcome as? ImageGenerationExecutionError == .workerUnavailable,
                "a non-monotonic progress sequence must be contained");
            #expect(
                journey.supervisor.workerHealthSnapshot().status == WorkerHealthStatus.unavailable,
                "the contained worker must report unavailable");
            _ = try? journey.supervisor.shutdown();
        }
    }

    @Test
    func should_keep_memory_cache_and_replacement_controls_busy_until_image_finalization() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let abandonedByClient: ImageAbandonmentFlag = ImageAbandonmentFlag();
        let activeImageOutcome: ImageGenerationJourneyOutcome =
            ImageExecutionJourneySupport.startImageOnThread(
                journey.supervisor,
                imageCommand: ImageExecutionJourneySupport.imageCommand(
                    requestId: 106,
                    prompt: "delayed-image-generation-fixture"),
                isClientAbandoned: { () -> Bool in
                    return abandonedByClient.isAbandoned;
                });
        Thread.sleep(forTimeInterval: 0.1);

        let memoryLimitOutcome: MlxMemoryLimitUpdateOutcome = try journey.supervisor.updateMlxMemoryLimit(
            32_000_000_000,
            configurationGeneration: "queued-memory-generation");
        #expect(memoryLimitOutcome == .queued);
        let cacheClearOutcome: PromptCacheClearOutcome = try journey.supervisor.clearPromptCache(modelId: nil);
        #expect(cacheClearOutcome == .queued);
        let replacementOutcome: Error? = ImageExecutionJourneySupport.captureError({ () throws -> WorkerRuntimeFeatureConfiguration in
            return try journey.supervisor.restartWorkerWithStartupConfiguration(
                candidateWorkerExecutablePath: IdleWorkerJourneySupport.locateBuiltExecutable(
                    executableName: IdleWorkerJourneySupport.WORKER_EXECUTABLE_NAME),
                candidateWorkerArguments: [],
                candidateModelPolicyCatalog: [:],
                candidateStartupConfiguration: WorkerStartupConfiguration(
                    configurationGeneration: "unused-generation",
                    globalPromptCacheRootDirectory: "/tmp/astronomical-supervisor-cache",
                    globalPromptCacheMaximumSizeBytes: 1_073_741_824,
                    persistentPromptCacheEnabled: true,
                    configuredMaximumMlxMemoryBytes: nil,
                    performanceAttributionEnabled: false,
                    loggingDirectory: "/tmp/astronomical-supervisor-logs",
                    loggingLevel: .info,
                    retainedLogFileCount: 3));
        });
        #expect(replacementOutcome as? WorkerControlError == .generationBusy,
            "an in-flight image must keep replacement controls busy");

        abandonedByClient.abandon();
        _ = activeImageOutcome.awaitOutcome(
            deadlineSeconds: 5,
            journeyLabel: "the busy-controls image should release");
        let workerReady: Bool = ImageExecutionJourneySupport.waitUntilTrue({ () -> Bool in
            return journey.supervisor.workerHealthSnapshot().status == WorkerHealthStatus.ready;
        });
        #expect(workerReady, "the worker should return to ready after the bounded release");
    }

}
