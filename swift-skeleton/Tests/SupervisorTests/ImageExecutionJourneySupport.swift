import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;

@testable import Supervisor;

/**
 * Shared harness for the image execution journeys, migrating the fixture
 * helpers of apps/supervisor/tests/hermetic/image_generation.rs: one
 * scripted image-capable worker with the chat fixtures the FIFO journey
 * queues, the prompt-keyed image command shape, the thread-owned in-flight
 * image outcome, and the bounded condition waits every journey reuses.
 */
enum ImageExecutionJourneySupport {


    static func launchImageWorker(
        cancellationAcknowledgementTimeout: TimeInterval = WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds
    ) throws -> IdleWorkerJourneySupport.IdleWorkerHarness {
        return try IdleWorkerJourneySupport.launchConfiguredIdleWorker(
            modelPolicyCatalog: [
                IdleWorkerJourneySupport.IMAGE_MODEL_ID:
                    IdleWorkerJourneySupport.imageRuntimeModelPolicy(
                        IdleWorkerJourneySupport.IMAGE_MODEL_ID,
                        modelDirectory: "/models/image-generation-model"),
                "astronomical/delayed-fragment-chat-fixture":
                    IdleWorkerJourneySupport.runtimeModelPolicy(
                        "astronomical/delayed-fragment-chat-fixture",
                        modelDirectory: "/models/delayed-fragment-chat-fixture",
                        maximumOutputTokens: 128),
                "astronomical/test-worker":
                    IdleWorkerJourneySupport.runtimeModelPolicy(
                        "astronomical/test-worker",
                        modelDirectory: "/models/test-worker",
                        maximumOutputTokens: 128),
            ],
            modelLoadTimeout: 10,
            startupConfigurationBuilder: { (journeyDirectoryPath: String) -> WorkerStartupConfiguration? in
                return nil
            },
            workerArguments: [],
            cancellationAcknowledgementTimeout: cancellationAcknowledgementTimeout);
    }

    static func imageCommand(requestId: UInt64, prompt: String) -> ImageGenerationCommand {
        return ImageGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: IdleWorkerJourneySupport.IMAGE_MODEL_ID,
            prompt: prompt,
            settings: ImageGenerationSettings(
                widthPixels: 64,
                heightPixels: 64,
                steps: 4,
                guidanceThousandths: 1_000,
                seed: 7));
    }

    static func startImageOnThread(
        _ supervisor: WorkerSupervisor,
        imageCommand: ImageGenerationCommand,
        timeouts: ImageGenerationTimeouts = ImageGenerationTimeouts.default,
        isClientAbandoned: (() -> Bool)? = nil
    ) -> ImageGenerationJourneyOutcome {
        let imageOutcome: ImageGenerationJourneyOutcome = ImageGenerationJourneyOutcome(
            workerThread: Thread());
        let workerThread: Thread = Thread(block: { () -> Void in
            do {
                let imageOutput: ImageGenerationOutput = try supervisor.startImageGeneration(
                    imageCommand,
                    timeouts: timeouts,
                    isClientAbandoned: isClientAbandoned);
                imageOutcome.record(output: imageOutput);
            } catch let imageError {
                imageOutcome.record(error: imageError);
            }
        });
        workerThread.name = "astronomical-image-journey";
        imageOutcome.workerThread = workerThread;
        workerThread.start();
        return imageOutcome;
    }

    static func awaitImageOutcome(
        _ supervisor: WorkerSupervisor,
        imageCommand: ImageGenerationCommand,
        timeouts: ImageGenerationTimeouts,
        journeyLabel: String
    ) -> Result<ImageGenerationOutput, Error>? {
        let imageOutcome: ImageGenerationJourneyOutcome = ImageExecutionJourneySupport.startImageOnThread(
            supervisor,
            imageCommand: imageCommand,
            timeouts: timeouts);
        return imageOutcome.awaitOutcome(deadlineSeconds: 5, journeyLabel: journeyLabel);
    }

    static func captureError<T>(_ attemptedOperation: () throws -> T) -> Error? {
        do {
            _ = try attemptedOperation();
            return nil;
        } catch let attemptedError {
            return attemptedError;
        }
    }

    static func waitUntilTrue(_ isSatisfied: () -> Bool) -> Bool {
        let conditionDeadline: Date = Date().addingTimeInterval(5);
        while (Date() < conditionDeadline) {
            if isSatisfied() {
                return true;
            }
            Thread.sleep(forTimeInterval: 0.02);
        }
        return false;
    }

    static func firstJsonRow(inFileAtPath filePath: String) throws -> [String: Any] {
        let logDocument: String = try String(
            contentsOf: URL(fileURLWithPath: filePath),
            encoding: String.Encoding.utf8);
        let logLines: Array<Substring> = logDocument.split(separator: "\n");
        guard let firstLogLine: Substring = logLines.first else {
            throw ImageGenerationJourneyFailure.missingPerformanceRow;
        }
        guard let parsedRow: [String: Any] = try JSONSerialization.jsonObject(
            with: Data(firstLogLine.utf8)) as? [String: Any] else {
            throw ImageGenerationJourneyFailure.missingPerformanceRow;
        }
        return parsedRow;
    }
}

/// One chat stream handle created on a worker thread, released by the
/// journey thread once the queue observation is complete.
final class ChatStreamHandleBox: @unchecked Sendable {

    private let boxLock: NSLock;
    private var streamHandle: ChatGenerationStreamHandle?;

    init() {
        self.boxLock = NSLock();
        self.streamHandle = nil;
    }

    func store(_ chatStreamHandle: ChatGenerationStreamHandle) -> Void {
        self.boxLock.lock();
        self.streamHandle = chatStreamHandle;
        self.boxLock.unlock();
    }

    func abandon() -> Void {
        self.boxLock.lock();
        let storedHandle: ChatGenerationStreamHandle? = self.streamHandle;
        self.streamHandle = nil;
        self.boxLock.unlock();
        storedHandle?.abandon();
    }
}

/// The disconnect probe one image journey raises when its client stops
/// waiting; the executor diverts into the bounded cancellation path.
final class ImageAbandonmentFlag: @unchecked Sendable {

    private let flagLock: NSLock;
    private var abandoned: Bool;

    init() {
        self.flagLock = NSLock();
        self.abandoned = false;
    }

    var isAbandoned: Bool {
        self.flagLock.lock();
        defer { self.flagLock.unlock() }
        return self.abandoned;
    }

    func abandon() -> Void {
        self.flagLock.lock();
        self.abandoned = true;
        self.flagLock.unlock();
    }
}

enum ImageGenerationJourneyFailure: Error {

    case missingPerformanceRow;
}
