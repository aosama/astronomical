import Foundation;

import IpcProtocol;

/**
 * The scripted image terminal orderings of the supervisor test worker,
 * migrating apps/supervisor/tests/fixtures/scripted_worker_image.rs: each
 * prompt key selects the exact event sequence the journey needs — a
 * bounded stall, a monotonic progress refresh, a completion held until
 * cleanup finalization, or a deliberate protocol breach the supervisor
 * must contain.
 */
enum IdleWorkerImageScenario {

    static let DELAYED_IMAGE_PROMPT: String = "delayed-image-generation-fixture";
    static let UNACKNOWLEDGED_IMAGE_CANCELLATION_PROMPT: String =
        "unacknowledged-image-cancellation-fixture";
    static let PROGRESS_STALL_IMAGE_PROMPT: String = "progress-stall-image-fixture";
    static let DUPLICATE_PROGRESS_STALL_IMAGE_PROMPT: String = "duplicate-progress-stall-image-fixture";
    static let ELAPSED_PROGRESS_REFRESH_IMAGE_PROMPT: String = "elapsed-progress-refresh-image-fixture";
    static let PROGRESS_SNAPSHOT_IMAGE_PROMPT: String = "progress-snapshot-image-fixture";
    static let FAILED_IMAGE_PROMPT: String = "failed-image-generation-fixture";
    static let COMPLETION_BEFORE_FINALIZATION_PROMPT: String = "completion-before-finalization-fixture";
    static let PHASE_REGRESSION_IMAGE_PROMPT: String = "phase-regression-image-fixture";
    static let STEP_REGRESSION_IMAGE_PROMPT: String = "step-regression-image-fixture";
    static let ELAPSED_REGRESSION_IMAGE_PROMPT: String = "elapsed-regression-image-fixture";
    static let GUIDANCE_MISMATCH_IMAGE_PROMPT: String = "guidance-mismatch-image-fixture";
    static let MALFORMED_PNG_IMAGE_PROMPT: String = "malformed-png-image-fixture";
    static let PNG_DIMENSION_MISMATCH_IMAGE_PROMPT: String = "png-dimension-mismatch-image-fixture";
    static let FINALIZATION_DELAY_SECONDS: TimeInterval = 0.15;
    static let PROGRESS_REFRESH_GAP_SECONDS: TimeInterval = 0.6;
    static let DUPLICATE_PROGRESS_GAP_SECONDS: TimeInterval = 0.07;
    static let SNAPSHOT_HOLD_SECONDS: TimeInterval = 0.3;

    /// One in-flight image request that stays silent until the supervisor
    /// cancels it; `shouldAcknowledge` selects whether the cancel ever gets
    /// its terminal pair.
    final class PendingImageCancellation {

        let shouldAcknowledge: Bool;

        init(shouldAcknowledge: Bool) {
            self.shouldAcknowledge = shouldAcknowledge;
        }
    }

    /// Emits the scripted event sequence one image command selects by its
    /// prompt, returning the pending cancellation the worker now owes a
    /// later cancel command, or nil when the request already reached its
    /// terminal pair.
    static func emitScriptedImageSequence(
        _ generationCommand: ImageGenerationCommand,
        eventWriter: ProtocolWriter
    ) throws -> PendingImageCancellation? {
        switch (generationCommand.prompt) {
        case IdleWorkerImageScenario.DELAYED_IMAGE_PROMPT:
            return PendingImageCancellation(shouldAcknowledge: true);
        case IdleWorkerImageScenario.UNACKNOWLEDGED_IMAGE_CANCELLATION_PROMPT:
            return PendingImageCancellation(shouldAcknowledge: false);
        case IdleWorkerImageScenario.PROGRESS_STALL_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendProgress(
                generationCommand,
                phase: .preparing,
                completedSteps: 0,
                elapsedMillis: 1,
                mlxMemorySnapshot: nil,
                eventWriter: eventWriter);
            return PendingImageCancellation(shouldAcknowledge: true);
        case IdleWorkerImageScenario.DUPLICATE_PROGRESS_STALL_IMAGE_PROMPT:
            for _ in 0..<3 {
                try IdleWorkerImageScenario.sendProgress(
                    generationCommand,
                    phase: .preparing,
                    completedSteps: 0,
                    elapsedMillis: 1,
                    mlxMemorySnapshot: nil,
                    eventWriter: eventWriter);
                Thread.sleep(forTimeInterval: IdleWorkerImageScenario.DUPLICATE_PROGRESS_GAP_SECONDS);
            }
            return PendingImageCancellation(shouldAcknowledge: true);
        case IdleWorkerImageScenario.ELAPSED_PROGRESS_REFRESH_IMAGE_PROMPT:
            for (progressIndex, elapsedMillis) in [1, 2, 3].enumerated() {
                try IdleWorkerImageScenario.sendProgress(
                    generationCommand,
                    phase: .preparing,
                    completedSteps: 0,
                    elapsedMillis: UInt64(elapsedMillis),
                    mlxMemorySnapshot: nil,
                    eventWriter: eventWriter);
                if (progressIndex < 2) {
                    Thread.sleep(forTimeInterval: IdleWorkerImageScenario.PROGRESS_REFRESH_GAP_SECONDS);
                }
            }
            try IdleWorkerImageScenario.sendCompletedImage(
                generationCommand,
                finalizationDelaySeconds: 0,
                eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.PROGRESS_SNAPSHOT_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendProgress(
                generationCommand,
                phase: .denoising,
                completedSteps: 1,
                elapsedMillis: 5,
                mlxMemorySnapshot: WorkerMlxMemorySnapshot(
                    source: .imageGenerationStep,
                    activeMemoryBytes: 10_480_000_000,
                    allocatorCacheMemoryBytes: 120_000_000,
                    peakMemoryBytes: 10_600_000_000,
                    expertPayloadBytes: 0,
                    modelCorePayloadBytes: 0,
                    contextStatePayloadBytes: 0,
                    memoryCeilingUtilization: nil),
                eventWriter: eventWriter);
            Thread.sleep(forTimeInterval: IdleWorkerImageScenario.SNAPSHOT_HOLD_SECONDS);
            try IdleWorkerImageScenario.sendCompletedImage(
                generationCommand,
                finalizationDelaySeconds: 0,
                eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.FAILED_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendFailedImage(generationCommand.requestId, eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.COMPLETION_BEFORE_FINALIZATION_PROMPT:
            try IdleWorkerImageScenario.sendCompletedImage(
                generationCommand,
                finalizationDelaySeconds: IdleWorkerImageScenario.FINALIZATION_DELAY_SECONDS,
                eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.PHASE_REGRESSION_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendMalformedProgress(
                generationCommand,
                firstPhase: .decoding,
                secondPhase: .denoising,
                firstSteps: 2,
                secondSteps: 2,
                firstElapsedMillis: 20,
                secondElapsedMillis: 21,
                eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.STEP_REGRESSION_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendMalformedProgress(
                generationCommand,
                firstPhase: .denoising,
                secondPhase: .denoising,
                firstSteps: 2,
                secondSteps: 1,
                firstElapsedMillis: 20,
                secondElapsedMillis: 21,
                eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.ELAPSED_REGRESSION_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendMalformedProgress(
                generationCommand,
                firstPhase: .denoising,
                secondPhase: .denoising,
                firstSteps: 1,
                secondSteps: 2,
                firstElapsedMillis: 20,
                secondElapsedMillis: 19,
                eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.GUIDANCE_MISMATCH_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendCompletion(
                generationCommand,
                encodedBytes: IdleWorkerPngEncoder.encodeTruecolorPng(
                    widthPixels: generationCommand.settings.widthPixels,
                    heightPixels: generationCommand.settings.heightPixels),
                guidanceThousandths: generationCommand.settings.guidanceThousandths + 1,
                eventWriter: eventWriter);
            try IdleWorkerImageScenario.sendFinalizedImage(generationCommand.requestId, eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.MALFORMED_PNG_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendCompletion(
                generationCommand,
                encodedBytes: [137, 80, 78, 71],
                guidanceThousandths: generationCommand.settings.guidanceThousandths,
                eventWriter: eventWriter);
            return nil;
        case IdleWorkerImageScenario.PNG_DIMENSION_MISMATCH_IMAGE_PROMPT:
            try IdleWorkerImageScenario.sendCompletion(
                generationCommand,
                encodedBytes: IdleWorkerPngEncoder.encodeTruecolorPng(
                    widthPixels: generationCommand.settings.widthPixels / 2,
                    heightPixels: generationCommand.settings.heightPixels),
                guidanceThousandths: generationCommand.settings.guidanceThousandths,
                eventWriter: eventWriter);
            return nil;
        default:
            try IdleWorkerImageScenario.sendCompletedImage(
                generationCommand,
                finalizationDelaySeconds: 0,
                eventWriter: eventWriter);
            return nil;
        }
    }

    /// Resolves a pending image cancellation: the acknowledging shape is the
    /// failed-then-finalized pair the bounded cancel path waits for.
    static func acknowledgePendingCancellation(
        _ pendingImageCancellation: PendingImageCancellation,
        requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        if (pendingImageCancellation.shouldAcknowledge) {
            try IdleWorkerImageScenario.sendFailedImage(requestId, eventWriter: eventWriter);
        }
    }

    private static func sendCompletedImage(
        _ generationCommand: ImageGenerationCommand,
        finalizationDelaySeconds: TimeInterval,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try IdleWorkerImageScenario.sendProgress(
            generationCommand,
            phase: .denoising,
            completedSteps: generationCommand.settings.steps,
            elapsedMillis: 20,
            mlxMemorySnapshot: nil,
            eventWriter: eventWriter);
        try IdleWorkerImageScenario.sendCompletion(
            generationCommand,
            encodedBytes: IdleWorkerPngEncoder.encodeTruecolorPng(
                widthPixels: generationCommand.settings.widthPixels,
                heightPixels: generationCommand.settings.heightPixels),
            guidanceThousandths: generationCommand.settings.guidanceThousandths,
            eventWriter: eventWriter);
        Thread.sleep(forTimeInterval: finalizationDelaySeconds);
        try IdleWorkerImageScenario.sendFinalizedImage(generationCommand.requestId, eventWriter: eventWriter);
    }

    private static func sendCompletion(
        _ generationCommand: ImageGenerationCommand,
        encodedBytes: Array<UInt8>,
        guidanceThousandths: UInt32,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try eventWriter.sendEvent(.imageGenerationCompleted(
            requestId: generationCommand.requestId,
            generatedImage: GeneratedImage(
                mimeType: "image/png",
                encodedBytes: encodedBytes),
            resultMetadata: ImageGenerationResultMetadata(
                widthPixels: generationCommand.settings.widthPixels,
                heightPixels: generationCommand.settings.heightPixels,
                steps: generationCommand.settings.steps,
                guidanceThousandths: guidanceThousandths,
                seed: generationCommand.settings.seed,
                elapsedMillis: 25)));
    }

    private static func sendProgress(
        _ generationCommand: ImageGenerationCommand,
        phase: ImageGenerationPhase,
        completedSteps: UInt16,
        elapsedMillis: UInt64,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try eventWriter.sendEvent(.imageGenerationProgress(
            requestId: generationCommand.requestId,
            phase: phase,
            completedSteps: completedSteps,
            totalSteps: generationCommand.settings.steps,
            elapsedMillis: elapsedMillis,
            mlxMemorySnapshot: mlxMemorySnapshot));
    }

    private static func sendMalformedProgress(
        _ generationCommand: ImageGenerationCommand,
        firstPhase: ImageGenerationPhase,
        secondPhase: ImageGenerationPhase,
        firstSteps: UInt16,
        secondSteps: UInt16,
        firstElapsedMillis: UInt64,
        secondElapsedMillis: UInt64,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try IdleWorkerImageScenario.sendProgress(
            generationCommand,
            phase: firstPhase,
            completedSteps: firstSteps,
            elapsedMillis: firstElapsedMillis,
            mlxMemorySnapshot: nil,
            eventWriter: eventWriter);
        try IdleWorkerImageScenario.sendProgress(
            generationCommand,
            phase: secondPhase,
            completedSteps: secondSteps,
            elapsedMillis: secondElapsedMillis,
            mlxMemorySnapshot: nil,
            eventWriter: eventWriter);
    }

    private static func sendFailedImage(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try eventWriter.sendEvent(.imageGenerationFailed(
            requestId: requestId,
            reason: .cancelled));
        try IdleWorkerImageScenario.sendFinalizedImage(requestId, eventWriter: eventWriter);
    }

    private static func sendFinalizedImage(
        _ requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        try eventWriter.sendEvent(.imageGenerationFinalized(
            requestId: requestId,
            elapsedMillis: 30,
            mlxMemorySnapshot: WorkerMlxMemorySnapshot(
                source: .finalized,
                activeMemoryBytes: 96_000_000,
                allocatorCacheMemoryBytes: 0,
                peakMemoryBytes: 512_000_000,
                expertPayloadBytes: 0,
                modelCorePayloadBytes: 96_000_000,
                contextStatePayloadBytes: 0,
                memoryCeilingUtilization: nil)));
    }
}
