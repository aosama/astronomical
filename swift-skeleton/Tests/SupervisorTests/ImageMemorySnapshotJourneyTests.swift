import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * End-to-end coverage for live MLX memory snapshots published from image
 * render step boundaries, migrating
 * apps/supervisor/tests/hermetic/image_memory_snapshot.rs: the menu shows a
 * measured footprint mid-render, and the finalized cleanup snapshot replaces
 * it once the worker releases its request arrays.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class ImageMemorySnapshotJourneyTests {

    @Test
    func should_publish_a_live_mlx_memory_snapshot_from_image_progress_steps() throws {
        let journey: IdleWorkerJourneySupport.IdleWorkerHarness =
            try ImageExecutionJourneySupport.launchImageWorker();
        defer { journey.dispose() }

        let snapshotImageOutcome: ImageGenerationJourneyOutcome =
            ImageExecutionJourneySupport.startImageOnThread(
                journey.supervisor,
                imageCommand: ImageExecutionJourneySupport.imageCommand(
                    requestId: 120,
                    prompt: "progress-snapshot-image-fixture"));

        let renderSnapshotArrived: Bool = ImageExecutionJourneySupport.waitUntilTrue({ () -> Bool in
            return journey.supervisor.workerHealthSnapshot().latestMlxMemorySnapshot?.source
                == MlxMemorySnapshotSource.imageGenerationStep;
        });
        #expect(renderSnapshotArrived, "the image render snapshot should surface mid-generation");
        let renderSnapshot: WorkerMlxMemorySnapshot =
            journey.supervisor.workerHealthSnapshot().latestMlxMemorySnapshot!;
        #expect(renderSnapshot.activeMemoryBytes > 0);

        let snapshotOutcome: Result<ImageGenerationOutput, Error>? =
            snapshotImageOutcome.awaitOutcome(
                deadlineSeconds: 3,
                journeyLabel: "the snapshot image should finish");
        guard case .success = snapshotOutcome else {
            Issue.record(Comment(stringLiteral: "the snapshot image failed: \(String(describing: snapshotOutcome))"));
            return;
        }

        let finalizedSnapshotArrived: Bool = ImageExecutionJourneySupport.waitUntilTrue({ () -> Bool in
            return journey.supervisor.workerHealthSnapshot().latestMlxMemorySnapshot?.source
                == MlxMemorySnapshotSource.finalized;
        });
        #expect(finalizedSnapshotArrived, "the finalized cleanup snapshot should replace the render sample");
    }

}
