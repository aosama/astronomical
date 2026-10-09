import Foundation;

import Testing;

import JourneyCategories;

extension MlxGpuJourneyContainer {

    @Suite(.tags(.hermeticJourney))
    final class MlxGpuSerializationSecondContractTests {

        @Test(.timeLimit(.minutes(1)))
        func should_run_without_an_active_gpu_sibling() async throws {
            let activeJourneyCount: Int = await MlxGpuSerializationProbe.enterJourney();
            try await Task.sleep(for: .milliseconds(250));
            await MlxGpuSerializationProbe.leaveJourney();
            #expect(activeJourneyCount == 1,
                "a sibling MLX/GPU journey must not overlap this journey");
        }
    }
}
