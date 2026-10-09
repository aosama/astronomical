import Foundation;

actor MlxGpuSerializationProbe {

    private static let sharedProbe: MlxGpuSerializationProbe = MlxGpuSerializationProbe();

    private var activeJourneyCount: Int = 0;

    internal static func enterJourney() async -> Int {
        return await Self.sharedProbe.enter();
    }

    internal static func leaveJourney() async -> Void {
        await Self.sharedProbe.leave();
    }

    private func enter() -> Int {
        self.activeJourneyCount += 1;
        return self.activeJourneyCount;
    }

    private func leave() -> Void {
        self.activeJourneyCount -= 1;
    }
}
