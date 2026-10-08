import Foundation;

import Testing;

import ModelServing;

/// Cumulative macOS process input/output samples must yield monotonic
/// deltas and reject regressed counters as unavailable evidence, port of
/// crates/model-serving/tests/hermetic/macos_process_io.rs.
@Suite
final class MacosProcessIoTests {

    @Test
    func shouldCalculateMonotonicProcessIoDeltas() {
        let earlierSnapshot = MacosProcessIoSnapshot.fromCumulativeBytes(
            physicalDiskReadBytes: 1_000,
            physicalDiskWrittenBytes: 400);
        let laterSnapshot = MacosProcessIoSnapshot.fromCumulativeBytes(
            physicalDiskReadBytes: 1_750,
            physicalDiskWrittenBytes: 460);

        guard case .success(let processIoDelta) = laterSnapshot.deltaSince(earlierSnapshot) else {
            Issue.record("monotonic process I/O counters should produce a delta");
            return;
        }

        #expect(processIoDelta.physicalDiskReadBytes == 750);
        #expect(processIoDelta.physicalDiskWrittenBytes == 60);
    }

    @Test
    func shouldRejectARegressedProcessIoCounter() {
        let earlierSnapshot = MacosProcessIoSnapshot.fromCumulativeBytes(
            physicalDiskReadBytes: 1_000,
            physicalDiskWrittenBytes: 400);
        let laterSnapshot = MacosProcessIoSnapshot.fromCumulativeBytes(
            physicalDiskReadBytes: 999,
            physicalDiskWrittenBytes: 460);

        guard case .failure(.counterRegressed(
            let counterName,
            let earlierBytes,
            let laterBytes)) = laterSnapshot.deltaSince(earlierSnapshot) else {
            Issue.record("a regressed process I/O counter must not wrap");
            return;
        }

        #expect(counterName == "ri_diskio_bytesread");
        #expect(earlierBytes == 1_000);
        #expect(laterBytes == 999);
    }

    @Test
    func shouldSampleCurrentMacosProcessIo() {
        guard case .success(let processIoSnapshot) = MacosProcessIo.sampleCurrentProcessIo() else {
            Issue.record("the current macOS process should expose resource usage");
            return;
        }

        guard case .success(let unchangedDelta) = processIoSnapshot.deltaSince(processIoSnapshot) else {
            Issue.record("one snapshot compared with itself should remain monotonic");
            return;
        }

        #expect(unchangedDelta.physicalDiskReadBytes == 0);
        #expect(unchangedDelta.physicalDiskWrittenBytes == 0);
    }
}
