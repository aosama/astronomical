import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import Supervisor;
import JourneyCategories;

@testable import AstronomicalCli;

/// Hermetic coverage for the `astronomical status` journey: the probe and
/// renderer against a real daemon IPC service on a temporary socket. Paths
/// stay short because a unix socket path must fit sun_path.
@Suite(.serialized, .tags(.hermeticJourney))
final class StatusCommandTests {

    @Test
    func should_report_the_stubbed_worker_against_a_live_daemon() throws {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "acli-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let service: DaemonIpcService = try DaemonIpcService.start(
            instancePaths: instancePaths,
            healthProvider: { return DaemonStatusReport(workerStatus: .ready, readyModelId: nil); });
        defer {
            service.shutdown();
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }

        let report: String = try StatusCommand.run(instancePaths: instancePaths);
        #expect(report.contains("worker:   ready"), "report should name the worker state: \(report)");
        #expect(report.contains("download:"), "report should carry the download line: \(report)");
    }

    @Test
    func should_report_a_daemon_that_is_not_running() throws {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "acli-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        defer {
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }

        let report: String = try StatusCommand.run(instancePaths: instancePaths);
        #expect(report.contains("daemon is not running"), "report should explain the missing daemon: \(report)");
    }

    @Test
    func should_reject_an_unexpected_response_frame() throws {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "acli-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        // A daemon whose health provider never lets the status verb through is
        // simulated by rejecting on the wire: the service shell answers
        // requestRejected for verbs it has not wired, and downloadStatus is
        // one of them, so the probe must surface that as a rejection.
        let service: DaemonIpcService = try DaemonIpcService.start(
            instancePaths: instancePaths,
            healthProvider: { return DaemonStatusReport.unavailable(); });
        defer {
            service.shutdown();
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }

        do {
            _ = try DaemonProbe.activeDownloadJob(
                candidateSocketPaths: [instancePaths.ipcSocketFilePath.string]);
            Issue.record("expected a daemon rejection");
        } catch let probeError as DaemonProbeError {
            guard case let .daemonRejected(reason) = probeError else {
                Issue.record(Comment(stringLiteral: "expected a daemon rejection, got \(probeError)"));
                return;
            }
            #expect(!reason.isEmpty);
        }
    }
}
