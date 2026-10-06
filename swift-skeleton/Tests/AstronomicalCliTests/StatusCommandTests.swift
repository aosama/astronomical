import XCTest;

import AstronomicalConfig;
import IpcProtocol;
import Supervisor;

@testable import AstronomicalCli;

/// Hermetic coverage for the `astronomical status` journey: the probe and
/// renderer against a real daemon IPC service on a temporary socket. Paths
/// stay short because a unix socket path must fit sun_path.
final class StatusCommandTests: XCTestCase {

    func testStatusReportsTheStubbedWorkerAgainstALiveDaemon() throws {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "acli-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let service: DaemonIpcService = try DaemonIpcService.start(
            instancePaths: instancePaths,
            healthProvider: { return DaemonWorkerStatus.ready; });
        defer {
            service.shutdown();
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }

        let report: String = try StatusCommand.run(instancePaths: instancePaths);
        XCTAssertTrue(report.contains("worker:   ready"), "report should name the worker state: \(report)");
        XCTAssertTrue(report.contains("download:"), "report should carry the download line: \(report)");
    }

    func testStatusReportsADaemonThatIsNotRunning() throws {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "acli-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        defer {
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }

        let report: String = try StatusCommand.run(instancePaths: instancePaths);
        XCTAssertTrue(report.contains("daemon is not running"), "report should explain the missing daemon: \(report)");
    }

    func testProbeRejectsAnUnexpectedResponseFrame() throws {
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
            healthProvider: { return DaemonWorkerStatus.unavailable; });
        defer {
            service.shutdown();
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
        }

        XCTAssertThrowsError(try DaemonProbe.activeDownloadJob(
            candidateSocketPaths: [instancePaths.ipcSocketFilePath.string])) { (thrownError: any Error) in
            guard case let DaemonProbeError.daemonRejected(reason) = thrownError else {
                return XCTFail("expected a daemon rejection, got \(thrownError)");
            }
            XCTAssertFalse(reason.isEmpty);
        }
    }
}
