import Foundation

import Testing;

import AstronomicalCli;
import IpcProtocol;
import JourneyCategories;

@testable import AstronomicalCli;

/// Hermetic `astronomical status` journeys, porting status_command.rs against
/// the stub daemon on a real unix socket.
@Suite(.serialized, .tags(.hermeticJourney))
final class StatusCommandTests {

    @Test
    func should_parse_status_as_a_plain_verb() {
        #expect(CliJourneySupport.parse(["status"]) == .success(.status));
        guard case let .failure(usageError) = CliJourneySupport.parse(["status", "extra"]) else {
            Issue.record("a trailing argument after status must be a usage error");
            return;
        }
        #expect(usageError == .unknownArgument("extra"));
    }

    @Test
    func should_report_a_ready_worker_with_resident_model_and_default() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.installedModels = [.chat("test/local-chatter", isResident: true)];
        stubConfig.defaultModelId = "test/local-chatter";
        try CliJourneySupport.withStubDaemon("status-ready", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let report: String = try StatusCommand.run(candidateSocketPaths: [socketPath]);
            #expect(report.contains("worker:   ready (resident: test/local-chatter)"));
            #expect(report.contains("default:  test/local-chatter"));
            #expect(report.contains("download: none"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_report_a_loading_worker_without_a_resident_model() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.workerStatus = .loading;
        try CliJourneySupport.withStubDaemon("status-loading", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let report: String = try StatusCommand.run(candidateSocketPaths: [socketPath]);
            #expect(report.contains("worker:   loading"));
            #expect(report.contains("default:  none"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_report_an_unavailable_worker() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.workerStatus = .unavailable;
        try CliJourneySupport.withStubDaemon("status-unavailable", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let report: String = try StatusCommand.run(candidateSocketPaths: [socketPath]);
            #expect(report.contains("worker:   unavailable"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_report_an_active_download_with_decimal_gigabytes() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.downloadJobs = [stubDownloadJob("test/model", "downloading", 1_500_000_000, 2_000_000_000, nil)];
        try CliJourneySupport.withStubDaemon("status-download", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let report: String = try StatusCommand.run(candidateSocketPaths: [socketPath]);
            #expect(report.contains("download: test/model — downloading 1.5 GB / 2 GB"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_report_a_failed_download_job() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.downloadJobs = [stubDownloadJob("test/model", "failed", 0, 0, "checksum_mismatch")];
        try CliJourneySupport.withStubDaemon("status-failed", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let report: String = try StatusCommand.run(candidateSocketPaths: [socketPath]);
            #expect(report.contains("download: test/model (failed: checksum_mismatch)"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_report_a_missing_daemon_as_not_running_for_status() throws {
        let missingSocketPath: String = CliJourneySupport.freshTestDirectory("status-missing") + "/ipc.sock";
        do {
            _ = try StatusCommand.run(candidateSocketPaths: [missingSocketPath]);
            Issue.record("a missing daemon must fail the status verb");
        } catch let statusError as StatusError {
            guard case StatusError.daemonNotRunning = statusError else {
                Issue.record("a missing socket must surface as the not-running error: \(statusError)");
                return;
            }
            #expect(String(describing: statusError).contains("Astronomical isn't running"));
        }
    }
}
