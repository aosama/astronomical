import Foundation

import Testing;

import AstronomicalCli;
import IpcProtocol;
import JourneyCategories;

@testable import AstronomicalCli;

/// Hermetic `astronomical models` journeys, porting models_command.rs.
@Suite(.serialized, .tags(.hermeticJourney))
final class ModelsCommandTests {

    @Test
    func should_parse_models_subcommands() {
        #expect(CliJourneySupport.parse(["models", "list"]) == .success(.models(.list)));
        #expect(CliJourneySupport.parse(["models", "supported"]) == .success(.models(.supported)));
        #expect(CliJourneySupport.parse(["models", "default"]) == .success(.models(.default(modelId: nil))));
        #expect(CliJourneySupport.parse(["models", "default", "test/model"])
            == .success(.models(.default(modelId: "test/model"))));
        #expect(CliJourneySupport.parse(["models", "download", "test/model"])
            == .success(.models(.download(modelId: "test/model"))));
    }

    @Test
    func should_reject_models_without_a_subcommand_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["models"]) else {
            Issue.record("models without a subcommand must be a usage error");
            return;
        }
        #expect(usageError == .modelsSubcommandRequired);
    }

    @Test
    func should_reject_models_with_an_unknown_subcommand_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["models", "reboot"]) else {
            Issue.record("an unknown models subcommand must be a usage error");
            return;
        }
        #expect(usageError == .unknownModelsSubcommand("reboot"));
    }

    @Test
    func should_reject_models_download_without_a_model_id_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["models", "download"]) else {
            Issue.record("models download without a model id must be a usage error");
            return;
        }
        #expect(usageError == .modelsDownloadModelRequired);
    }

    @Test
    func should_list_installed_models_with_the_resident_marker() throws {
        try CliJourneySupport.withStubDaemon("models-list", CliJourneySupport.catalogStubConfig()) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let modelsOutcome: Result<Void, Error> = CliJourneySupport.runModels(
                .list,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = modelsOutcome else {
                Issue.record("models list should complete against the stub daemon: \(modelsOutcome)");
                return;
            }
            let renderedStdout: String = stdout.text;
            #expect(renderedStdout.contains("test/local-chatter"));
            #expect(renderedStdout.contains("test/local-embedder"));
            let residentLine: Substring = renderedStdout.split(separator: "\n").first { (line: Substring) -> Bool in
                return line.contains("test/local-chatter");
            } ?? "";
            #expect(residentLine.hasPrefix("*"), "the resident model line should carry the marker");
            _ = stubDaemon;
        }
    }

    @Test
    func should_render_the_catalog_with_local_states() throws {
        try CliJourneySupport.withStubDaemon("models-supported", CliJourneySupport.catalogStubConfig()) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let modelsOutcome: Result<Void, Error> = CliJourneySupport.runModels(
                .supported,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = modelsOutcome else {
                Issue.record("models supported should complete against the stub daemon: \(modelsOutcome)");
                return;
            }
            let renderedStdout: String = stdout.text;
            #expect(renderedStdout.contains("ready-model") && renderedStdout.contains("ready"));
            #expect(renderedStdout.contains("half-model") && renderedStdout.contains("downloading"));
            #expect(renderedStdout.contains("absent-model") && renderedStdout.contains("not on this Mac"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_show_and_set_the_default_model() throws {
        try CliJourneySupport.withStubDaemon("models-default", CliJourneySupport.catalogStubConfig()) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let showOutcome: Result<Void, Error> = CliJourneySupport.runModels(
                .default(modelId: nil),
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = showOutcome else {
                Issue.record("models default (show) should complete: \(showOutcome)");
                return;
            }
            #expect(stdout.text.contains("default model: test/local-chatter"));

            stdout.clear();
            let setOutcome: Result<Void, Error> = CliJourneySupport.runModels(
                .default(modelId: "test/ready-model"),
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = setOutcome else {
                Issue.record("models default (set) should complete: \(setOutcome)");
                return;
            }
            #expect(stdout.text.contains("default model: test/ready-model"));

            stdout.clear();
            let showAgainOutcome: Result<Void, Error> = CliJourneySupport.runModels(
                .default(modelId: nil),
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = showAgainOutcome else {
                Issue.record("models default (show again) should complete: \(showAgainOutcome)");
                return;
            }
            #expect(stdout.text.contains("default model: test/ready-model"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_download_a_missing_model_before_persisting_the_default() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.catalogEntries = [.chat("test/downloaded-model", "downloaded-model", false)];
        stubConfig.catalogEntriesAfterDownload = [.chat("test/downloaded-model", "downloaded-model", true)];
        stubConfig.downloadJobs = [
            stubDownloadJob("test/downloaded-model", "downloading", 500_000_000, 2_000_000_000, nil),
            nil,
        ];
        try CliJourneySupport.withStubDaemon("models-default-dl", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let modelsOutcome: Result<Void, Error> = CliJourneySupport.runModels(
                .default(modelId: "test/downloaded-model"),
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = modelsOutcome else {
                Issue.record("setting the default should download the missing model first: \(modelsOutcome)");
                return;
            }
            #expect(stdout.text.contains("default model: test/downloaded-model"));
            #expect(stderr.text.contains("downloading test/downloaded-model"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_download_a_missing_model_with_live_progress() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.catalogEntries = [.chat("test/downloaded-model", "downloaded-model", false)];
        stubConfig.catalogEntriesAfterDownload = [.chat("test/downloaded-model", "downloaded-model", true)];
        stubConfig.downloadJobs = [
            stubDownloadJob("test/downloaded-model", "downloading", 500_000_000, 2_000_000_000, nil),
            nil,
        ];
        try CliJourneySupport.withStubDaemon("models-download", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let modelsOutcome: Result<Void, Error> = CliJourneySupport.runModels(
                .download(modelId: "test/downloaded-model"),
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = modelsOutcome else {
                Issue.record("the download journey should complete with live progress: \(modelsOutcome)");
                return;
            }
            #expect(stderr.text.contains("downloading test/downloaded-model"));
            #expect(stderr.text.contains("test/downloaded-model is available"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_fail_to_download_a_model_outside_the_catalog() throws {
        try CliJourneySupport.withStubDaemon("models-outside", CliJourneySupport.catalogStubConfig()) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let modelsOutcome: Result<Void, Error> = CliJourneySupport.runModels(
                .download(modelId: "test/no-such-model"),
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case let .failure(modelsError) = modelsOutcome,
                  case let ModelsVerbError.modelUnavailable(reason) = modelsError else {
                Issue.record("a model outside the catalog must fail the download verb: \(modelsOutcome)");
                return;
            }
            #expect(reason.contains("not in the release catalog"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_report_a_missing_daemon_as_not_running() throws {
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let missingSocketPath: String = CliJourneySupport.freshTestDirectory("models-missing") + "/ipc.sock";
        let modelsOutcome: Result<Void, Error> = CliJourneySupport.runModels(
            .list,
            candidateSocketPaths: [missingSocketPath],
            stdout: stdout,
            stderr: stderr
        );
        guard case let .failure(modelsError) = modelsOutcome,
              case ModelsVerbError.daemonNotRunning = modelsError else {
            Issue.record("a missing daemon must fail the models verb: \(modelsOutcome)");
            return;
        }
    }
}
