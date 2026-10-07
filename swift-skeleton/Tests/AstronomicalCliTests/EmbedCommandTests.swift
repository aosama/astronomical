import Foundation

import Testing;

import AstronomicalCli;
import IpcProtocol;
import JourneyCategories;

@testable import AstronomicalCli;

/// Hermetic `astronomical embed` journeys, porting embed_command.rs.
@Suite(.serialized, .tags(.hermeticJourney))
final class EmbedCommandTests {

    @Test
    func should_parse_embed_with_a_bare_text_argument() {
        #expect(CliJourneySupport.parse(["embed", "hello world"])
            == .success(.embed(EmbedArguments(text: "hello world", filePath: nil, modelId: nil))));
    }

    @Test
    func should_parse_embed_with_model_and_file() {
        #expect(CliJourneySupport.parse(["embed", "--model", "test/local-embedder", "--file", "input.txt"])
            == .success(.embed(EmbedArguments(
                text: nil,
                filePath: "input.txt",
                modelId: "test/local-embedder"
            ))));
    }

    @Test
    func should_parse_embed_with_no_input_for_stdin_mode() {
        #expect(CliJourneySupport.parse(["embed"])
            == .success(.embed(EmbedArguments(text: nil, filePath: nil, modelId: nil))));
    }

    @Test
    func should_reject_embed_with_an_unknown_argument_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["embed", "hello", "--unknown"]) else {
            Issue.record("an unknown embed argument must be a usage error");
            return;
        }
        #expect(usageError == .unknownArgument("--unknown"));
    }

    @Test
    func should_reject_embed_with_both_text_and_file_as_a_usage_error() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["embed", "hello", "--file", "input.txt"]) else {
            Issue.record("text plus --file must be a usage error");
            return;
        }
        #expect(usageError == .embedInputConflict);
    }

    @Test
    func should_embed_a_text_argument_into_one_json_vector_document() throws {
        try CliJourneySupport.withStubDaemon("embed-text", CliJourneySupport.embedderDefaultStubConfig()) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let embedOutcome: Result<Void, Error> = CliJourneySupport.runEmbed(
                text: "hello",
                filePath: nil,
                modelId: nil,
                standardInputText: "",
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = embedOutcome else {
                Issue.record("the embed journey should complete against the stub daemon: \(embedOutcome)");
                return;
            }
            let vectorDocument: Dictionary<String, Any> = try #require(
                JSONSerialization.jsonObject(with: Data(stdout.text.utf8)) as? Dictionary<String, Any>
            );
            #expect(vectorDocument["model"] as? String == "test/local-embedder");
            #expect(vectorDocument["embedding"] as? Array<Double> == [0.25, -0.5]);
            #expect(vectorDocument["input_tokens"] as? Int == 3);
            _ = stubDaemon;
        }
    }

    @Test
    func should_embed_text_from_a_file() throws {
        let inputFileDirectory: String = CliJourneySupport.freshTestDirectory("embed-file");
        defer { try? FileManager.default.removeItem(atPath: inputFileDirectory) }
        let inputFilePath: String = inputFileDirectory + "/input.txt";
        try Data("hello from a file".utf8).write(to: URL(fileURLWithPath: inputFilePath));
        try CliJourneySupport.withStubDaemon("embed-file-in", CliJourneySupport.embedderDefaultStubConfig()) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let embedOutcome: Result<Void, Error> = CliJourneySupport.runEmbed(
                text: nil,
                filePath: inputFilePath,
                modelId: nil,
                standardInputText: "",
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = embedOutcome else {
                Issue.record("the file-input embed journey should complete: \(embedOutcome)");
                return;
            }
            let vectorDocument: Dictionary<String, Any> = try #require(
                JSONSerialization.jsonObject(with: Data(stdout.text.utf8)) as? Dictionary<String, Any>
            );
            #expect(vectorDocument["model"] as? String == "test/local-embedder");
            #expect(vectorDocument["embedding"] as? Array<Double> == [0.25, -0.5]);
            _ = stubDaemon;
        }
    }

    @Test
    func should_embed_text_from_stdin() throws {
        try CliJourneySupport.withStubDaemon("embed-stdin", CliJourneySupport.embedderDefaultStubConfig()) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let embedOutcome: Result<Void, Error> = CliJourneySupport.runEmbed(
                text: nil,
                filePath: nil,
                modelId: nil,
                standardInputText: "hello from stdin",
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = embedOutcome else {
                Issue.record("the stdin embed journey should complete: \(embedOutcome)");
                return;
            }
            let vectorDocument: Dictionary<String, Any> = try #require(
                JSONSerialization.jsonObject(with: Data(stdout.text.utf8)) as? Dictionary<String, Any>
            );
            #expect(vectorDocument["model"] as? String == "test/local-embedder");
            _ = stubDaemon;
        }
    }

    @Test
    func should_report_a_missing_daemon_as_not_running_for_embed() throws {
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let missingSocketPath: String = CliJourneySupport.freshTestDirectory("embed-missing") + "/ipc.sock";
        let embedOutcome: Result<Void, Error> = CliJourneySupport.runEmbed(
            text: "hello",
            filePath: nil,
            modelId: nil,
            standardInputText: "",
            candidateSocketPaths: [missingSocketPath],
            stdout: stdout,
            stderr: stderr
        );
        guard case let .failure(embedError) = embedOutcome,
              case EmbedError.daemonNotRunning = embedError else {
            Issue.record("a missing daemon must fail with the not-running error: \(embedOutcome)");
            return;
        }
        #expect(String(describing: embedError).contains("Astronomical isn't running"));
    }

    @Test
    func should_reject_embed_when_no_model_is_available() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.installedModels = [.chat("test/local-chatter", isResident: true)];
        try CliJourneySupport.withStubDaemon("embed-no-model", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let embedOutcome: Result<Void, Error> = CliJourneySupport.runEmbed(
                text: nil,
                filePath: nil,
                modelId: nil,
                standardInputText: "hello",
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case let .failure(embedError) = embedOutcome,
                  case let EmbedError.modelUnavailable(reason) = embedError else {
                Issue.record("no embeddings model anywhere must fail with the pointer to the catalog: \(embedOutcome)");
                return;
            }
            #expect(reason.contains("astronomical models supported"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_refuse_a_chat_only_model_for_embed() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.installedModels = [.chat("test/local-chatter", isResident: true)];
        stubConfig.defaultModelId = "test/local-chatter";
        try CliJourneySupport.withStubDaemon("embed-capmismatch", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let embedOutcome: Result<Void, Error> = CliJourneySupport.runEmbed(
                text: nil,
                filePath: nil,
                modelId: nil,
                standardInputText: "hello",
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case let .failure(embedError) = embedOutcome,
                  case let EmbedError.modelUnavailable(reason) = embedError else {
                Issue.record("an embed request against a chat-only model must fail on capability: \(embedOutcome)");
                return;
            }
            #expect(reason.contains("not an embeddings model"));
            #expect(stdout.text.isEmpty, "no vector document may appear on a capability rejection");
            _ = stubDaemon;
        }
    }

    @Test
    func should_fail_with_the_worker_reason_when_embeddings_fail() throws {
        var stubConfig: StubDaemon.StubDaemonConfig = CliJourneySupport.embedderDefaultStubConfig();
        stubConfig.embeddingsOutcome = .contextLengthExceeded;
        try CliJourneySupport.withStubDaemon("embed-worker-failure", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let embedOutcome: Result<Void, Error> = CliJourneySupport.runEmbed(
                text: "a very long text",
                filePath: nil,
                modelId: nil,
                standardInputText: "",
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case let .failure(embedError) = embedOutcome,
                  case let EmbedError.embeddingsFailed(reason) = embedError,
                  case let EmbeddingsFailureReason.contextLengthExceeded(actualTokens, maximumTokens) = reason,
                  actualTokens == 5_000,
                  maximumTokens == 2_048 else {
                Issue.record("a worker-side embeddings failure must surface its typed reason: \(embedOutcome)");
                return;
            }
            #expect(String(describing: embedError).contains("context"));
            _ = stubDaemon;
        }
    }
}
