import Foundation

import Testing;

import AstronomicalCli;
import IpcProtocol;
import JourneyCategories;

@testable import AstronomicalCli;

/// Hermetic `astronomical respond` journeys, porting respond_command.rs.
@Suite(.serialized, .tags(.hermeticJourney))
final class RespondCommandTests {

    @Test
    func should_report_a_missing_daemon_as_not_running() throws {
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let missingSocketPath: String = CliJourneySupport.freshTestDirectory("respond-not-running") + "/ipc.sock";

        let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespond(
            prompt: "Say hello",
            modelId: nil,
            noStream: false,
            candidateSocketPaths: [missingSocketPath],
            stdout: stdout,
            stderr: stderr
        );
        guard case let .failure(respondError) = respondOutcome,
              case RespondError.daemonNotRunning = respondError else {
            Issue.record("a missing socket must surface as the not-running error: \(respondOutcome)");
            return;
        }
        #expect(String(describing: respondError).contains("Astronomical isn't running"));
    }

    @Test
    func should_stream_generated_text_to_stdout() throws {
        try CliJourneySupport.withResidentDefaultStubDaemon(chatFragments: ["Hello", " world"]) { (socketPath: String, stubDaemon: StubDaemon, stdout: BufferedTextOutputWriter, stderr: BufferedTextOutputWriter) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespond(
                prompt: "Say hello",
                modelId: nil,
                noStream: false,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = respondOutcome else {
                Issue.record("the respond journey should complete against the stub daemon: \(respondOutcome)");
                return;
            }
            #expect(stdout.text == "Hello world");
            _ = stubDaemon;
        }
    }

    @Test
    func should_print_the_finished_answer_once_with_no_stream() throws {
        try CliJourneySupport.withResidentDefaultStubDaemon(chatFragments: ["Hello", " world"]) { (socketPath: String, stubDaemon: StubDaemon, stdout: BufferedTextOutputWriter, stderr: BufferedTextOutputWriter) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespond(
                prompt: "Say hello",
                modelId: nil,
                noStream: true,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = respondOutcome else {
                Issue.record("the respond journey should complete against the stub daemon: \(respondOutcome)");
                return;
            }
            #expect(stdout.text == "Hello world");
            _ = stubDaemon;
        }
    }

    @Test
    func should_fail_when_the_requested_model_is_unknown() throws {
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.installedModels = [.chat("test/other-model", isResident: true)];
        try CliJourneySupport.withStubDaemon("respond-unknown-model", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespond(
                prompt: "Say hello",
                modelId: "test/wanted-model",
                noStream: false,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case let .failure(respondError) = respondOutcome,
                  case let RespondError.modelUnavailable(reason) = respondError else {
                Issue.record("requesting a model the machine cannot serve must fail with the model id: \(respondOutcome)");
                return;
            }
            #expect(reason.contains("test/wanted-model"));
            #expect(reason.contains("did you mean: test/other-model"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_auto_load_the_builtin_default_when_nothing_is_resident() throws {
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.catalogEntries = [.chat("mlx-community/Qwen3.5-2B-4bit", "Qwen3.5-2B-4bit", true)];
        stubConfig.chatFragments = ["Loaded", " fine"];
        try CliJourneySupport.withStubDaemon("respond-auto-load", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespond(
                prompt: "Say hello",
                modelId: nil,
                noStream: false,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = respondOutcome else {
                Issue.record("a cold daemon with the built-in default on disk must serve the request: \(respondOutcome)");
                return;
            }
            #expect(stdout.text == "Loaded fine");
            _ = stubDaemon;
        }
    }

    @Test
    func should_download_a_missing_model_before_streaming() throws {
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.catalogEntries = [.chat("test/downloaded-model", "downloaded-model", false)];
        stubConfig.catalogEntriesAfterDownload = [.chat("test/downloaded-model", "downloaded-model", true)];
        stubConfig.downloadJobs = [
            stubDownloadJob("test/downloaded-model", "downloading", 1_000_000_000, 2_000_000_000, nil),
            nil,
        ];
        stubConfig.chatFragments = ["Downloaded"];
        try CliJourneySupport.withStubDaemon("respond-auto-download", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespond(
                prompt: "Say hello",
                modelId: "test/downloaded-model",
                noStream: false,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = respondOutcome else {
                Issue.record("an auto-download must complete before the stream: \(respondOutcome)");
                return;
            }
            #expect(stdout.text == "Downloaded");
            #expect(stderr.text.contains("1 GB / 2 GB"));
            _ = stubDaemon;
        }
    }

    @Test
    func should_reject_a_chat_request_for_an_embeddings_only_model_before_downloading() throws {
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.catalogEntries = [.embeddings("test/only-embeddings", "only-embeddings", false)];
        try CliJourneySupport.withStubDaemon("respond-capmismatch", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespond(
                prompt: "Say hello",
                modelId: "test/only-embeddings",
                noStream: false,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case let .failure(respondError) = respondOutcome,
                  case let RespondError.modelUnavailable(reason) = respondError else {
                Issue.record("a chat request against an embeddings-only model must fail on capability: \(respondOutcome)");
                return;
            }
            #expect(reason.contains("not a chat model"));
            #expect(stdout.text.isEmpty, "no answer may stream on a capability rejection");
            #expect(!stderr.text.contains("downloading"), "no download may start for a capability mismatch");
            _ = stubDaemon;
        }
    }

    @Test
    func should_send_supplied_image_bytes_to_the_daemon() throws {
        let imageDirectory: String = CliJourneySupport.freshTestDirectory("respond-with-image");
        defer { try? FileManager.default.removeItem(atPath: imageDirectory) }
        let imagePath: String = imageDirectory + "/picture.png";
        try Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]).write(to: URL(fileURLWithPath: imagePath + ".tmp"));
        try FileManager.default.moveItem(atPath: imagePath + ".tmp", toPath: imagePath);
        try Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x70, 0x61, 0x79, 0x6C, 0x6F, 0x61, 0x64, 0x2D, 0x62, 0x79, 0x74, 0x65, 0x73]).write(to: URL(fileURLWithPath: imagePath));
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        try CliJourneySupport.withResidentDefaultStubDaemon(chatFragments: ["ok"]) { (socketPath: String, stubDaemon: StubDaemon, writer: BufferedTextOutputWriter, errorWriter: BufferedTextOutputWriter) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespond(
                prompt: "Describe this",
                imagePaths: [imagePath],
                modelId: nil,
                noStream: false,
                candidateSocketPaths: [socketPath],
                stdout: writer,
                stderr: errorWriter
            );
            guard case .success = respondOutcome else {
                Issue.record("the respond journey should complete against the stub daemon: \(respondOutcome)");
                return;
            }
            let capturedImages: Array<ChatImageInput>? = stubDaemon.takeCapturedImages();
            let capturedImageList: Array<ChatImageInput> = capturedImages ?? [];
            #expect(capturedImageList.count == 1, "exactly one image should cross the boundary");
            #expect(capturedImageList.first?.mimeType == "image/png");
            _ = stdout;
            _ = stderr;
        }
    }

    @Test
    func should_send_instructions_as_the_initial_system_message() throws {
        try CliJourneySupport.withResidentDefaultStubDaemon(chatFragments: ["ok"]) { (socketPath: String, stubDaemon: StubDaemon, stdout: BufferedTextOutputWriter, stderr: BufferedTextOutputWriter) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespondWithControls(
                prompt: "Say hello",
                instructions: "Be terse",
                thinkingBudget: nil,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = respondOutcome else {
                Issue.record("the respond journey should complete against the stub daemon: \(respondOutcome)");
                return;
            }
            let capturedMessages: Array<ChatMessage> = stubDaemon.takeCapturedMessages() ?? [];
            #expect(capturedMessages.count == 2);
            guard capturedMessages.count == 2 else {
                return;
            }
            guard case let .system(systemContent) = capturedMessages[0] else {
                Issue.record("the instructions must be the initial system message");
                return;
            }
            #expect(systemContent == "Be terse");
            guard case let .user(userContent, userImages) = capturedMessages[1] else {
                Issue.record("the user prompt must follow the system message");
                return;
            }
            #expect(userContent == "Say hello");
            #expect(userImages.isEmpty);
        }
    }

    @Test
    func should_send_the_thinking_budget_in_the_generation_settings() throws {
        try CliJourneySupport.withResidentDefaultStubDaemon(chatFragments: ["ok"]) { (socketPath: String, stubDaemon: StubDaemon, stdout: BufferedTextOutputWriter, stderr: BufferedTextOutputWriter) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespondWithControls(
                prompt: "Say hello",
                instructions: nil,
                thinkingBudget: 512,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = respondOutcome else {
                Issue.record("the respond journey should complete against the stub daemon: \(respondOutcome)");
                return;
            }
            let capturedSettings: ChatGenerationSettings? = stubDaemon.takeCapturedSettings();
            #expect(capturedSettings?.thinkingBudget == 512, "the thinking budget the user supplied must reach the daemon");
            #expect(capturedSettings?.maxOutputTokens == 0, "the CLI keeps the no-opinion sentinel for the output budget");
        }
    }

    @Test
    func should_send_the_schema_file_text_to_the_daemon() throws {
        let schemaDirectory: String = CliJourneySupport.freshTestDirectory("respond-schema");
        defer { try? FileManager.default.removeItem(atPath: schemaDirectory) }
        let schemaPath: String = schemaDirectory + "/schema.json";
        try Data("{\"type\":\"object\"}".utf8).write(to: URL(fileURLWithPath: schemaPath));
        try CliJourneySupport.withResidentDefaultStubDaemon(chatFragments: ["{}"]) { (socketPath: String, stubDaemon: StubDaemon, stdout: BufferedTextOutputWriter, stderr: BufferedTextOutputWriter) in
            let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespondWithSchema(
                prompt: "Say hello",
                schemaPath: schemaPath,
                candidateSocketPaths: [socketPath],
                stdout: stdout,
                stderr: stderr
            );
            guard case .success = respondOutcome else {
                Issue.record("the respond journey should complete against the stub daemon: \(respondOutcome)");
                return;
            }
            #expect(stubDaemon.takeCapturedSchemaJson() == "{\"type\":\"object\"}");
        }
    }

    @Test
    func should_fail_before_connecting_when_the_schema_file_is_missing() throws {
        let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        // No stub daemon at all: a missing schema file must fail locally.
        let respondOutcome: Result<Void, Error> = CliJourneySupport.runRespondWithSchema(
            prompt: "Say hello",
            schemaPath: CliJourneySupport.freshTestDirectory("respond-schema-missing") + "/nope.json",
            candidateSocketPaths: [CliJourneySupport.freshTestDirectory("respond-schema-missing") + "/ipc.sock"],
            stdout: stdout,
            stderr: stderr
        );
        guard case let .failure(respondError) = respondOutcome,
              case RespondError.schemaReadFailed = respondError else {
            Issue.record("a missing schema file must fail before any daemon work: \(respondOutcome)");
            return;
        }
    }
}
