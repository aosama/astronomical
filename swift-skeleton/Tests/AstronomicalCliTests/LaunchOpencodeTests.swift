import Foundation

import Testing;

import AstronomicalCli;
import JourneyCategories;

@testable import AstronomicalCli;

/// Hermetic `astronomical launch` journeys, porting launch_opencode.rs: the
/// loopback stand-in serves `/v1/status` and `/v1/models`, the PATH fixture
/// carries a runnable `opencode` stub, and the journey inspects
/// `PreparedLaunch` instead of exec-ing the test process.
@Suite(.serialized, .tags(.hermeticJourney))
final class LaunchOpencodeTests {

    private static func chatModelsBody(_ modelIds: Array<String>) -> String {
        let advertisedModels: Array<String> = modelIds.map { (modelId: String) -> String in
            return "{\"id\":\"\(modelId)\",\"supported_endpoints\":[\"/v1/chat/completions\"],"
                + "\"context_window\":32768}";
        };
        return "{\"object\":\"list\",\"data\":[\(advertisedModels.joined(separator: ","))]}";
    }

    private static func statusBody() -> String {
        return "{\"application\":\"astronomical\",\"state\":\"ready\"}";
    }

    private func prepare(
        stubServer: StubAstronomicalServer?,
        toolArguments: Array<String>,
        isInteractive: Bool = false,
        selectionInput: String? = nil,
        pathValue: String? = nil,
        modelsBody: String? = nil,
        modelsStatusLine: String = "200 OK"
    ) -> Result<LaunchCommand.PreparedLaunch, LaunchError> {
        let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
        var candidateEndpoints: Array<(host: String, port: UInt16)> = [];
        if let stubServer: StubAstronomicalServer = stubServer {
            candidateEndpoints.append(("127.0.0.1", stubServer.port));
        }
        if let stubServer: StubAstronomicalServer = stubServer, let modelsBody: String = modelsBody {
            // The stub server answers models on the same port; rebuilding it
            // with the new body happens at the call site.
            _ = modelsBody;
            _ = modelsStatusLine;
        }
        let launchArgumentsOutcome: Result<LaunchArguments, UsageError> = {
            switch (CliJourneySupport.parse(["launch"] + toolArguments)) {
            case let .success(.launch(launchArguments)):
                return .success(launchArguments);
            case let .failure(usageError):
                return .failure(usageError);
            default:
                return .failure(.missingCommand);
            }
        }();
        guard case let .success(launchArguments) = launchArgumentsOutcome else {
            return .failure(.openCodeConfigFailed);
        }
        let launchDependencies: LaunchCommand.LaunchDependencies = LaunchCommand.LaunchDependencies(
            candidateBindEndpoints: candidateEndpoints,
            pathValue: pathValue ?? "/nonexistent-bin-dir",
            isInteractive: isInteractive,
            selectionInput: selectionInput,
            stderr: stderr,
            httpTimeoutSeconds: 2
        );
        return LaunchCommand.prepareLaunch(
            launchArguments: launchArguments,
            launchDependencies: launchDependencies
        );
    }

    @Test
    func should_fail_when_opencode_is_missing() {
        guard case let .failure(launchError) = self.prepare(
            stubServer: StubAstronomicalServer(
                statusBody: LaunchOpencodeTests.statusBody(),
                modelsBody: LaunchOpencodeTests.chatModelsBody(["m1"])
            ),
            toolArguments: []
        ) else {
            Issue.record("a missing OpenCode installation must fail launch");
            return;
        }
        #expect(launchError == .openCodeMissing);
    }

    @Test
    func should_reject_an_unsupported_tool_name() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        guard case let .failure(launchError) = self.prepare(
            stubServer: StubAstronomicalServer(
                statusBody: LaunchOpencodeTests.statusBody(),
                modelsBody: LaunchOpencodeTests.chatModelsBody(["m1"])
            ),
            toolArguments: ["vscode"],
            pathValue: pathValue
        ) else {
            Issue.record("an unsupported tool name must fail launch");
            return;
        }
        guard case let .unknownTool(requestedTool) = launchError else {
            Issue.record("the unsupported tool name must be named: \(launchError)");
            return;
        }
        #expect(requestedTool == "vscode");
    }

    @Test
    func should_fail_when_astronomical_is_not_running() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        guard case let .failure(launchError) = self.prepare(
            stubServer: nil,
            toolArguments: [],
            pathValue: pathValue
        ) else {
            Issue.record("a missing Astronomical instance must fail launch");
            return;
        }
        #expect(launchError == .astronomicalUnavailable);
    }

    @Test
    func should_use_the_only_chat_model_without_prompting() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["only-model"])
        ));
        defer { stubServer.stop() }
        guard case let .success(preparedLaunch) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            pathValue: pathValue
        ) else {
            Issue.record("one chat model should launch without prompting");
            return;
        }
        #expect(preparedLaunch.programPath.hasSuffix("opencode"));
        let configEnvironment: (String, String)? = preparedLaunch.extraEnvironment.first;
        let configContent: String = try #require(configEnvironment?.1);
        #expect(configEnvironment?.0 == "OPENCODE_CONFIG_CONTENT");
        #expect(configContent.contains("astronomical/only-model"));
        #expect(configContent.contains("baseURL\":\"http://127.0.0.1:\(stubServer.port)/v1\""));
    }

    @Test
    func should_treat_bare_launch_like_launch_opencode_when_installed() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["only-model"])
        ));
        defer { stubServer.stop() }
        #expect(self.prepare(stubServer: stubServer, toolArguments: [], pathValue: pathValue).isLaunchSuccess);
        #expect(self.prepare(stubServer: stubServer, toolArguments: ["opencode"], pathValue: pathValue).isLaunchSuccess);
    }

    @Test
    func should_not_offer_embedding_models_as_launch_targets() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let modelsBody: String = "{\"data\":[{\"id\":\"embedder\",\"supported_endpoints\":[\"/v1/embeddings\"]}]}";
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: modelsBody
        ));
        defer { stubServer.stop() }
        guard case let .failure(launchError) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            pathValue: pathValue
        ) else {
            Issue.record("an embeddings-only Library must fail launch");
            return;
        }
        #expect(launchError == .noChatModels);
    }

    @Test
    func should_require_model_flag_when_several_chat_models_exist_without_a_terminal() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["m1", "m2"])
        ));
        defer { stubServer.stop() }
        guard case let .failure(launchError) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            isInteractive: false,
            pathValue: pathValue
        ) else {
            Issue.record("several chat models without a terminal must require --model");
            return;
        }
        #expect(launchError == .modelPickerRequired);
    }

    @Test
    func should_pick_a_chat_model_from_a_tty_list() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["first-model", "second-model"])
        ));
        defer { stubServer.stop() }
        guard case let .success(preparedLaunch) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            isInteractive: true,
            selectionInput: "2\n",
            pathValue: pathValue
        ) else {
            Issue.record("a numbered picker selection should launch");
            return;
        }
        let configContent: String = try #require(preparedLaunch.extraEnvironment.first?.1);
        #expect(configContent.contains("astronomical/second-model"));
    }

    @Test
    func should_accept_a_model_id_typed_into_the_picker() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["first-model", "second-model"])
        ));
        defer { stubServer.stop() }
        guard case let .success(preparedLaunch) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            isInteractive: true,
            selectionInput: "first-model\n",
            pathValue: pathValue
        ) else {
            Issue.record("a typed model id should launch");
            return;
        }
        let configContent: String = try #require(preparedLaunch.extraEnvironment.first?.1);
        #expect(configContent.contains("astronomical/first-model"));
    }

    @Test
    func should_reject_an_invalid_picker_choice() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["m1", "m2"])
        ));
        defer { stubServer.stop() }
        guard case let .failure(launchError) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            isInteractive: true,
            selectionInput: "nope\n",
            pathValue: pathValue
        ) else {
            Issue.record("an invalid picker choice must fail launch");
            return;
        }
        #expect(launchError == .invalidModelSelection);
    }

    @Test
    func should_honor_model_flag_without_prompting() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["m1", "m2"])
        ));
        defer { stubServer.stop() }
        guard case let .success(preparedLaunch) = self.prepare(
            stubServer: stubServer,
            toolArguments: ["--model", "m2"],
            isInteractive: false,
            pathValue: pathValue
        ) else {
            Issue.record("the --model flag should skip the picker");
            return;
        }
        let configContent: String = try #require(preparedLaunch.extraEnvironment.first?.1);
        #expect(configContent.contains("astronomical/m2"));
    }

    @Test
    func should_reject_a_model_flag_that_is_not_a_library_chat_model() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["m1"])
        ));
        defer { stubServer.stop() }
        guard case let .failure(launchError) = self.prepare(
            stubServer: stubServer,
            toolArguments: ["--model", "absent"],
            pathValue: pathValue
        ) else {
            Issue.record("an absent --model id must fail launch");
            return;
        }
        guard case let .requestedModelMissing(requestedModelId) = launchError else {
            Issue.record("the absent model must be named: \(launchError)");
            return;
        }
        #expect(requestedModelId == "absent");
    }

    @Test
    func should_fail_when_the_library_has_no_chat_models() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: "{\"data\":[]}"
        ));
        defer { stubServer.stop() }
        guard case let .failure(launchError) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            pathValue: pathValue
        ) else {
            Issue.record("an empty Library must fail launch");
            return;
        }
        #expect(launchError == .noChatModels);
    }

    @Test
    func should_not_treat_a_json_listener_without_application_as_astronomical() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: "{\"object\":\"list\"}",
            modelsBody: LaunchOpencodeTests.chatModelsBody(["m1"])
        ));
        defer { stubServer.stop() }
        guard case let .failure(launchError) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            pathValue: pathValue
        ) else {
            Issue.record("a listener without the application field must not count");
            return;
        }
        #expect(launchError == .astronomicalUnavailable);
    }

    @Test
    func should_say_the_model_list_failed_when_status_is_healthy() throws {
        let (toolDirectory, pathValue): (String, String) = try LaunchPathFixture.makeOpencodePath();
        defer { try? FileManager.default.removeItem(atPath: toolDirectory) }
        let stubServer: StubAstronomicalServer = try #require(StubAstronomicalServer(
            statusBody: LaunchOpencodeTests.statusBody(),
            modelsBody: LaunchOpencodeTests.chatModelsBody(["m1"]),
            modelsStatusLine: "500 Internal Server Error"
        ));
        defer { stubServer.stop() }
        guard case let .failure(launchError) = self.prepare(
            stubServer: stubServer,
            toolArguments: [],
            pathValue: pathValue
        ) else {
            Issue.record("an unhealthy models response must fail launch");
            return;
        }
        #expect(launchError == .modelListUnavailable);
    }
}

extension Result {

    var isLaunchSuccess: Bool {
        if case .success = self {
            return true;
        }
        return false;
    }
}
