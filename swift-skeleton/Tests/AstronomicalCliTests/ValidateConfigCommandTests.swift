import Foundation

import Testing;

import AstronomicalCli;
import AstronomicalConfig;
import JourneyCategories;

@testable import AstronomicalCli;

/// Hermetic `astronomical validate config` journeys, porting
/// validate_config_command.rs. No daemon, no GPU, no network.
@Suite(.tags(.hermeticJourney))
final class ValidateConfigCommandTests {

    private static let minimalValidConfigJson: String = """
    {
      "$schema": "./astronomical-config.schema.json",
      "schema_version": 1,
      "runtime": { "model_directories": ["/absolute/model/library"] }
    }
    """;

    private static func parsedValidateConfig(
        _ arguments: Array<String>
    ) throws -> ValidateConfigArguments {
        switch (CliJourneySupport.parse(["validate", "config"] + arguments)) {
        case let .success(.validateConfig(validateArguments)):
            return validateArguments;
        case let .failure(usageError):
            throw usageError;
        default:
            throw UsageError.unknownValidateTarget("internal");
        }
    }

    private static func hermeticInstanceState() throws -> (stateDirectory: String, instancePaths: AstronomicalInstancePaths) {
        let stateDirectory: String = CliJourneySupport.freshTestDirectory("validate-config") + "/state";
        try FileManager.default.createDirectory(atPath: stateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: stateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        return (stateDirectory, instancePaths);
    }

    private static func writeConfigFile(
        _ instancePaths: AstronomicalInstancePaths,
        contents: String
    ) throws -> String {
        let configFilePath: String = instancePaths.configFilePath.string;
        try FileManager.default.createDirectory(
            atPath: (configFilePath as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true
        );
        try Data(contents.utf8).write(to: URL(fileURLWithPath: configFilePath));
        return configFilePath;
    }

    private static func renderedReport(
        _ instancePaths: AstronomicalInstancePaths,
        renderJson: Bool = false
    ) -> Result<String, Error> {
        let renderedOutput: BufferedTextOutputWriter = BufferedTextOutputWriter();
        return Result {
            try ValidateConfigCommand.run(
                validateArguments: ValidateConfigArguments(
                    runtimeInstance: .development,
                    renderJson: renderJson
                ),
                instancePaths: instancePaths,
                renderedOutput: renderedOutput
            );
            return renderedOutput.text;
        };
    }

    @Test
    func should_default_validate_config_to_development_instance() throws {
        let validateArguments: ValidateConfigArguments = try ValidateConfigCommandTests.parsedValidateConfig([]);
        #expect(validateArguments.runtimeInstance == .development);
        #expect(!validateArguments.renderJson);
    }

    @Test
    func should_parse_stable_instance_flag() throws {
        let validateArguments: ValidateConfigArguments = try ValidateConfigCommandTests.parsedValidateConfig([
            "--instance", "stable",
        ]);
        #expect(validateArguments.runtimeInstance == .stable);
    }

    @Test
    func should_reject_unknown_instance_name() {
        guard case let .failure(usageError) = CliJourneySupport.parse([
            "validate", "config", "--instance", "beta",
        ]) else {
            Issue.record("an unknown instance name must be a usage error");
            return;
        }
        #expect(usageError == .unknownInstance("beta"));
    }

    @Test
    func should_reject_validate_without_config_noun() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["validate", "settings"]) else {
            Issue.record("validate without the config noun must be a usage error");
            return;
        }
        #expect(usageError == .unknownValidateTarget("settings"));
    }

    @Test
    func should_report_effective_values_for_valid_config() throws {
        let (stateDirectory, instancePaths): (String, AstronomicalInstancePaths) = try ValidateConfigCommandTests.hermeticInstanceState();
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        _ = try ValidateConfigCommandTests.writeConfigFile(instancePaths, contents: ValidateConfigCommandTests.minimalValidConfigJson);
        let reportOutcome: Result<String, Error> = ValidateConfigCommandTests.renderedReport(instancePaths);
        guard case let .success(reportText) = reportOutcome else {
            Issue.record("a valid config should render its report: \(reportOutcome)");
            return;
        }
        #expect(reportText.contains("Configuration file: "));
        #expect(reportText.contains("Runtime instance: development"));
        #expect(reportText.contains("Model directories: 1"));
        #expect(reportText.contains("/absolute/model/library"));
        #expect(reportText.contains("Maximum MLX memory: not set"));
        #expect(reportText.contains("Persistent prompt cache: enabled"));
        #expect(reportText.contains("Performance attribution: "));
    }

    @Test
    func should_fail_when_config_file_is_missing() throws {
        let (stateDirectory, instancePaths): (String, AstronomicalInstancePaths) = try ValidateConfigCommandTests.hermeticInstanceState();
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let reportOutcome: Result<String, Error> = ValidateConfigCommandTests.renderedReport(instancePaths);
        guard case let .failure(validateError) = reportOutcome,
              case ValidateConfigError.configFileMissing = validateError else {
            Issue.record("a missing config file must fail with the recovery message: \(reportOutcome)");
            return;
        }
        #expect(String(describing: validateError).contains("Start Astronomical once to create one"));
    }

    @Test
    func should_fail_when_config_document_is_invalid() throws {
        let (stateDirectory, instancePaths): (String, AstronomicalInstancePaths) = try ValidateConfigCommandTests.hermeticInstanceState();
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        _ = try ValidateConfigCommandTests.writeConfigFile(
            instancePaths,
            contents: "{\"schema_version\": 1, \"runtime\": {\"model_directories\": \"not-a-list\"}}"
        );
        let reportOutcome: Result<String, Error> = ValidateConfigCommandTests.renderedReport(instancePaths);
        guard case let .failure(validateError) = reportOutcome,
              case ValidateConfigError.invalidConfiguration = validateError else {
            Issue.record("an invalid config document must fail with the invalid-configuration error: \(reportOutcome)");
            return;
        }
    }

    @Test
    func should_render_json_report_for_scripts() throws {
        let (stateDirectory, instancePaths): (String, AstronomicalInstancePaths) = try ValidateConfigCommandTests.hermeticInstanceState();
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        _ = try ValidateConfigCommandTests.writeConfigFile(instancePaths, contents: ValidateConfigCommandTests.minimalValidConfigJson);
        let reportOutcome: Result<String, Error> = ValidateConfigCommandTests.renderedReport(
            instancePaths,
            renderJson: true
        );
        guard case let .success(reportText) = reportOutcome else {
            Issue.record("a valid config should render its JSON report: \(reportOutcome)");
            return;
        }
        let reportDocument: Dictionary<String, Any> = try #require(
            JSONSerialization.jsonObject(with: Data(reportText.utf8)) as? Dictionary<String, Any>
        );
        #expect(reportDocument["runtime_instance"] as? String == "development");
        #expect((reportDocument["model_directories"] as? Array<String>)?.count == 1);
        #expect(reportDocument["maximum_mlx_memory_bytes"] is NSNull);
    }
}
