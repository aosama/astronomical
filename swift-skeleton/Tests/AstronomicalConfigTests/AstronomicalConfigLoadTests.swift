import Foundation;

import Testing;

import JourneyCategories;

@testable import AstronomicalConfig;

/**
 * Acceptance journeys for loading (or first-run creating) the strict v1
 * user configuration from one instance boundary, mirroring the Rust
 * runtime_instance config-load tests.
 */
@Suite(.tags(.hermeticJourney))
final class AstronomicalConfigLoadTests {
    private static let MINIMAL_VALID_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}";
    private static let CONFIG_SCHEMA_FILE_NAME: String = "astronomical-config.schema.json";

    private var temporaryDirectoryFixture: TemporaryDirectoryFixture?;

    deinit {
        if let fixture: TemporaryDirectoryFixture = self.temporaryDirectoryFixture {
            try? fixture.destroy();
        }
    }

    private func makeTemporaryDirectoryFixture() throws -> TemporaryDirectoryFixture {
        let fixture: TemporaryDirectoryFixture = try TemporaryDirectoryFixture();
        self.temporaryDirectoryFixture = fixture;
        return fixture;
    }

    private func writeConfigFile(contents: String, beneathStateDirectory stateDirectory: FilePath) throws -> Void {
        try FileManager.default.createDirectory(
            atPath: stateDirectory.string,
            withIntermediateDirectories: true
        );
        let configFileUrl: URL = URL(fileURLWithPath: stateDirectory.appending(component: "config.json").string);
        try contents.write(to: configFileUrl, atomically: true, encoding: String.Encoding.utf8);
    }

    private func readFileBytes(atPath path: FilePath) throws -> Data {
        return try Data(contentsOf: URL(fileURLWithPath: path.string));
    }

    @Test
    func should_generate_the_development_first_run_config_with_the_development_port() throws -> Void {
        let fictionalHomeDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        let developmentConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);

        let supervisorBindAddress: SocketEndpoint = try developmentConfig.supervisorBindAddress();
        #expect(supervisorBindAddress.description == "127.0.0.1:6733");
        #expect(
            FileManager.default.fileExists(atPath: developmentPaths.configFilePath.string),
            "first run should materialize config.json on disk"
        );
        let adjacentSchemaPath: FilePath = developmentPaths.stateDirectory.appending(
            component: AstronomicalConfigLoadTests.CONFIG_SCHEMA_FILE_NAME
        );
        #expect(
            FileManager.default.fileExists(atPath: adjacentSchemaPath.string),
            "first run should write the validation schema beside config.json"
        );
    }

    @Test
    func should_load_a_supplied_test_home_only_from_the_development_channel() throws -> Void {
        let testHomeDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stableStateDirectory: FilePath = testHomeDirectoryFixture.rootDirectoryPath.appending(component: ".astronomical");
        let developmentStateDirectory: FilePath = testHomeDirectoryFixture.rootDirectoryPath.appending(component: ".astronomical-dev");
        try self.writeConfigFile(contents: "not valid JSON", beneathStateDirectory: stableStateDirectory);
        try self.writeConfigFile(
            contents: AstronomicalConfigLoadTests.MINIMAL_VALID_CONFIG_JSON,
            beneathStateDirectory: developmentStateDirectory
        );

        let developmentConfig: AstronomicalConfig = try AstronomicalConfig.loadFromDevelopmentHomeDirectory(
            testHomeDirectoryFixture.rootDirectoryPath
        );

        let supervisorBindAddress: SocketEndpoint = try developmentConfig.supervisorBindAddress();
        #expect(supervisorBindAddress.description == "127.0.0.1:6733");
        let stableSentinelBytes: Data = try self.readFileBytes(
            atPath: stableStateDirectory.appending(component: "config.json")
        );
        #expect(
            stableSentinelBytes == Data("not valid JSON".utf8),
            "the Stable channel sentinel must never be read or rewritten by a Development load"
        );
    }

    @Test
    func should_let_an_explicit_test_state_directory_select_an_ephemeral_endpoint() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let explicitPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            defaultBindAddress: SocketEndpoint.loopback(port: 0)
        );
        try self.writeConfigFile(
            contents: AstronomicalConfigLoadTests.MINIMAL_VALID_CONFIG_JSON,
            beneathStateDirectory: explicitPaths.stateDirectory
        );

        let explicitConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(explicitPaths);

        let supervisorBindAddress: SocketEndpoint = try explicitConfig.supervisorBindAddress();
        #expect(supervisorBindAddress.description == "127.0.0.1:0");
    }

    @Test
    func should_reload_its_own_first_run_config_file() throws -> Void {
        let fictionalHomeDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);

        let reloadedConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);

        let supervisorBindAddress: SocketEndpoint = try reloadedConfig.supervisorBindAddress();
        #expect(supervisorBindAddress.description == "127.0.0.1:6733");
    }

    @Test
    func should_reject_invalid_json_when_loading_the_config_file() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "not valid JSON",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        do {
            _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
            Issue.record("expected parseConfigFile");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.parseConfigFile = configError else {
                Issue.record(Comment(stringLiteral: "expected parseConfigFile, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_reject_a_duplicate_config_key_when_loading_the_config_file() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        do {
            _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
            Issue.record("expected duplicateConfigKey for schema_version");
        } catch let resolutionError as ConfigResolutionError {
            guard case ConfigResolutionError.duplicateConfigKey(_, "schema_version") = resolutionError else {
                Issue.record(Comment(stringLiteral: "expected duplicateConfigKey for schema_version, got \(resolutionError)"));
                return;
            }
        }
    }

    @Test
    func should_reject_an_unknown_top_level_config_field() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"unexpected_field\":true,\"runtime\":{\"model_directories\":[]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        do {
            _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
            Issue.record("expected parseConfigFile");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.parseConfigFile = configError else {
                Issue.record(Comment(stringLiteral: "expected parseConfigFile, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_reject_a_relative_model_directory() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[\"relative-models\"]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        do {
            _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
            Issue.record("expected pathMustBeAbsolute for runtime.model_directories");
        } catch let configError as AstronomicalConfigError {
            guard case let .pathMustBeAbsolute(fieldName, _) = configError, fieldName == "runtime.model_directories" else {
                Issue.record(Comment(stringLiteral: "expected pathMustBeAbsolute for runtime.model_directories, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_reject_an_unsupported_schema_version() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":2,\"runtime\":{\"model_directories\":[]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        do {
            _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
            Issue.record("expected unsupportedSchemaVersion");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.unsupportedSchemaVersion = configError else {
                Issue.record(Comment(stringLiteral: "expected unsupportedSchemaVersion, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_reject_an_incorrect_schema_reference() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./other.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        do {
            _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
            Issue.record("expected invalidSchemaReference");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.invalidSchemaReference = configError else {
                Issue.record(Comment(stringLiteral: "expected invalidSchemaReference, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_reject_an_oversized_config_file() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        let oversizedConfigPadding: String = String(repeating: "a", count: 1_048_570);
        try self.writeConfigFile(
            contents: "{\"padding\":\"\(oversizedConfigPadding)\"}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        do {
            _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
            Issue.record("expected configFileTooLarge");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.configFileTooLarge = configError else {
                Issue.record(Comment(stringLiteral: "expected configFileTooLarge, got \(configError)"));
                return;
            }
        }
    }
}
