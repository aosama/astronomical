import Foundation;
import XCTest;
@testable import AstronomicalConfig;

/**
 * Acceptance journeys for loading (or first-run creating) the strict v1
 * user configuration from one instance boundary, mirroring the Rust
 * runtime_instance config-load tests.
 */
final class AstronomicalConfigLoadTests: XCTestCase {
    private static let MINIMAL_VALID_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}";
    private static let CONFIG_SCHEMA_FILE_NAME: String = "astronomical-config.schema.json";

    private var temporaryDirectoryFixture: TemporaryDirectoryFixture?;

    override func tearDown() {
        guard let fixture: TemporaryDirectoryFixture = self.temporaryDirectoryFixture else {
            super.tearDown();
            return;
        }
        do {
            try fixture.destroy();
        } catch {
            XCTFail("temporary directory should be removed: \(error)");
        }
        self.temporaryDirectoryFixture = nil;
        super.tearDown();
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

    func testShouldGenerateTheDevelopmentFirstRunConfigWithTheDevelopmentPort() throws -> Void {
        let fictionalHomeDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );

        let developmentConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);

        let supervisorBindAddress: SocketEndpoint = try developmentConfig.supervisorBindAddress();
        XCTAssertEqual(supervisorBindAddress.description, "127.0.0.1:6733");
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: developmentPaths.configFilePath.string),
            "first run should materialize config.json on disk"
        );
        let adjacentSchemaPath: FilePath = developmentPaths.stateDirectory.appending(
            component: AstronomicalConfigLoadTests.CONFIG_SCHEMA_FILE_NAME
        );
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: adjacentSchemaPath.string),
            "first run should write the validation schema beside config.json"
        );
    }

    func testShouldLoadASuppliedTestHomeOnlyFromTheDevelopmentChannel() throws -> Void {
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
        XCTAssertEqual(supervisorBindAddress.description, "127.0.0.1:6733");
        let stableSentinelBytes: Data = try self.readFileBytes(
            atPath: stableStateDirectory.appending(component: "config.json")
        );
        XCTAssertEqual(
            stableSentinelBytes,
            Data("not valid JSON".utf8),
            "the Stable channel sentinel must never be read or rewritten by a Development load"
        );
    }

    func testShouldAllowAnExplicitTestStateDirectoryToSelectAnEphemeralEndpoint() throws -> Void {
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
        XCTAssertEqual(supervisorBindAddress.description, "127.0.0.1:0");
    }

    func testShouldReloadItsOwnFirstRunConfigFile() throws -> Void {
        let fictionalHomeDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fictionalHomeDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        _ = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);

        let reloadedConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(developmentPaths);

        let supervisorBindAddress: SocketEndpoint = try reloadedConfig.supervisorBindAddress();
        XCTAssertEqual(supervisorBindAddress.description, "127.0.0.1:6733");
    }

    func testShouldRejectInvalidJSONWhenLoadingTheConfigFile() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "not valid JSON",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        XCTAssertThrowsError(
            try AstronomicalConfig.loadFromInstancePaths(developmentPaths)
        ) { (thrownError: Error) in
            guard case AstronomicalConfigError.parseConfigFile = thrownError else {
                XCTFail("expected parseConfigFile, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldRejectADuplicateConfigKeyWhenLoadingTheConfigFile() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        XCTAssertThrowsError(
            try AstronomicalConfig.loadFromInstancePaths(developmentPaths)
        ) { (thrownError: Error) in
            guard case ConfigResolutionError.duplicateConfigKey(_, "schema_version") = thrownError else {
                XCTFail("expected duplicateConfigKey for schema_version, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldRejectAnUnknownTopLevelConfigField() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"unexpected_field\":true,\"runtime\":{\"model_directories\":[]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        XCTAssertThrowsError(
            try AstronomicalConfig.loadFromInstancePaths(developmentPaths)
        ) { (thrownError: Error) in
            guard case AstronomicalConfigError.parseConfigFile = thrownError else {
                XCTFail("expected parseConfigFile, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldRejectARelativeModelDirectory() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[\"relative-models\"]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        XCTAssertThrowsError(
            try AstronomicalConfig.loadFromInstancePaths(developmentPaths)
        ) { (thrownError: Error) in
            guard case let AstronomicalConfigError.pathMustBeAbsolute(
                fieldName,
                _
            ) = thrownError, fieldName == "runtime.model_directories" else {
                XCTFail("expected pathMustBeAbsolute for runtime.model_directories, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldRejectAnUnsupportedSchemaVersion() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":2,\"runtime\":{\"model_directories\":[]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        XCTAssertThrowsError(
            try AstronomicalConfig.loadFromInstancePaths(developmentPaths)
        ) { (thrownError: Error) in
            guard case AstronomicalConfigError.unsupportedSchemaVersion = thrownError else {
                XCTFail("expected unsupportedSchemaVersion, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldRejectAnIncorrectSchemaReference() throws -> Void {
        let testStateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            testStateDirectoryFixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(
            contents: "{\"$schema\":\"./other.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}",
            beneathStateDirectory: developmentPaths.stateDirectory
        );

        XCTAssertThrowsError(
            try AstronomicalConfig.loadFromInstancePaths(developmentPaths)
        ) { (thrownError: Error) in
            guard case AstronomicalConfigError.invalidSchemaReference = thrownError else {
                XCTFail("expected invalidSchemaReference, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldRejectAnOversizedConfigFile() throws -> Void {
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

        XCTAssertThrowsError(
            try AstronomicalConfig.loadFromInstancePaths(developmentPaths)
        ) { (thrownError: Error) in
            guard case AstronomicalConfigError.configFileTooLarge = thrownError else {
                XCTFail("expected configFileTooLarge, got \(thrownError)");
                return;
            }
        }
    }
}
