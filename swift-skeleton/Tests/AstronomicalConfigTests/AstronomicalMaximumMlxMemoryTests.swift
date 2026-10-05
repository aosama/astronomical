import Foundation;
import XCTest;
@testable import AstronomicalConfig;

/**
 * Acceptance journeys for the maximum_mlx_memory_gb setting transaction,
 * porting the intent of crates/config/src/maximum_mlx_memory.rs: a validated
 * prepare step and an all-or-nothing commit that refuses to clobber a config
 * changed underneath it.
 */
final class AstronomicalMaximumMlxMemoryTests: XCTestCase {
    private static let MINIMAL_VALID_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}";
    private static let CONFIG_FILE_NAME: String = "config.json";
    private static let LEGACY_BACKUP_FILE_NAME: String = "config.legacy-v0.json";
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

    private func writeFileBytes(_ contents: Data, toPath path: FilePath) throws -> Void {
        try FileManager.default.createDirectory(
            atPath: path.parentDirectory()?.string ?? path.string,
            withIntermediateDirectories: true
        );
        try contents.write(to: URL(fileURLWithPath: path.string), options: Data.WritingOptions.atomic);
    }

    private func writeFile(_ contents: String, toPath path: FilePath) throws -> Void {
        return try self.writeFileBytes(Data(contents.utf8), toPath: path);
    }

    private func readFileBytes(atPath path: FilePath) throws -> Data {
        return try Data(contentsOf: URL(fileURLWithPath: path.string));
    }

    private func readConfigJsonDictionary(stateDirectory: FilePath) throws -> Dictionary<String, Any> {
        let configFilePath: FilePath = stateDirectory.appending(component: AstronomicalMaximumMlxMemoryTests.CONFIG_FILE_NAME);
        let configBytes: Data = try self.readFileBytes(atPath: configFilePath);
        let parsedConfigValue: Any = try JSONSerialization.jsonObject(with: configBytes, options: []);
        guard let configJson: Dictionary<String, Any> = parsedConfigValue as? Dictionary<String, Any> else {
            XCTFail("config.json should hold a JSON object");
            return Dictionary<String, Any>();
        }
        return configJson;
    }

    private func readRuntimeJsonDictionary(stateDirectory: FilePath) throws -> Dictionary<String, Any> {
        let configJson: Dictionary<String, Any> = try self.readConfigJsonDictionary(stateDirectory: stateDirectory);
        guard let runtimeJson: Dictionary<String, Any> = configJson["runtime"] as? Dictionary<String, Any> else {
            XCTFail("config.json should hold a runtime object");
            return Dictionary<String, Any>();
        }
        return runtimeJson;
    }

    func testShouldConvertPositiveDecimalGigabytesToExactDecimalBytes() throws -> Void {
        let oneGigabyteBytes: UInt64 = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(1);
        let twelveGigabyteBytes: UInt64 = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(12);
        XCTAssertEqual(oneGigabyteBytes, 1_000_000_000, "end-user memory values are decimal SI gigabytes");
        XCTAssertEqual(twelveGigabyteBytes, 12_000_000_000);
    }

    func testShouldRejectAZeroGigabyteSetting() throws -> Void {
        XCTAssertThrowsError(try MaximumMlxMemory.maximumMlxMemoryGbToBytes(0)) { (thrownError: Error) in
            guard case AstronomicalConfigError.invalidMaximumMlxMemoryGb = thrownError else {
                XCTFail("expected invalidMaximumMlxMemoryGb, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldRejectAGigabyteSettingBeyondTheUnsignedByteRange() throws -> Void {
        // 18_446_744_073 gigabytes still fits in 64 bits; one more gigabyte
        // overflows UInt64.max = 18_446_744_073_709_551_615 bytes.
        let largestFittingGigabytes: UInt64 = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(18_446_744_073);
        XCTAssertEqual(largestFittingGigabytes, 18_446_744_073_000_000_000);
        XCTAssertThrowsError(try MaximumMlxMemory.maximumMlxMemoryGbToBytes(18_446_744_074)) { (thrownError: Error) in
            guard case AstronomicalConfigError.invalidMaximumMlxMemoryGb = thrownError else {
                XCTFail("expected invalidMaximumMlxMemoryGb, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldPersistTheOverrideIntoAFirstRunConfigFile() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;

        let configUpdate: MaximumMlxMemoryConfigUpdate = try MaximumMlxMemory.writeMaximumMlxMemoryGb(
            stateDirectory: stateDirectory,
            maximumMlxMemoryGb: 24
        );
        XCTAssertEqual(configUpdate.priorConfigBytes, nil, "first run has no prior config document");

        let runtimeJson: Dictionary<String, Any> = try self.readRuntimeJsonDictionary(stateDirectory: stateDirectory);
        XCTAssertEqual(runtimeJson["maximum_mlx_memory_gb"] as? UInt64, 24);
        let adjacentSchemaPath: FilePath = stateDirectory.appending(
            component: AstronomicalMaximumMlxMemoryTests.CONFIG_SCHEMA_FILE_NAME
        );
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: adjacentSchemaPath.string),
            "the schema should be written beside the first-run config document"
        );
    }

    func testShouldReplaceAndClearTheOverrideOnAnExistingV1Config() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;
        try self.writeFile(
            AstronomicalMaximumMlxMemoryTests.MINIMAL_VALID_CONFIG_JSON,
            toPath: stateDirectory.appending(component: AstronomicalMaximumMlxMemoryTests.CONFIG_FILE_NAME)
        );

        let _ = try MaximumMlxMemory.writeMaximumMlxMemoryGb(stateDirectory: stateDirectory, maximumMlxMemoryGb: 8);
        let replacedRuntimeJson: Dictionary<String, Any> = try self.readRuntimeJsonDictionary(stateDirectory: stateDirectory);
        XCTAssertEqual(replacedRuntimeJson["maximum_mlx_memory_gb"] as? UInt64, 8);
        XCTAssertEqual(replacedRuntimeJson["model_directories"] as? Array<String>, Array<String>());

        let _ = try MaximumMlxMemory.writeMaximumMlxMemoryGb(stateDirectory: stateDirectory, maximumMlxMemoryGb: nil);
        let clearedRuntimeJson: Dictionary<String, Any> = try self.readRuntimeJsonDictionary(stateDirectory: stateDirectory);
        XCTAssertFalse(
            clearedRuntimeJson.keys.contains("maximum_mlx_memory_gb"),
            "clearing the override should omit the key rather than persist a null"
        );
        XCTAssertEqual(clearedRuntimeJson["model_directories"] as? Array<String>, Array<String>());
    }

    func testShouldMigrateALegacyConfigAndKeepTheLegacyBackup() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;
        let configFilePath: FilePath = stateDirectory.appending(component: AstronomicalMaximumMlxMemoryTests.CONFIG_FILE_NAME);
        try self.writeFile("{}", toPath: configFilePath);

        let _ = try MaximumMlxMemory.writeMaximumMlxMemoryGb(stateDirectory: stateDirectory, maximumMlxMemoryGb: 16);

        let configJson: Dictionary<String, Any> = try self.readConfigJsonDictionary(stateDirectory: stateDirectory);
        XCTAssertEqual(configJson["schema_version"] as? Int, 1, "the legacy document should migrate to schema v1");
        let runtimeJson: Dictionary<String, Any> = try self.readRuntimeJsonDictionary(stateDirectory: stateDirectory);
        XCTAssertEqual(runtimeJson["maximum_mlx_memory_gb"] as? UInt64, 16);
        let legacyBackupPath: FilePath = stateDirectory.appending(
            component: AstronomicalMaximumMlxMemoryTests.LEGACY_BACKUP_FILE_NAME
        );
        let legacyBackupBytes: Data = try self.readFileBytes(atPath: legacyBackupPath);
        XCTAssertEqual(legacyBackupBytes, Data("{}".utf8), "the backup should hold the exact legacy document");
    }

    func testShouldRefuseToCommitWhenTheConfigChangedDuringPrepare() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;
        let configFilePath: FilePath = stateDirectory.appending(component: AstronomicalMaximumMlxMemoryTests.CONFIG_FILE_NAME);
        try self.writeFile(
            AstronomicalMaximumMlxMemoryTests.MINIMAL_VALID_CONFIG_JSON,
            toPath: configFilePath
        );

        let configUpdate: MaximumMlxMemoryConfigUpdate = try MaximumMlxMemory.prepareMaximumMlxMemoryGbUpdate(
            stateDirectory: stateDirectory,
            maximumMlxMemoryGb: 24
        );
        try self.writeFile(
            "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[\"/models\"]}}",
            toPath: configFilePath
        );

        XCTAssertThrowsError(
            try MaximumMlxMemory.commitMaximumMlxMemoryGbUpdate(stateDirectory: stateDirectory, configUpdate: configUpdate)
        ) { (thrownError: Error) in
            guard case AstronomicalConfigError.configChangedDuringUpdate = thrownError else {
                XCTFail("expected configChangedDuringUpdate, got \(thrownError)");
                return;
            }
        }
    }

    func testShouldLeaveTheConfigUntouchedWhenTheGigabyteSettingIsInvalid() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;
        let configFilePath: FilePath = stateDirectory.appending(component: AstronomicalMaximumMlxMemoryTests.CONFIG_FILE_NAME);
        try self.writeFile(
            AstronomicalMaximumMlxMemoryTests.MINIMAL_VALID_CONFIG_JSON,
            toPath: configFilePath
        );
        let originalConfigBytes: Data = try self.readFileBytes(atPath: configFilePath);

        XCTAssertThrowsError(
            try MaximumMlxMemory.writeMaximumMlxMemoryGb(stateDirectory: stateDirectory, maximumMlxMemoryGb: 0)
        ) { (thrownError: Error) in
            guard case AstronomicalConfigError.invalidMaximumMlxMemoryGb = thrownError else {
                XCTFail("expected invalidMaximumMlxMemoryGb, got \(thrownError)");
                return;
            }
        }
        XCTAssertEqual(
            try self.readFileBytes(atPath: configFilePath),
            originalConfigBytes,
            "a rejected setting must not touch the source of truth"
        );
    }
}
