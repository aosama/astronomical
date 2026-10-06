import Foundation;

import Testing;

import JourneyCategories;

@testable import AstronomicalConfig;

/**
 * Acceptance journeys for the maximum_mlx_memory_gb setting transaction,
 * porting the intent of crates/config/src/maximum_mlx_memory.rs: a validated
 * prepare step and an all-or-nothing commit that refuses to clobber a config
 * changed underneath it.
 */
@Suite(.tags(.hermeticJourney))
final class AstronomicalMaximumMlxMemoryTests {
    private static let MINIMAL_VALID_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}";
    private static let CONFIG_FILE_NAME: String = "config.json";
    private static let LEGACY_BACKUP_FILE_NAME: String = "config.legacy-v0.json";
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
            Issue.record("config.json should hold a JSON object");
            return Dictionary<String, Any>();
        }
        return configJson;
    }

    private func readRuntimeJsonDictionary(stateDirectory: FilePath) throws -> Dictionary<String, Any> {
        let configJson: Dictionary<String, Any> = try self.readConfigJsonDictionary(stateDirectory: stateDirectory);
        guard let runtimeJson: Dictionary<String, Any> = configJson["runtime"] as? Dictionary<String, Any> else {
            Issue.record("config.json should hold a runtime object");
            return Dictionary<String, Any>();
        }
        return runtimeJson;
    }

    @Test
    func should_convert_positive_decimal_gigabytes_to_exact_decimal_bytes() throws -> Void {
        let oneGigabyteBytes: UInt64 = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(1);
        let twelveGigabyteBytes: UInt64 = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(12);
        #expect(oneGigabyteBytes == 1_000_000_000, "end-user memory values are decimal SI gigabytes");
        #expect(twelveGigabyteBytes == 12_000_000_000);
    }

    @Test
    func should_reject_a_zero_gigabyte_setting() throws -> Void {
        do {
            _ = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(0);
            Issue.record("expected invalidMaximumMlxMemoryGb");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.invalidMaximumMlxMemoryGb = configError else {
                Issue.record(Comment(stringLiteral: "expected invalidMaximumMlxMemoryGb, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_reject_a_gigabyte_setting_beyond_the_unsigned_byte_range() throws -> Void {
        // 18_446_744_073 gigabytes still fits in 64 bits; one more gigabyte
        // overflows UInt64.max = 18_446_744_073_709_551_615 bytes.
        let largestFittingGigabytes: UInt64 = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(18_446_744_073);
        #expect(largestFittingGigabytes == 18_446_744_073_000_000_000);
        do {
            _ = try MaximumMlxMemory.maximumMlxMemoryGbToBytes(18_446_744_074);
            Issue.record("expected invalidMaximumMlxMemoryGb");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.invalidMaximumMlxMemoryGb = configError else {
                Issue.record(Comment(stringLiteral: "expected invalidMaximumMlxMemoryGb, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_persist_the_override_into_a_first_run_config_file() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;

        let configUpdate: MaximumMlxMemoryConfigUpdate = try MaximumMlxMemory.writeMaximumMlxMemoryGb(
            stateDirectory: stateDirectory,
            maximumMlxMemoryGb: 24
        );
        #expect(configUpdate.priorConfigBytes == nil, "first run has no prior config document");

        let runtimeJson: Dictionary<String, Any> = try self.readRuntimeJsonDictionary(stateDirectory: stateDirectory);
        #expect(runtimeJson["maximum_mlx_memory_gb"] as? UInt64 == 24);
        let adjacentSchemaPath: FilePath = stateDirectory.appending(
            component: AstronomicalMaximumMlxMemoryTests.CONFIG_SCHEMA_FILE_NAME
        );
        #expect(
            FileManager.default.fileExists(atPath: adjacentSchemaPath.string),
            "the schema should be written beside the first-run config document"
        );
    }

    @Test
    func should_replace_and_clear_the_override_on_an_existing_v1_config() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;
        try self.writeFile(
            AstronomicalMaximumMlxMemoryTests.MINIMAL_VALID_CONFIG_JSON,
            toPath: stateDirectory.appending(component: AstronomicalMaximumMlxMemoryTests.CONFIG_FILE_NAME)
        );

        _ = try MaximumMlxMemory.writeMaximumMlxMemoryGb(stateDirectory: stateDirectory, maximumMlxMemoryGb: 8);
        let replacedRuntimeJson: Dictionary<String, Any> = try self.readRuntimeJsonDictionary(stateDirectory: stateDirectory);
        #expect(replacedRuntimeJson["maximum_mlx_memory_gb"] as? UInt64 == 8);
        #expect(replacedRuntimeJson["model_directories"] as? Array<String> == Array<String>());

        _ = try MaximumMlxMemory.writeMaximumMlxMemoryGb(stateDirectory: stateDirectory, maximumMlxMemoryGb: nil);
        let clearedRuntimeJson: Dictionary<String, Any> = try self.readRuntimeJsonDictionary(stateDirectory: stateDirectory);
        #expect(
            !clearedRuntimeJson.keys.contains("maximum_mlx_memory_gb"),
            "clearing the override should omit the key rather than persist a null"
        );
        #expect(clearedRuntimeJson["model_directories"] as? Array<String> == Array<String>());
    }

    @Test
    func should_migrate_a_legacy_config_and_keep_the_legacy_backup() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;
        let configFilePath: FilePath = stateDirectory.appending(component: AstronomicalMaximumMlxMemoryTests.CONFIG_FILE_NAME);
        try self.writeFile("{}", toPath: configFilePath);

        _ = try MaximumMlxMemory.writeMaximumMlxMemoryGb(stateDirectory: stateDirectory, maximumMlxMemoryGb: 16);

        let configJson: Dictionary<String, Any> = try self.readConfigJsonDictionary(stateDirectory: stateDirectory);
        #expect(configJson["schema_version"] as? Int == 1, "the legacy document should migrate to schema v1");
        let runtimeJson: Dictionary<String, Any> = try self.readRuntimeJsonDictionary(stateDirectory: stateDirectory);
        #expect(runtimeJson["maximum_mlx_memory_gb"] as? UInt64 == 16);
        let legacyBackupPath: FilePath = stateDirectory.appending(
            component: AstronomicalMaximumMlxMemoryTests.LEGACY_BACKUP_FILE_NAME
        );
        let legacyBackupBytes: Data = try self.readFileBytes(atPath: legacyBackupPath);
        #expect(legacyBackupBytes == Data("{}".utf8), "the backup should hold the exact legacy document");
    }

    @Test
    func should_refuse_to_commit_when_the_config_changed_during_prepare() throws -> Void {
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

        do {
            _ = try MaximumMlxMemory.commitMaximumMlxMemoryGbUpdate(stateDirectory: stateDirectory, configUpdate: configUpdate);
            Issue.record("expected configChangedDuringUpdate");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.configChangedDuringUpdate = configError else {
                Issue.record(Comment(stringLiteral: "expected configChangedDuringUpdate, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_leave_the_config_untouched_when_the_gigabyte_setting_is_invalid() throws -> Void {
        let stateDirectoryFixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let stateDirectory: FilePath = stateDirectoryFixture.rootDirectoryPath;
        let configFilePath: FilePath = stateDirectory.appending(component: AstronomicalMaximumMlxMemoryTests.CONFIG_FILE_NAME);
        try self.writeFile(
            AstronomicalMaximumMlxMemoryTests.MINIMAL_VALID_CONFIG_JSON,
            toPath: configFilePath
        );
        let originalConfigBytes: Data = try self.readFileBytes(atPath: configFilePath);

        do {
            _ = try MaximumMlxMemory.writeMaximumMlxMemoryGb(stateDirectory: stateDirectory, maximumMlxMemoryGb: 0);
            Issue.record("expected invalidMaximumMlxMemoryGb");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.invalidMaximumMlxMemoryGb = configError else {
                Issue.record(Comment(stringLiteral: "expected invalidMaximumMlxMemoryGb, got \(configError)"));
                return;
            }
        }
        #expect(
            try self.readFileBytes(atPath: configFilePath) == originalConfigBytes,
            "a rejected setting must not touch the source of truth"
        );
    }
}
