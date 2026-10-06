import Foundation;
import XCTest;
@testable import AstronomicalConfig;

/**
 * Acceptance journeys for the resolved runtime accessors the worker-startup
 * bootstrap needs: prompt-cache policy, bounded logging, the optional MLX
 * ceiling, attribution toggles, and the configuration digest, mirroring the
 * precedence of crates/config/src/lib.rs.
 */
final class RuntimeAccessorsTests: XCTestCase {

    private static let MINIMAL_VALID_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}";

    private static let OVERRIDDEN_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
        + "\"runtime\":{\"model_directories\":[],\"maximum_mlx_memory_gb\":32,"
        + "\"experimental_qwen_thinking_channel_seed_enabled\":true},"
        + "\"prompt_cache\":{\"enabled\":false,\"maximum_size_gb\":7},"
        + "\"diagnostics\":{\"log_level\":\"debug\",\"retained_log_files\":3,"
        + "\"performance_attribution_enabled\":true,\"completion_attribution_enabled\":true}}";

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

    private func loadConfig(
        contents: String,
        fixture: TemporaryDirectoryFixture
    ) throws -> AstronomicalConfig {
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try self.writeConfigFile(contents: contents, beneathStateDirectory: developmentPaths.stateDirectory);
        return try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
    }

    func testFirstRunDefaultsResolveTheWorkerStartupPolicy() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: RuntimeAccessorsTests.MINIMAL_VALID_CONFIG_JSON,
            fixture: fixture);

        let promptCacheConfig: PromptCacheConfig = try developmentConfig.promptCache();
        XCTAssertEqual(
            promptCacheConfig.globalPromptCacheRootDirectory,
            developmentConfig.instancePaths.promptCacheDirectory);
        XCTAssertEqual(promptCacheConfig.globalPromptCacheMaximumSizeBytes, 50_000_000_000);
        XCTAssertEqual(promptCacheConfig.activeModelPromptCacheDirectory, promptCacheConfig.globalPromptCacheRootDirectory);
        XCTAssertTrue(developmentConfig.persistentPromptCacheEnabled());
        XCTAssertNil(developmentConfig.configuredPersistentPromptCacheEnabled());
        XCTAssertNil(try developmentConfig.configuredPromptCacheMaximumSizeBytes());

        let loggingConfig: LoggingConfig = developmentConfig.logging();
        XCTAssertEqual(loggingConfig.directory, developmentConfig.instancePaths.loggingDirectory);
        XCTAssertEqual(loggingConfig.level, LogLevel.warn);
        XCTAssertEqual(loggingConfig.retainedFiles, 7);
        XCTAssertEqual(loggingConfig.bufferedLineLimit, 1024);

        XCTAssertNil(try developmentConfig.maximumMlxMemoryBytes());
        XCTAssertFalse(developmentConfig.performanceAttributionEnabled());
        XCTAssertFalse(developmentConfig.completionAttributionEnabled());
        XCTAssertFalse(developmentConfig.experimentalQwenThinkingChannelSeedEnabled());
    }

    func testAuthoredOverridesTakePrecedenceOverDefaults() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: RuntimeAccessorsTests.OVERRIDDEN_CONFIG_JSON,
            fixture: fixture);

        let promptCacheConfig: PromptCacheConfig = try developmentConfig.promptCache();
        XCTAssertEqual(promptCacheConfig.globalPromptCacheMaximumSizeBytes, 7_000_000_000);
        XCTAssertFalse(developmentConfig.persistentPromptCacheEnabled());
        XCTAssertEqual(developmentConfig.configuredPersistentPromptCacheEnabled(), false);
        XCTAssertEqual(try developmentConfig.configuredPromptCacheMaximumSizeBytes(), 7_000_000_000);

        let loggingConfig: LoggingConfig = developmentConfig.logging();
        XCTAssertEqual(loggingConfig.level, LogLevel.debug);
        XCTAssertEqual(loggingConfig.retainedFiles, 3);

        XCTAssertEqual(try developmentConfig.maximumMlxMemoryBytes(), 32_000_000_000);
        XCTAssertTrue(developmentConfig.performanceAttributionEnabled());
        XCTAssertTrue(developmentConfig.completionAttributionEnabled());
        XCTAssertTrue(developmentConfig.experimentalQwenThinkingChannelSeedEnabled());
    }

    func testPerModelCacheDirectoryStaysBeneathTheGlobalRoot() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: RuntimeAccessorsTests.MINIMAL_VALID_CONFIG_JSON,
            fixture: fixture);

        let modelCacheConfig: PromptCacheConfig = try developmentConfig.promptCache().forModel(
            modelId: "qwen3-model",
            revision: "main");
        XCTAssertEqual(modelCacheConfig.globalPromptCacheRootDirectory, developmentConfig.instancePaths.promptCacheDirectory);
        XCTAssertEqual(
            modelCacheConfig.activeModelPromptCacheDirectory.string.hasSuffix("/qwen3-model/main"),
            true,
            "active cache directory should be model-and-revision scoped: \(modelCacheConfig.activeModelPromptCacheDirectory)");
        XCTAssertEqual(modelCacheConfig.globalPromptCacheMaximumSizeBytes, 50_000_000_000);
    }

    func testZeroPromptCacheCapacityIsRejected() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let zeroCapacityConfigJson: String =
            "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
            + "\"runtime\":{\"model_directories\":[]},\"prompt_cache\":{\"maximum_size_gb\":0}}";
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: zeroCapacityConfigJson,
            fixture: fixture);

        XCTAssertThrowsError(try developmentConfig.promptCache()) { (thrownError: any Error) in
            guard case AstronomicalConfigError.invalidPromptCacheMaxSizeGb = thrownError else {
                return XCTFail("expected a prompt-cache capacity rejection, got \(thrownError)");
            }
        }
    }

    func testGenerationDigestIsStableAndHexShaped() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: RuntimeAccessorsTests.MINIMAL_VALID_CONFIG_JSON,
            fixture: fixture);

        let firstGeneration: String = try developmentConfig.generation();
        let reloadedConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(
            developmentConfig.instancePaths);
        XCTAssertEqual(firstGeneration, try reloadedConfig.generation());
        XCTAssertEqual(firstGeneration.count, 64);
        let hexCharacterSet: Set<Character> = Set<Character>("0123456789abcdef");
        XCTAssertTrue(
            firstGeneration.allSatisfy { (generationCharacter: Character) -> Bool in
                return hexCharacterSet.contains(generationCharacter)
            },
            "generation should be lowercase hex: \(firstGeneration)");
    }
}
