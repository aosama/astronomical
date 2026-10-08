import Foundation;

import Testing;

import JourneyCategories;

@testable import AstronomicalConfig;

/**
 * Acceptance journeys for the resolved runtime accessors the worker-startup
 * bootstrap needs: prompt-cache policy, bounded logging, the optional MLX
 * ceiling, attribution toggles, and the configuration digest, mirroring the
 * precedence of crates/config/src/lib.rs.
 */
@Suite(.tags(.hermeticJourney))
final class RuntimeAccessorsTests {

    private static let MINIMAL_VALID_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}";

    private static let OVERRIDDEN_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
        + "\"runtime\":{\"model_directories\":[],\"maximum_mlx_memory_gb\":32},"
        + "\"prompt_cache\":{\"enabled\":false,\"maximum_size_gb\":7},"
        + "\"diagnostics\":{\"log_level\":\"debug\",\"retained_log_files\":3,"
        + "\"performance_attribution_enabled\":true,\"completion_attribution_enabled\":true}}";

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

    @Test
    func should_resolve_first_run_defaults_into_the_worker_startup_policy() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: RuntimeAccessorsTests.MINIMAL_VALID_CONFIG_JSON,
            fixture: fixture);

        let promptCacheConfig: PromptCacheConfig = try developmentConfig.promptCache();
        #expect(
            promptCacheConfig.globalPromptCacheRootDirectory
                == developmentConfig.instancePaths.promptCacheDirectory);
        #expect(promptCacheConfig.globalPromptCacheMaximumSizeBytes == 50_000_000_000);
        #expect(promptCacheConfig.activeModelPromptCacheDirectory == promptCacheConfig.globalPromptCacheRootDirectory);
        #expect(developmentConfig.persistentPromptCacheEnabled());
        #expect(developmentConfig.configuredPersistentPromptCacheEnabled() == nil);
        #expect(try developmentConfig.configuredPromptCacheMaximumSizeBytes() == nil);

        let loggingConfig: LoggingConfig = developmentConfig.logging();
        #expect(loggingConfig.directory == developmentConfig.instancePaths.loggingDirectory);
        #expect(loggingConfig.level == LogLevel.warn);
        #expect(loggingConfig.retainedFiles == 7);
        #expect(loggingConfig.bufferedLineLimit == 1024);

        #expect(try developmentConfig.maximumMlxMemoryBytes() == nil);
        #expect(!developmentConfig.performanceAttributionEnabled());
        #expect(!developmentConfig.completionAttributionEnabled());
    }

    @Test
    func should_let_authored_overrides_take_precedence_over_defaults() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: RuntimeAccessorsTests.OVERRIDDEN_CONFIG_JSON,
            fixture: fixture);

        let promptCacheConfig: PromptCacheConfig = try developmentConfig.promptCache();
        #expect(promptCacheConfig.globalPromptCacheMaximumSizeBytes == 7_000_000_000);
        #expect(!developmentConfig.persistentPromptCacheEnabled());
        #expect(developmentConfig.configuredPersistentPromptCacheEnabled() == false);
        #expect(try developmentConfig.configuredPromptCacheMaximumSizeBytes() == 7_000_000_000);

        let loggingConfig: LoggingConfig = developmentConfig.logging();
        #expect(loggingConfig.level == LogLevel.debug);
        #expect(loggingConfig.retainedFiles == 3);

        #expect(try developmentConfig.maximumMlxMemoryBytes() == 32_000_000_000);
        #expect(developmentConfig.performanceAttributionEnabled());
        #expect(developmentConfig.completionAttributionEnabled());
    }

    @Test
    func should_scope_the_per_model_cache_directory_beneath_the_global_root() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: RuntimeAccessorsTests.MINIMAL_VALID_CONFIG_JSON,
            fixture: fixture);

        let modelCacheConfig: PromptCacheConfig = try developmentConfig.promptCache().forModel(
            modelId: "qwen3-model",
            revision: "main");
        #expect(modelCacheConfig.globalPromptCacheRootDirectory == developmentConfig.instancePaths.promptCacheDirectory);
        #expect(
            modelCacheConfig.activeModelPromptCacheDirectory.string.hasSuffix("/qwen3-model/main"),
            "active cache directory should be model-and-revision scoped: \(modelCacheConfig.activeModelPromptCacheDirectory)");
        #expect(modelCacheConfig.globalPromptCacheMaximumSizeBytes == 50_000_000_000);
    }

    @Test
    func should_reject_a_zero_prompt_cache_capacity() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let zeroCapacityConfigJson: String =
            "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
            + "\"runtime\":{\"model_directories\":[]},\"prompt_cache\":{\"maximum_size_gb\":0}}";
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: zeroCapacityConfigJson,
            fixture: fixture);

        do {
            _ = try developmentConfig.promptCache();
            Issue.record("expected a prompt-cache capacity rejection");
        } catch let configError as AstronomicalConfigError {
            guard case AstronomicalConfigError.invalidPromptCacheMaxSizeGb = configError else {
                Issue.record(Comment(stringLiteral: "expected a prompt-cache capacity rejection, got \(configError)"));
                return;
            }
        }
    }

    @Test
    func should_keep_the_generation_digest_stable_and_hex_shaped() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: RuntimeAccessorsTests.MINIMAL_VALID_CONFIG_JSON,
            fixture: fixture);

        let firstGeneration: String = try developmentConfig.generation();
        let reloadedConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(
            developmentConfig.instancePaths);
        #expect(try firstGeneration == reloadedConfig.generation());
        #expect(firstGeneration.count == 64);
        let hexCharacterSet: Set<Character> = Set<Character>("0123456789abcdef");
        #expect(
            firstGeneration.allSatisfy { (generationCharacter: Character) -> Bool in
                return hexCharacterSet.contains(generationCharacter)
            },
            "generation should be lowercase hex: \(firstGeneration)");
    }
}
