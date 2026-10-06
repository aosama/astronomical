import Foundation;
import XCTest;
@testable import AstronomicalConfig;

/**
 * Acceptance journeys for one model's inherited policy: context/output
 * limits, generation defaults, and merged chunking, mirroring the Rust
 * resolved_model_config tests. Config may narrow an artifact's capability
 * but never widen it.
 */
final class ResolvedModelConfigTests: XCTestCase {

    private static let DEFAULT_MODEL_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}";

    private static func modelOverrideConfigJson(modelOverrideObject: String) -> String {
        return "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
            + "\"runtime\":{\"model_directories\":[]},\"models\":{\"qwen3-model\":"
            + modelOverrideObject + "}}";
    }

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

    private func loadConfig(
        contents: String,
        fixture: TemporaryDirectoryFixture
    ) throws -> AstronomicalConfig {
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            fixture.rootDirectoryPath,
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
        try FileManager.default.createDirectory(
            atPath: developmentPaths.stateDirectory.string,
            withIntermediateDirectories: true);
        let configFileUrl: URL = URL(fileURLWithPath: developmentPaths.stateDirectory.appending(component: "config.json").string);
        try contents.write(to: configFileUrl, atomically: true, encoding: String.Encoding.utf8);
        return try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
    }

    func testUnconfiguredModelKeepsArtifactContextAndInternalOutputDefault() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.DEFAULT_MODEL_CONFIG_JSON,
            fixture: fixture);

        let resolvedModelConfig: ResolvedModelConfig = try developmentConfig.resolvedModelConfig(
            modelId: "qwen3-model",
            artifactMaximumContextTokens: 32_768);

        XCTAssertNil(resolvedModelConfig.maximumContextTokens());
        XCTAssertEqual(resolvedModelConfig.maximumOutputTokens(), ResolvedModelConfig.defaultMaximumOutputTokens);
        XCTAssertFalse(resolvedModelConfig.hasExplicitMaximumOutputTokens());
        XCTAssertNil(resolvedModelConfig.temperature());
        XCTAssertNil(resolvedModelConfig.topP());
        XCTAssertEqual(resolvedModelConfig.chunking(), ChunkingConfig.defaultConfig());
        XCTAssertFalse(resolvedModelConfig.configuredChunkingFields().fixedPromptProcessingChunkSizeTokens);
    }

    func testTinyArtifactClampsTheInternalOutputDefaultToOnePromptToken() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.DEFAULT_MODEL_CONFIG_JSON,
            fixture: fixture);

        let resolvedModelConfig: ResolvedModelConfig = try developmentConfig.resolvedModelConfig(
            modelId: "qwen3-model",
            artifactMaximumContextTokens: 8);

        XCTAssertEqual(resolvedModelConfig.maximumOutputTokens(), 7);
    }

    func testConfiguredLimitsAndDefaultsInheritAndAreMarkedAsConfigured() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let overrideObject: String =
            "{\"limits\":{\"maximum_context_tokens\":16384},"
            + "\"generation_defaults\":{\"maximum_output_tokens\":4096,\"temperature\":0.5,\"top_p\":0.9}}";
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.modelOverrideConfigJson(modelOverrideObject: overrideObject),
            fixture: fixture);

        let resolvedModelConfig: ResolvedModelConfig = try developmentConfig.resolvedModelConfig(
            modelId: "qwen3-model",
            artifactMaximumContextTokens: 32_768);

        XCTAssertEqual(resolvedModelConfig.maximumContextTokens(), 16_384);
        XCTAssertEqual(resolvedModelConfig.maximumOutputTokens(), 4_096);
        XCTAssertTrue(resolvedModelConfig.hasExplicitMaximumOutputTokens());
        XCTAssertEqual(resolvedModelConfig.temperature(), 0.5);
        XCTAssertEqual(resolvedModelConfig.topP(), 0.9);
    }

    func testPerModelChunkingOverrideWinsOverGlobalDefaults() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let overrideObject: String =
            "{\"chunking\":{\"fixed_prompt_processing_chunk_size_tokens\":512,"
            + "\"prompt_cache_common_prefix_stride_blocks\":8}}";
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.modelOverrideConfigJson(modelOverrideObject: overrideObject),
            fixture: fixture);

        let resolvedModelConfig: ResolvedModelConfig = try developmentConfig.resolvedModelConfig(
            modelId: "qwen3-model",
            artifactMaximumContextTokens: 32_768);

        XCTAssertEqual(resolvedModelConfig.chunking().fixedPromptProcessingChunkSizeTokens(), 512);
        XCTAssertEqual(resolvedModelConfig.chunking().promptCacheCommonPrefixStrideBlocks(), 8);
        // Inherited fields keep the built-in defaults and stay unconfigured.
        XCTAssertEqual(resolvedModelConfig.chunking().fullAttentionKeyValueGrowthTokens(), 256);
        XCTAssertFalse(resolvedModelConfig.configuredChunkingFields().fullAttentionKeyValueGrowthTokens);
        XCTAssertTrue(resolvedModelConfig.configuredChunkingFields().fixedPromptProcessingChunkSizeTokens);
        XCTAssertTrue(resolvedModelConfig.configuredChunkingFields().promptCacheCommonPrefixStrideBlocks);
    }

    func testConfiguredContextAboveTheArtifactMaximumIsRejected() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let overrideObject: String = "{\"limits\":{\"maximum_context_tokens\":65536}}";
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.modelOverrideConfigJson(modelOverrideObject: overrideObject),
            fixture: fixture);

        XCTAssertThrowsError(try developmentConfig.resolvedModelConfig(
            modelId: "qwen3-model",
            artifactMaximumContextTokens: 32_768)) { (thrownError: any Error) in
            guard case let AstronomicalConfigError.configuredContextExceedsArtifact(
                modelId,
                configuredMaximumContextTokens,
                artifactMaximumContextTokens) = thrownError else {
                return XCTFail("expected a context-exceeds-artifact rejection, got \(thrownError)");
            }
            XCTAssertEqual(modelId, "qwen3-model");
            XCTAssertEqual(configuredMaximumContextTokens, 65_536);
            XCTAssertEqual(artifactMaximumContextTokens, 32_768);
        }
    }

    func testConfiguredOutputAtOrAboveTheEffectiveContextIsRejected() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let overrideObject: String =
            "{\"limits\":{\"maximum_context_tokens\":16384},"
            + "\"generation_defaults\":{\"maximum_output_tokens\":16384}}";
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.modelOverrideConfigJson(modelOverrideObject: overrideObject),
            fixture: fixture);

        XCTAssertThrowsError(try developmentConfig.resolvedModelConfig(
            modelId: "qwen3-model",
            artifactMaximumContextTokens: 32_768)) { (thrownError: any Error) in
            guard case let AstronomicalConfigError.configuredOutputNotSmallerThanContext(
                modelId,
                configuredMaximumOutputTokens,
                effectiveMaximumContextTokens) = thrownError else {
                return XCTFail("expected an output-not-smaller-than-context rejection, got \(thrownError)");
            }
            XCTAssertEqual(modelId, "qwen3-model");
            XCTAssertEqual(configuredMaximumOutputTokens, 16_384);
            XCTAssertEqual(effectiveMaximumContextTokens, 16_384);
        }
    }
}
