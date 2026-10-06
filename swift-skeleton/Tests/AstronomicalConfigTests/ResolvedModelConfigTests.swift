import Foundation;

import Testing;

import JourneyCategories;

@testable import AstronomicalConfig;

/**
 * Acceptance journeys for one model's inherited policy: context/output
 * limits, generation defaults, and merged chunking, mirroring the Rust
 * resolved_model_config tests. Config may narrow an artifact's capability
 * but never widen it.
 */
@Suite(.tags(.hermeticJourney))
final class ResolvedModelConfigTests {

    private static let DEFAULT_MODEL_CONFIG_JSON: String =
        "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}";

    private static func modelOverrideConfigJson(modelOverrideObject: String) -> String {
        return "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
            + "\"runtime\":{\"model_directories\":[]},\"models\":{\"qwen3-model\":"
            + modelOverrideObject + "}}";
    }

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

    @Test
    func should_keep_the_artifact_context_and_internal_output_default_for_an_unconfigured_model() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.DEFAULT_MODEL_CONFIG_JSON,
            fixture: fixture);

        let resolvedModelConfig: ResolvedModelConfig = try developmentConfig.resolvedModelConfig(
            modelId: "qwen3-model",
            artifactMaximumContextTokens: 32_768);

        #expect(resolvedModelConfig.maximumContextTokens() == nil);
        #expect(resolvedModelConfig.maximumOutputTokens() == ResolvedModelConfig.defaultMaximumOutputTokens);
        #expect(!resolvedModelConfig.hasExplicitMaximumOutputTokens());
        #expect(resolvedModelConfig.temperature() == nil);
        #expect(resolvedModelConfig.topP() == nil);
        #expect(resolvedModelConfig.chunking() == ChunkingConfig.defaultConfig());
        #expect(!resolvedModelConfig.configuredChunkingFields().fixedPromptProcessingChunkSizeTokens);
    }

    @Test
    func should_clamp_a_tiny_artifact_internal_output_default_to_one_prompt_token() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.DEFAULT_MODEL_CONFIG_JSON,
            fixture: fixture);

        let resolvedModelConfig: ResolvedModelConfig = try developmentConfig.resolvedModelConfig(
            modelId: "qwen3-model",
            artifactMaximumContextTokens: 8);

        #expect(resolvedModelConfig.maximumOutputTokens() == 7);
    }

    @Test
    func should_inherit_configured_limits_and_defaults_marked_as_configured() throws {
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

        #expect(resolvedModelConfig.maximumContextTokens() == 16_384);
        #expect(resolvedModelConfig.maximumOutputTokens() == 4_096);
        #expect(resolvedModelConfig.hasExplicitMaximumOutputTokens());
        #expect(resolvedModelConfig.temperature() == 0.5);
        #expect(resolvedModelConfig.topP() == 0.9);
    }

    @Test
    func should_let_the_per_model_chunking_override_win_over_global_defaults() throws {
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

        #expect(resolvedModelConfig.chunking().fixedPromptProcessingChunkSizeTokens() == 512);
        #expect(resolvedModelConfig.chunking().promptCacheCommonPrefixStrideBlocks() == 8);
        // Inherited fields keep the built-in defaults and stay unconfigured.
        #expect(resolvedModelConfig.chunking().fullAttentionKeyValueGrowthTokens() == 256);
        #expect(!resolvedModelConfig.configuredChunkingFields().fullAttentionKeyValueGrowthTokens);
        #expect(resolvedModelConfig.configuredChunkingFields().fixedPromptProcessingChunkSizeTokens);
        #expect(resolvedModelConfig.configuredChunkingFields().promptCacheCommonPrefixStrideBlocks);
    }

    @Test
    func should_reject_a_configured_context_above_the_artifact_maximum() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let overrideObject: String = "{\"limits\":{\"maximum_context_tokens\":65536}}";
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.modelOverrideConfigJson(modelOverrideObject: overrideObject),
            fixture: fixture);

        do {
            _ = try developmentConfig.resolvedModelConfig(
                modelId: "qwen3-model",
                artifactMaximumContextTokens: 32_768);
            Issue.record("expected a context-exceeds-artifact rejection");
        } catch let configError as AstronomicalConfigError {
            guard case let .configuredContextExceedsArtifact(
                modelId,
                configuredMaximumContextTokens,
                artifactMaximumContextTokens) = configError else {
                Issue.record(Comment(stringLiteral: "expected a context-exceeds-artifact rejection, got \(configError)"));
                return;
            }
            #expect(modelId == "qwen3-model");
            #expect(configuredMaximumContextTokens == 65_536);
            #expect(artifactMaximumContextTokens == 32_768);
        }
    }

    @Test
    func should_reject_a_configured_output_at_or_above_the_effective_context() throws {
        let fixture: TemporaryDirectoryFixture = try self.makeTemporaryDirectoryFixture();
        let overrideObject: String =
            "{\"limits\":{\"maximum_context_tokens\":16384},"
            + "\"generation_defaults\":{\"maximum_output_tokens\":16384}}";
        let developmentConfig: AstronomicalConfig = try self.loadConfig(
            contents: ResolvedModelConfigTests.modelOverrideConfigJson(modelOverrideObject: overrideObject),
            fixture: fixture);

        do {
            _ = try developmentConfig.resolvedModelConfig(
                modelId: "qwen3-model",
                artifactMaximumContextTokens: 32_768);
            Issue.record("expected an output-not-smaller-than-context rejection");
        } catch let configError as AstronomicalConfigError {
            guard case let .configuredOutputNotSmallerThanContext(
                modelId,
                configuredMaximumOutputTokens,
                effectiveMaximumContextTokens) = configError else {
                Issue.record(Comment(stringLiteral: "expected an output-not-smaller-than-context rejection, got \(configError)"));
                return;
            }
            #expect(modelId == "qwen3-model");
            #expect(configuredMaximumOutputTokens == 16_384);
            #expect(effectiveMaximumContextTokens == 16_384);
        }
    }
}
