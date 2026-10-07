import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic file-boundary journeys for the optional bounded Qwen thinking
 * seed, migrating apps/supervisor/tests/hermetic/
 * qwen_thinking_channel_seed.rs: the seed loads from the instance state
 * only for a discovered Qwen3.5 model with the experiment enabled, and a
 * missing, whitespace-only, non-UTF-8, disabled, or over-boundary file is
 * simply absent — the seed is a convenience the user authors, never a
 * serving dependency.
 */
@Suite(.tags(.hermeticJourney))
final class QwenThinkingChannelSeedTests {

    private static let seededModelId: String = "astronomical/thinking-seed-model";

    @Test
    func should_load_thinking_markdown_from_a_temp_instance_state_directory() throws {
        let seededInstancePaths: AstronomicalInstancePaths = try QwenThinkingChannelSeedTests.seededInstancePaths(
            thinkingFileBytes: Array("Two households, both alike in dignity, in Romeo and Juliet.\n".utf8));

        let thinkingSeed: String? = QwenThinkingChannelSeed.load(
            resolvedRuntimeConfig: try QwenThinkingChannelSeedTests.seedEnabledConfig(),
            instancePaths: seededInstancePaths,
            modelId: QwenThinkingChannelSeedTests.seededModelId);

        #expect(
            thinkingSeed == "Two households, both alike in dignity, in Romeo and Juliet.",
            "the seed text loads trimmed, got: \(thinkingSeed ?? "nil")");
    }

    @Test
    func should_treat_a_missing_thinking_markdown_file_as_absent() throws {
        let seededInstancePaths: AstronomicalInstancePaths = QwenThinkingChannelSeedTests.bareInstancePaths();

        let thinkingSeed: String? = QwenThinkingChannelSeed.load(
            resolvedRuntimeConfig: try QwenThinkingChannelSeedTests.seedEnabledConfig(),
            instancePaths: seededInstancePaths,
            modelId: QwenThinkingChannelSeedTests.seededModelId);

        #expect(thinkingSeed == nil, "a missing file must read as absent");
    }

    @Test
    func should_treat_whitespace_only_thinking_markdown_as_absent() throws {
        let seededInstancePaths: AstronomicalInstancePaths = try QwenThinkingChannelSeedTests.seededInstancePaths(
            thinkingFileBytes: Array("  \n\t\n".utf8));

        let thinkingSeed: String? = QwenThinkingChannelSeed.load(
            resolvedRuntimeConfig: try QwenThinkingChannelSeedTests.seedEnabledConfig(),
            instancePaths: seededInstancePaths,
            modelId: QwenThinkingChannelSeedTests.seededModelId);

        #expect(thinkingSeed == nil, "whitespace-only content must read as absent");
    }

    @Test
    func should_ignore_existing_thinking_markdown_when_the_experiment_is_disabled() throws {
        let seededInstancePaths: AstronomicalInstancePaths = try QwenThinkingChannelSeedTests.seededInstancePaths(
            thinkingFileBytes: Array("Two households, both alike in dignity, in Romeo and Juliet.\n".utf8));

        let thinkingSeed: String? = QwenThinkingChannelSeed.load(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(
                discoveredModels: [QwenThinkingChannelSeedTests.seededQwenModel()]),
            instancePaths: seededInstancePaths,
            modelId: QwenThinkingChannelSeedTests.seededModelId);

        #expect(thinkingSeed == nil, "a disabled experiment must never read instance state");
    }

    @Test
    func should_ignore_thinking_markdown_that_exceeds_the_worker_boundary() throws {
        let seedByteBoundary: Int = ChatGenerationCommand.maximumQwenThinkingChannelSeedBytes;
        let seededInstancePaths: AstronomicalInstancePaths = try QwenThinkingChannelSeedTests.seededInstancePaths(
            thinkingFileBytes: Array(repeating: UInt8(ascii: "R"), count: seedByteBoundary + 1));

        let thinkingSeed: String? = QwenThinkingChannelSeed.load(
            resolvedRuntimeConfig: try QwenThinkingChannelSeedTests.seedEnabledConfig(),
            instancePaths: seededInstancePaths,
            modelId: QwenThinkingChannelSeedTests.seededModelId);

        #expect(thinkingSeed == nil, "one byte over the worker boundary must read as absent");
    }

    @Test
    func should_accept_thinking_markdown_at_the_exact_worker_boundary() throws {
        let seedByteBoundary: Int = ChatGenerationCommand.maximumQwenThinkingChannelSeedBytes;
        let seededInstancePaths: AstronomicalInstancePaths = try QwenThinkingChannelSeedTests.seededInstancePaths(
            thinkingFileBytes: Array(repeating: UInt8(ascii: "R"), count: seedByteBoundary));

        let thinkingSeed: String? = QwenThinkingChannelSeed.load(
            resolvedRuntimeConfig: try QwenThinkingChannelSeedTests.seedEnabledConfig(),
            instancePaths: seededInstancePaths,
            modelId: QwenThinkingChannelSeedTests.seededModelId);

        #expect(
            thinkingSeed?.utf8.count == seedByteBoundary,
            "boundary-sized content must load in full");
    }

    @Test
    func should_treat_non_utf8_thinking_markdown_as_absent() throws {
        let seededInstancePaths: AstronomicalInstancePaths = try QwenThinkingChannelSeedTests.seededInstancePaths(
            thinkingFileBytes: [0xff, 0xfe]);

        let thinkingSeed: String? = QwenThinkingChannelSeed.load(
            resolvedRuntimeConfig: try QwenThinkingChannelSeedTests.seedEnabledConfig(),
            instancePaths: seededInstancePaths,
            modelId: QwenThinkingChannelSeedTests.seededModelId);

        #expect(thinkingSeed == nil, "non-UTF-8 content must read as absent");
    }

    // MARK: Fixtures

    private static func seedEnabledConfig() throws -> ResolvedRuntimeConfig {
        return try RestChatJourneySupport.makeResolvedConfig(
            discoveredModels: [QwenThinkingChannelSeedTests.seededQwenModel()],
            experimentalQwenThinkingChannelSeedEnabled: true);
    }

    private static func seededQwenModel() -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: QwenThinkingChannelSeedTests.seededModelId,
            providerModelId: nil,
            modelFamily: .qwen35,
            revision: "thinking-seed-revision",
            modelDirectory: FilePath(string: "/fictional/models/thinking-seed-model"),
            capabilities: .chat(DiscoveryChatModelCapabilities(
                contextWindowTokens: 262_144,
                maximumInputTokens: 241_664,
                maximumOutputTokens: 20_480,
                supportsVision: false,
                supportsReasoning: true,
                supportsToolCalls: true)),
            license: nil,
            modelSizeBytes: 1_000);
    }

    private static func bareInstancePaths() -> AstronomicalInstancePaths {
        let stateDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-thinking-seed-\(UUID().uuidString)", isDirectory: true);
        try? FileManager.default.createDirectory(at: stateDirectoryUrl, withIntermediateDirectories: true);
        return AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: stateDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
    }

    private static func seededInstancePaths(thinkingFileBytes: Array<UInt8>) throws -> AstronomicalInstancePaths {
        let instancePaths: AstronomicalInstancePaths = QwenThinkingChannelSeedTests.bareInstancePaths();
        let seedFileUrl: URL = URL(fileURLWithPath: instancePaths.qwenThinkingChannelSeedFilePath.string);
        try FileManager.default.createDirectory(
            at: seedFileUrl.deletingLastPathComponent(),
            withIntermediateDirectories: true);
        try Data(thinkingFileBytes).write(to: seedFileUrl);
        return instancePaths;
    }
}
