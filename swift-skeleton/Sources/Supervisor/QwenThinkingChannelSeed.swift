import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// Config-gated loading of the optional Qwen `thinking.md` seed from
/// instance state.
///
/// Mirrors apps/supervisor/src/qwen_thinking_channel_seed.rs: missing, empty,
/// unreadable, non-UTF-8, or oversized files are absent — the seed is a
/// convenience the user authors, never a serving dependency. The supervisor
/// never creates this file. The synchronous local read stays bounded by the
/// seed byte cap, which is the same bound the Rust loader's async read enforces.
enum QwenThinkingChannelSeed {

    /// Loads the configured seed for one model, or nil when disabled, absent,
    /// or unloadable. Only a discovered Qwen3.5 model with the experimental
    /// flag enabled reads instance state.
    static func load(
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        instancePaths: AstronomicalInstancePaths?,
        modelId: String
    ) -> String? {
        guard resolvedRuntimeConfig.experimentalQwenThinkingChannelSeedEnabled else {
            return nil;
        }
        let isSeededModel: Bool = resolvedRuntimeConfig.discoveredModels.contains { (discoveredModel: DiscoveryDiscoveredModel) -> Bool in
            return discoveredModel.modelId == modelId && discoveredModel.modelFamily == .qwen35;
        };
        guard isSeededModel, let instancePaths = instancePaths else {
            return nil;
        }
        return QwenThinkingChannelSeed.readBoundedSeed(
            seedFilePath: instancePaths.qwenThinkingChannelSeedFilePath.string);
    }

    private static func readBoundedSeed(seedFilePath: String) -> String? {
        guard let seedFileHandle: FileHandle = FileHandle(forReadingAtPath: seedFilePath) else {
            return nil;
        }
        defer { seedFileHandle.closeFile(); }
        // One byte over the bound marks an oversized file; the read itself is
        // bounded so a huge file cannot balloon worker memory.
        let boundedSeedData: Data = seedFileHandle.readData(
            ofLength: ChatGenerationCommand.maximumQwenThinkingChannelSeedBytes + 1);
        guard let seedText: String = String(data: boundedSeedData, encoding: .utf8) else {
            return nil;
        }
        if seedText.utf8.count > ChatGenerationCommand.maximumQwenThinkingChannelSeedBytes {
            return nil;
        }
        let trimmedSeedText: String = seedText.trimmingCharacters(in: .whitespacesAndNewlines);
        if trimmedSeedText.isEmpty {
            return nil;
        }
        return trimmedSeedText;
    }
}
