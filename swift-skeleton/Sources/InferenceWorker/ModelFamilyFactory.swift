import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import ModelServing;

/// Maps a selected model directory onto its concrete serving runtime.
///
/// Mirrors apps/inference-worker/src/model_family_factory.rs: family
/// selection classifies the directory (never guesses from filenames), and
/// each family builds its matched processor and engine pair. The Qwen3.5
/// dense runtime validates the artifact and streams its shard weights into
/// the engine; families without a Swift runtime yet fail closed with the
/// bounded reason the `modelSwapFailed` event carries.
public struct ModelFamilyFactory: ChatModelRuntimeFactory {

    private let performanceAttributionEnabled: Bool;

    private let persistentPromptCachePolicy: Qwen35MoePromptCacheSpawnPolicy?;

    public init(
        performanceAttributionEnabled: Bool = false,
        persistentPromptCachePolicy: Qwen35MoePromptCacheSpawnPolicy? = nil
    ) {
        self.performanceAttributionEnabled = performanceAttributionEnabled;
        self.persistentPromptCachePolicy = persistentPromptCachePolicy;
    }

    public func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime {
        let modelFamily: ModelFamily? = try FamilyDiscovery.classifyModelDirectory(
            modelDirectory: FilePath(string: modelDirectory),
            attributionEnabled: self.performanceAttributionEnabled);
        switch (modelFamily, modelConfiguration) {
        case (.qwen35, .autoregressive):
            return try Qwen35ChatRuntime.buildArtifactRuntime(
                modelDirectory: modelDirectory,
                modelConfiguration: modelConfiguration,
                performanceAttributionEnabled: self.performanceAttributionEnabled,
                persistentPromptCachePolicy: self.persistentPromptCachePolicy);
        case (.none, _):
            throw WorkerModelLoadFailure.unclassifiedModelDirectory;
        case let (.some(unusableFamily), _):
            throw WorkerModelLoadFailure.familyWithoutChatRuntime(
                familyName: unusableFamily.rawValue);
        }
    }
}

/// Bounded model-load failure reasons the swap event carries.
enum WorkerModelLoadFailure: Error, CustomStringConvertible {

    case unclassifiedModelDirectory;
    case familyWithoutChatRuntime(familyName: String);

    var description: String {
        switch self {
        case .unclassifiedModelDirectory:
            return "selected model family could not be classified";
        case let .familyWithoutChatRuntime(familyName):
            return "the \(familyName) family has no chat runtime in this worker build";
        }
    }
}
