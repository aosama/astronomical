import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import ModelServing;

/// Maps a selected model directory onto its concrete serving runtime.
///
/// Mirrors apps/inference-worker/src/model_family_factory.rs: family
/// selection classifies the directory (never guesses from filenames), and
/// each family builds its matched processor and engine pair.
///
/// E2a scope: the Qwen3.5 dense runtime's weight loading from validated
/// shards lands with the artifact streaming slice, so real directories fail
/// closed here with the bounded reason the `modelSwapFailed` event carries.
/// The engine itself is fully proven through the in-memory journeys in
/// ModelServingTests.
public struct ModelFamilyFactory: ChatModelRuntimeFactory {

    private let performanceAttributionEnabled: Bool;

    public init(performanceAttributionEnabled: Bool = false) {
        self.performanceAttributionEnabled = performanceAttributionEnabled;
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
            throw WorkerModelLoadFailure.denseWeightsPendingArtifactStreamingSlice;
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

    case denseWeightsPendingArtifactStreamingSlice;
    case unclassifiedModelDirectory;
    case familyWithoutChatRuntime(familyName: String);

    var description: String {
        switch self {
        case .denseWeightsPendingArtifactStreamingSlice:
            return "Qwen3.5 dense weight loading lands with the artifact streaming slice";
        case .unclassifiedModelDirectory:
            return "selected model family could not be classified";
        case let .familyWithoutChatRuntime(familyName):
            return "the \(familyName) family has no chat runtime in this worker build";
        }
    }
}
