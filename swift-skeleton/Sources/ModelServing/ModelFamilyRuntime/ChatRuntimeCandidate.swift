import Foundation;

/// A validated runtime selection whose MLX resources are created only when
/// the worker has retired its prior model owner.
public final class ChatRuntimeCandidate {

    public let validatedArtifact: ValidatedQwen35Artifact?;

    private var runtimeLoader: (() throws -> LoadedChatRuntime)?;

    public init(
        validatedArtifact: ValidatedQwen35Artifact? = nil,
        runtimeLoader: @escaping () throws -> LoadedChatRuntime
    ) {
        self.validatedArtifact = validatedArtifact;
        self.runtimeLoader = runtimeLoader;
    }

    /// Loads and binds the selected model after the prior worker runtime is
    /// released.
    public func load() throws -> LoadedChatRuntime {
        guard let runtimeLoader: () throws -> LoadedChatRuntime = self.runtimeLoader else {
            throw InferenceEngineError.modelLoad(
                reason: "the selected chat runtime candidate was already consumed");
        }
        self.runtimeLoader = nil;
        return try runtimeLoader();
    }
}
