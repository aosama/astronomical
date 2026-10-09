import Foundation;

/// Validated persistence contract for projected image embeddings, port of
/// the Rust `PersistentVisualEmbeddingModelContract`.
public struct PersistentVisualEmbeddingModelContract: Equatable, Sendable {

    public let modelId: String;

    public let modelRevision: String;

    public let projectedEmbeddingHiddenSize: Int;

    public let maximumVisualEmbeddingTokenCount: Int;

    /// Binds a projected embedding layout to one exact model artifact.
    public init(
        modelId: String,
        modelRevision: String,
        projectedEmbeddingHiddenSize: Int,
        maximumVisualEmbeddingTokenCount: Int
    ) {
        self.modelId = modelId;
        self.modelRevision = modelRevision;
        self.projectedEmbeddingHiddenSize = projectedEmbeddingHiddenSize;
        self.maximumVisualEmbeddingTokenCount = maximumVisualEmbeddingTokenCount;
    }

    /// The persisted projected visual embedding shape.
    public func visualEmbeddingShape(visualTokenCount: Int) -> [Int] {
        return [visualTokenCount, self.projectedEmbeddingHiddenSize];
    }

    /// The projected visual embedding width consumed by the text model.
    public var visualEmbeddingHiddenSize: Int {
        return self.projectedEmbeddingHiddenSize;
    }

    /// The maximum visual rows accepted in one persisted image file.
    public var maximumVisualTokenCount: Int {
        return self.maximumVisualEmbeddingTokenCount;
    }
}
