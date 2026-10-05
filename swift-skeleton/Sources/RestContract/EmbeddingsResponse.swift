import Foundation;
import IpcProtocol;

/// OpenAI embeddings response serialization for native local vectors.
/// Port of crates/rest-contract/src/openai_embeddings_response.rs.

/// One completed embedding row in request order.
public struct OpenAiEmbedding: Equatable {
    public let object: String;
    public let index: UInt32;
    public let embedding: OpenAiEmbeddingVector;

    /// Builds one embedding row with the OpenAI `embedding` object type.
    public init(index: UInt32, embedding: OpenAiEmbeddingVector) {
        self.object = "embedding";
        self.index = index;
        self.embedding = embedding;
    }

    public func wireValue() -> JsonWireValue {
        var embeddingObject: JsonWireObject = JsonWireObject(entries: Array());
        embeddingObject.appendEntry(key: "object", value: .string(self.object));
        embeddingObject.appendEntry(key: "index", value: .unsignedInteger(UInt64(self.index)));
        embeddingObject.appendEntry(key: "embedding", value: self.embedding.wireValue());
        return .object(embeddingObject);
    }
}

/// Float components or base64-encoded Float32 little-endian bytes. The Rust
/// type uses a custom Serialize that unwraps the enum payload, so the wire
/// value is the bare array or string, never an object with a variant tag.
public enum OpenAiEmbeddingVector: Equatable {
    case float(Array<Float>);
    case base64(String);

    public func wireValue() -> JsonWireValue {
        switch self {
        case .float(let components):
            return JsonWireValue.mappedArray(components, mappedWireValue: { (component: Float) -> JsonWireValue in
                return .float32(component);
            });
        case .base64(let encodedText):
            return .string(encodedText);
        }
    }
}

/// One OpenAI embeddings list response.
public struct OpenAiEmbeddingsResponse: Equatable {
    public let object: String;
    public let data: Array<OpenAiEmbedding>;
    public let model: String;
    public let usage: OpenAiTokenUsage;

    /// Serializes the complete validated embedding list with token usage.
    public init(embeddings: Array<OpenAiEmbedding>, model: String, usage: OpenAiTokenUsage) {
        self.object = "list";
        self.data = embeddings;
        self.model = model;
        self.usage = usage;
    }

    public func wireValue() -> JsonWireValue {
        var responseObject: JsonWireObject = JsonWireObject(entries: Array());
        responseObject.appendEntry(key: "object", value: .string(self.object));
        responseObject.appendEntry(
            key: "data",
            value: JsonWireValue.mappedArray(self.data, mappedWireValue: { (embedding: OpenAiEmbedding) -> JsonWireValue in
                return embedding.wireValue();
            }));
        responseObject.appendEntry(key: "model", value: .string(self.model));
        responseObject.appendEntry(key: "usage", value: self.usage.wireValue());
        return .object(responseObject);
    }
}
