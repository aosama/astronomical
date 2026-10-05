import Foundation;
import IpcProtocol;

/// Response ownership for one OpenAI-compatible base64 image result.
/// Port of crates/rest-contract/src/openai_image_generation_response.rs.

/// Validated generated-image content and reproducibility metadata.
public struct OpenAiGeneratedImageParts: Equatable {
    public let b64Json: String;
    public let mimeType: String;
    public let modelRevision: String;
    public let effectiveSeed: UInt64;
    public let width: UInt32;
    public let height: UInt32;

    public init(
        b64Json: String, mimeType: String, modelRevision: String, effectiveSeed: UInt64,
        width: UInt32, height: UInt32) {
        self.b64Json = b64Json;
        self.mimeType = mimeType;
        self.modelRevision = modelRevision;
        self.effectiveSeed = effectiveSeed;
        self.width = width;
        self.height = height;
    }
}

/// One OpenAI-compatible image generation response.
public struct OpenAiImageGenerationResponse: Equatable {
    private let createdTimestamp: UInt64;
    private let dataImages: Array<OpenAiGeneratedImage>;

    /// Builds the single-image response supported by the initial endpoint.
    public init(created: UInt64, generatedImageParts: OpenAiGeneratedImageParts) {
        self.createdTimestamp = created;
        self.dataImages = [OpenAiGeneratedImage(parts: generatedImageParts)];
    }

    public func wireValue() -> JsonWireValue {
        var responseObject: JsonWireObject = JsonWireObject(entries: Array());
        responseObject.appendEntry(key: "created", value: .unsignedInteger(self.createdTimestamp));
        responseObject.appendEntry(
            key: "data",
            value: JsonWireValue.mappedArray(self.dataImages, mappedWireValue: { (generatedImage: OpenAiGeneratedImage) -> JsonWireValue in
                return generatedImage.wireValue();
            }));
        return .object(responseObject);
    }
}

/// One encoded image returned inside an image generation response.
public struct OpenAiGeneratedImage: Equatable {
    private let b64JsonText: String;
    private let mimeTypeName: String;
    private let modelRevisionName: String;
    private let effectiveSeedValue: UInt64;
    private let widthPixels: UInt32;
    private let heightPixels: UInt32;

    fileprivate init(parts: OpenAiGeneratedImageParts) {
        self.b64JsonText = parts.b64Json;
        self.mimeTypeName = parts.mimeType;
        self.modelRevisionName = parts.modelRevision;
        self.effectiveSeedValue = parts.effectiveSeed;
        self.widthPixels = parts.width;
        self.heightPixels = parts.height;
    }

    public func wireValue() -> JsonWireValue {
        var imageObject: JsonWireObject = JsonWireObject(entries: Array());
        imageObject.appendEntry(key: "b64_json", value: .string(self.b64JsonText));
        imageObject.appendEntry(key: "mime_type", value: .string(self.mimeTypeName));
        imageObject.appendEntry(key: "model_revision", value: .string(self.modelRevisionName));
        imageObject.appendEntry(key: "seed", value: .unsignedInteger(self.effectiveSeedValue));
        imageObject.appendEntry(key: "width", value: .unsignedInteger(UInt64(self.widthPixels)));
        imageObject.appendEntry(key: "height", value: .unsignedInteger(UInt64(self.heightPixels)));
        return .object(imageObject);
    }
}
