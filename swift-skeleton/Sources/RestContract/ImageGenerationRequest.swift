import Foundation;
import IpcProtocol;

/// Strict request validation for the OpenAI-compatible image-generation boundary.
/// Port of crates/rest-contract/src/openai_image_generation_request.rs.

/// Smallest image side supported by the initial native image profile.
public let MIN_OPENAI_IMAGE_DIMENSION_PIXELS: UInt32 = 64;
/// Largest image side supported by the initial native image profile.
public let MAX_OPENAI_IMAGE_DIMENSION_PIXELS: UInt32 = 1_024;

private let IMAGE_DIMENSION_MULTIPLE_PIXELS: UInt32 = 16;

/// One unknown top-level field absorbed by serde's flatten, kept in
/// BTreeMap (byte-ordered) sequence so the first rejection matches Rust.
private struct UnknownRequestField: Equatable {
    fileprivate let fieldName: String;
    fileprivate let fieldValue: JsonWireValue;
}

/// One strict request to the local OpenAI-compatible image generation endpoint.
///
/// The diffusion schedule is deliberately absent from the wire: the worker's model profile
/// owns the step count and guidance (the reference default schedule), so callers cannot
/// trade away quality with a mistuned control value.
public struct OpenAiImageGenerationRequest: Equatable {
    private let modelText: String;
    private let promptText: String;
    private let seedValue: UInt64?;
    private let widthPixels: UInt32;
    private let heightPixels: UInt32;
    private let responseFormatName: String;
    private let imageCount: UInt32;
    private let unknownFields: Array<UnknownRequestField>;

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiImageGenerationRequest {
        let requestObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        let knownFieldNames: Array<String> = ["model", "prompt", "seed", "width", "height", "response_format", "n"];
        var unknownFields: Array<UnknownRequestField> = Array();
        for propertyName: String in requestObject.keyNames {
            if knownFieldNames.contains(propertyName) == false {
                unknownFields.append(UnknownRequestField(
                    fieldName: propertyName, fieldValue: requestObject.value(forKey: propertyName)!));
            }
        }
        unknownFields.sort { (leftEntry: UnknownRequestField, rightEntry: UnknownRequestField) -> Bool in
            return Array(leftEntry.fieldName.utf8).lexicographicallyPrecedes(Array(rightEntry.fieldName.utf8));
        };
        return OpenAiImageGenerationRequest(
            modelText: try requestObject.decodeString(fieldName: "model"),
            promptText: try requestObject.decodeString(fieldName: "prompt"),
            seedValue: try requestObject.decodeOptionalUInt64AllowingAbsent(fieldName: "seed"),
            widthPixels: try requestObject.decodeUInt32(fieldName: "width"),
            heightPixels: try requestObject.decodeUInt32(fieldName: "height"),
            responseFormatName: try requestObject.decodeString(fieldName: "response_format"),
            imageCount: try requestObject.decodeOptionalUInt32AllowingAbsent(fieldName: "n") ?? 1,
            unknownFields: unknownFields);
    }

    /// Validates and consumes public input before queue admission or model loading.
    public func intoParts() throws -> OpenAiImageGenerationRequestParts {
        if let firstUnknownField: UnknownRequestField = self.unknownFields.first {
            throw OpenAiImageGenerationValidationError.unknownField(fieldName: firstUnknownField.fieldName);
        }
        if self.modelText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw OpenAiImageGenerationValidationError.blankModel;
        }
        if self.promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw OpenAiImageGenerationValidationError.blankPrompt;
        }
        try ImageGenerationValidation.validateDimension(parameterName: "width", actualPixels: self.widthPixels);
        try ImageGenerationValidation.validateDimension(parameterName: "height", actualPixels: self.heightPixels);
        if self.responseFormatName != "b64_json" {
            throw OpenAiImageGenerationValidationError.unsupportedResponseFormat(responseFormat: self.responseFormatName);
        }
        if self.imageCount != 1 {
            throw OpenAiImageGenerationValidationError.unsupportedImageCount(actualImages: self.imageCount);
        }
        return OpenAiImageGenerationRequestParts(
            model: self.modelText,
            prompt: self.promptText,
            seed: self.seedValue,
            width: self.widthPixels,
            height: self.heightPixels,
            responseFormat: .base64Json,
            imageCount: self.imageCount);
    }
}

/// Validated image request data ready for supervisor translation.
public struct OpenAiImageGenerationRequestParts: Equatable {
    public let model: String;
    public let prompt: String;
    public let seed: UInt64?;
    public let width: UInt32;
    public let height: UInt32;
    public let responseFormat: OpenAiImageGenerationResponseFormat;
    public let imageCount: UInt32;

    public init(
        model: String, prompt: String, seed: UInt64?, width: UInt32, height: UInt32,
        responseFormat: OpenAiImageGenerationResponseFormat, imageCount: UInt32) {
        self.model = model;
        self.prompt = prompt;
        self.seed = seed;
        self.width = width;
        self.height = height;
        self.responseFormat = responseFormat;
        self.imageCount = imageCount;
    }
}

/// Output encoding admitted by the initial local image endpoint.
public enum OpenAiImageGenerationResponseFormat: Equatable {
    case base64Json;
}

/// A request rejected before image-model queue admission.
public enum OpenAiImageGenerationValidationError: Error, Equatable {
    /// The caller supplied a field outside the supported request contract.
    case unknownField(fieldName: String);
    /// Model selection cannot resolve an empty identifier.
    case blankModel;
    /// Text conditioning requires visible prompt content.
    case blankPrompt;
    /// Native image geometry is bounded and aligned before expensive admission.
    case unsupportedDimension(
        parameterName: String, actualPixels: UInt32, minimumPixels: UInt32, maximumPixels: UInt32);
    /// Only inline base64 JSON preserves the initial endpoint's local transport contract.
    case unsupportedResponseFormat(responseFormat: String);
    /// The initial endpoint executes and returns one image per request.
    case unsupportedImageCount(actualImages: UInt32);

    public var errorDescription: String? {
        switch self {
        case .unknownField(let fieldName):
            return "request field '\(fieldName)' is unknown";
        case .blankModel:
            return "model must not be blank";
        case .blankPrompt:
            return "prompt must not be blank";
        case .unsupportedDimension(let parameterName, let actualPixels, let minimumPixels, let maximumPixels):
            return "\(parameterName) must be a multiple of 16 in the \(minimumPixels)..=\(maximumPixels) pixel range, received \(actualPixels)";
        case .unsupportedResponseFormat(let responseFormat):
            return "image response format '\(responseFormat)' is unsupported";
        case .unsupportedImageCount(let actualImages):
            return "image generation supports exactly one image, received \(actualImages)";
        }
    }
}

/// Dimension validation shared by the image-generation request boundary,
/// mirroring the private helper in the Rust request unit.
internal enum ImageGenerationValidation {

    internal static func validateDimension(parameterName: String, actualPixels: UInt32) throws -> Void {
        let isSupported: Bool = actualPixels >= MIN_OPENAI_IMAGE_DIMENSION_PIXELS
            && actualPixels <= MAX_OPENAI_IMAGE_DIMENSION_PIXELS
            && actualPixels % IMAGE_DIMENSION_MULTIPLE_PIXELS == 0;
        if isSupported {
            return;
        }
        throw OpenAiImageGenerationValidationError.unsupportedDimension(
            parameterName: parameterName,
            actualPixels: actualPixels,
            minimumPixels: MIN_OPENAI_IMAGE_DIMENSION_PIXELS,
            maximumPixels: MAX_OPENAI_IMAGE_DIMENSION_PIXELS);
    }
}
