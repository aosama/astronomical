import Foundation;

/// One validated text-to-image request sent to the local inference worker.
public struct ImageGenerationCommand: Equatable {
    public let requestId: RequestId;
    public let model: String;
    public let prompt: String;
    public let settings: ImageGenerationSettings;

    public init(requestId: RequestId, model: String, prompt: String, settings: ImageGenerationSettings) {
        self.requestId = requestId;
        self.model = model;
        self.prompt = prompt;
        self.settings = settings;
    }

    internal static let wireFieldNames: Array<String> = ["request_id", "model", "prompt", "settings"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "request_id", value: self.requestId.wireValue());
        wireObject.appendEntry(key: "model", value: .string(self.model));
        wireObject.appendEntry(key: "prompt", value: .string(self.prompt));
        wireObject.appendEntry(key: "settings", value: self.settings.wireValue());
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ImageGenerationCommand {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedCommand = ImageGenerationCommand(
            requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
            model: try wireObject.decodeString(fieldName: "model"),
            prompt: try wireObject.decodeString(fieldName: "prompt"),
            settings: try ImageGenerationSettings.fromWireValue(try wireObject.requireObjectValue(fieldName: "settings")));
        try wireObject.rejectUnknownFields(allowedFieldNames: ImageGenerationCommand.wireFieldNames);
        return parsedCommand;
    }
}

/// Bounded image dimensions, diffusion controls, and deterministic seed.
public struct ImageGenerationSettings: Equatable {
    public let widthPixels: UInt32;
    public let heightPixels: UInt32;
    public let steps: UInt16;
    /// Classifier-free guidance scale represented in thousandths to keep JSON deterministic.
    public let guidanceThousandths: UInt32;
    public let seed: UInt64;

    public init(
        widthPixels: UInt32,
        heightPixels: UInt32,
        steps: UInt16,
        guidanceThousandths: UInt32,
        seed: UInt64
    ) {
        self.widthPixels = widthPixels;
        self.heightPixels = heightPixels;
        self.steps = steps;
        self.guidanceThousandths = guidanceThousandths;
        self.seed = seed;
    }

    internal static let wireFieldNames: Array<String> = [
        "width_pixels", "height_pixels", "steps", "guidance_thousandths", "seed",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "width_pixels", value: .unsignedInteger(UInt64(self.widthPixels)));
        wireObject.appendEntry(key: "height_pixels", value: .unsignedInteger(UInt64(self.heightPixels)));
        wireObject.appendEntry(key: "steps", value: .unsignedInteger(UInt64(self.steps)));
        wireObject.appendEntry(key: "guidance_thousandths", value: .unsignedInteger(UInt64(self.guidanceThousandths)));
        wireObject.appendEntry(key: "seed", value: .unsignedInteger(self.seed));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ImageGenerationSettings {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedSettings = ImageGenerationSettings(
            widthPixels: try wireObject.decodeUInt32(fieldName: "width_pixels"),
            heightPixels: try wireObject.decodeUInt32(fieldName: "height_pixels"),
            steps: try wireObject.decodeUInt16(fieldName: "steps"),
            guidanceThousandths: try wireObject.decodeUInt32(fieldName: "guidance_thousandths"),
            seed: try wireObject.decodeUInt64(fieldName: "seed"));
        try wireObject.rejectUnknownFields(allowedFieldNames: ImageGenerationSettings.wireFieldNames);
        return parsedSettings;
    }
}

/// One generated encoded image; JSON uses base64 rather than an integer array.
public struct GeneratedImage: Equatable {
    public let mimeType: String;
    public let encodedBytes: Array<UInt8>;

    public init(mimeType: String, encodedBytes: Array<UInt8>) {
        self.mimeType = mimeType;
        self.encodedBytes = encodedBytes;
    }

    internal static let wireFieldNames: Array<String> = ["mime_type", "encoded_bytes"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "mime_type", value: .string(self.mimeType));
        wireObject.appendEntry(key: "encoded_bytes", value: .string(Base64Bytes.encode(imageFileBytes: self.encodedBytes)));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> GeneratedImage {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedImage = GeneratedImage(
            mimeType: try wireObject.decodeString(fieldName: "mime_type"),
            encodedBytes: try Base64Bytes.decode(encodedText: try wireObject.decodeString(fieldName: "encoded_bytes")));
        try wireObject.rejectUnknownFields(allowedFieldNames: GeneratedImage.wireFieldNames);
        return parsedImage;
    }
}

/// Reproducibility and timing facts for one completed image.
public struct ImageGenerationResultMetadata: Equatable {
    public let widthPixels: UInt32;
    public let heightPixels: UInt32;
    public let steps: UInt16;
    public let guidanceThousandths: UInt32;
    public let seed: UInt64;
    public let elapsedMillis: UInt64;

    public init(
        widthPixels: UInt32,
        heightPixels: UInt32,
        steps: UInt16,
        guidanceThousandths: UInt32,
        seed: UInt64,
        elapsedMillis: UInt64
    ) {
        self.widthPixels = widthPixels;
        self.heightPixels = heightPixels;
        self.steps = steps;
        self.guidanceThousandths = guidanceThousandths;
        self.seed = seed;
        self.elapsedMillis = elapsedMillis;
    }

    internal static let wireFieldNames: Array<String> = [
        "width_pixels", "height_pixels", "steps", "guidance_thousandths", "seed", "elapsed_millis",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "width_pixels", value: .unsignedInteger(UInt64(self.widthPixels)));
        wireObject.appendEntry(key: "height_pixels", value: .unsignedInteger(UInt64(self.heightPixels)));
        wireObject.appendEntry(key: "steps", value: .unsignedInteger(UInt64(self.steps)));
        wireObject.appendEntry(key: "guidance_thousandths", value: .unsignedInteger(UInt64(self.guidanceThousandths)));
        wireObject.appendEntry(key: "seed", value: .unsignedInteger(self.seed));
        wireObject.appendEntry(key: "elapsed_millis", value: .unsignedInteger(self.elapsedMillis));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ImageGenerationResultMetadata {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedMetadata = ImageGenerationResultMetadata(
            widthPixels: try wireObject.decodeUInt32(fieldName: "width_pixels"),
            heightPixels: try wireObject.decodeUInt32(fieldName: "height_pixels"),
            steps: try wireObject.decodeUInt16(fieldName: "steps"),
            guidanceThousandths: try wireObject.decodeUInt32(fieldName: "guidance_thousandths"),
            seed: try wireObject.decodeUInt64(fieldName: "seed"),
            elapsedMillis: try wireObject.decodeUInt64(fieldName: "elapsed_millis"));
        try wireObject.rejectUnknownFields(allowedFieldNames: ImageGenerationResultMetadata.wireFieldNames);
        return parsedMetadata;
    }
}

/// Worker execution phase used for user-visible image progress.
public enum ImageGenerationPhase: Equatable, Sendable {
    case preparing;
    case encodingPrompt;
    case denoising;
    case decoding;
    case encodingImage;

    private static let expectedVariantNames: Array<String> = [
        "preparing", "encoding_prompt", "denoising", "decoding", "encoding_image",
    ];

    internal var wireName: String {
        switch (self) {
        case .preparing: return "preparing";
        case .encodingPrompt: return "encoding_prompt";
        case .denoising: return "denoising";
        case .decoding: return "decoding";
        case .encodingImage: return "encoding_image";
        }
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.wireName);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ImageGenerationPhase {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "preparing": return .preparing;
        case "encoding_prompt": return .encodingPrompt;
        case "denoising": return .denoising;
        case "decoding": return .decoding;
        case "encoding_image": return .encodingImage;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(ImageGenerationPhase.expectedVariantNames))");
        }
    }
}

/// A request-scoped image failure that leaves the worker protocol responsive.
/// Wire shape is serde's externally tagged enum: unit variants serialize as
/// plain strings and struct variants as single-entry objects.
public enum ImageGenerationFailureReason: Equatable {
    /// The command failed worker-side semantic validation before model execution.
    case invalidRequest(reason: String);
    /// The loaded artifact exposes no image-generation surface.
    case modelDoesNotSupportImageGeneration;
    /// A different image request already owns the worker's bounded capacity.
    case engineBusy;
    /// Prompt encoding or latent decoding failed inside the worker pipeline.
    case encodingFailed(reason: String);
    /// A fatal model-execution failure reported before the worker exits.
    case fatalExecution(reason: String);
    /// The request was cancelled before completion.
    case cancelled;

    private static let expectedVariantNames: Array<String> = [
        "invalid_request", "model_does_not_support_image_generation", "engine_busy",
        "encoding_failed", "fatal_execution", "cancelled",
    ];

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case let .invalidRequest(reason):
            return .object(ImageGenerationFailureReason.singleEntryWireObject(variantName: "invalid_request", payloadWireValue: ImageGenerationFailureReason.reasonObjectWireValue(reason)));
        case .modelDoesNotSupportImageGeneration:
            return .string("model_does_not_support_image_generation");
        case .engineBusy:
            return .string("engine_busy");
        case let .encodingFailed(reason):
            return .object(ImageGenerationFailureReason.singleEntryWireObject(variantName: "encoding_failed", payloadWireValue: ImageGenerationFailureReason.reasonObjectWireValue(reason)));
        case let .fatalExecution(reason):
            return .object(ImageGenerationFailureReason.singleEntryWireObject(variantName: "fatal_execution", payloadWireValue: ImageGenerationFailureReason.reasonObjectWireValue(reason)));
        case .cancelled:
            return .string("cancelled");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ImageGenerationFailureReason {
        switch wireValue {
        case let .string(variantName):
            return try ImageGenerationFailureReason.unitVariant(variantName: variantName);
        case let .object(variantObject):
            return try ImageGenerationFailureReason.structVariant(variantObject: variantObject);
        default:
            throw JsonWireProblem.invalidType(expectedTypeName: "enum ImageGenerationFailureReason", found: wireValue.foundDescription);
        }
    }

    private static func unitVariant(variantName: String) throws -> ImageGenerationFailureReason {
        switch variantName {
        case "model_does_not_support_image_generation": return .modelDoesNotSupportImageGeneration;
        case "engine_busy": return .engineBusy;
        case "cancelled": return .cancelled;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(variantName)`, expected one of \(JsonWireProblem.formattedFieldList(ImageGenerationFailureReason.expectedVariantNames))");
        }
    }

    private static func structVariant(variantObject: JsonWireObject) throws -> ImageGenerationFailureReason {
        guard variantObject.entries.count == 1, let singleEntry = variantObject.entries.first else {
            throw JsonWireProblem.malformedDocument(problem: "expected map with a single entry");
        }
        let variantName = singleEntry.key;
        switch variantName {
        case "invalid_request":
            return .invalidRequest(reason: try ImageGenerationFailureReason.decodeReasonPayload(singleEntry.value));
        case "encoding_failed":
            return .encodingFailed(reason: try ImageGenerationFailureReason.decodeReasonPayload(singleEntry.value));
        case "fatal_execution":
            return .fatalExecution(reason: try ImageGenerationFailureReason.decodeReasonPayload(singleEntry.value));
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(variantName)`, expected one of \(JsonWireProblem.formattedFieldList(ImageGenerationFailureReason.expectedVariantNames))");
        }
    }

    private static func decodeReasonPayload(_ payloadWireValue: JsonWireValue) throws -> String {
        let payloadObject = try JsonWireValue.extractObject(payloadWireValue);
        let reasonText = try payloadObject.decodeString(fieldName: "reason");
        try payloadObject.rejectUnknownFields(allowedFieldNames: ["reason"]);
        return reasonText;
    }

    private static func reasonObjectWireValue(_ reasonText: String) -> JsonWireValue {
        var payloadObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        payloadObject.appendEntry(key: "reason", value: .string(reasonText));
        return .object(payloadObject);
    }

    private static func singleEntryWireObject(variantName: String, payloadWireValue: JsonWireValue) -> JsonWireObject {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: variantName, value: payloadWireValue);
        return wireObject;
    }
}
