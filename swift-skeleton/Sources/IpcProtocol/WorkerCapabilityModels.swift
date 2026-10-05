import Foundation;

/// Image limits advertised by one loaded worker model.
public struct ImageGenerationCapabilities: Equatable {
    public let minimumWidthPixels: UInt32;
    public let maximumWidthPixels: UInt32;
    public let minimumHeightPixels: UInt32;
    public let maximumHeightPixels: UInt32;
    public let dimensionMultiplePixels: UInt32;
    public let maximumSteps: UInt16;
    public let maximumGuidanceThousandths: UInt32;
    public let outputMimeTypes: Array<String>;

    public init(
        minimumWidthPixels: UInt32,
        maximumWidthPixels: UInt32,
        minimumHeightPixels: UInt32,
        maximumHeightPixels: UInt32,
        dimensionMultiplePixels: UInt32,
        maximumSteps: UInt16,
        maximumGuidanceThousandths: UInt32,
        outputMimeTypes: Array<String>
    ) {
        self.minimumWidthPixels = minimumWidthPixels;
        self.maximumWidthPixels = maximumWidthPixels;
        self.minimumHeightPixels = minimumHeightPixels;
        self.maximumHeightPixels = maximumHeightPixels;
        self.dimensionMultiplePixels = dimensionMultiplePixels;
        self.maximumSteps = maximumSteps;
        self.maximumGuidanceThousandths = maximumGuidanceThousandths;
        self.outputMimeTypes = outputMimeTypes;
    }

    internal static let wireFieldNames: Array<String> = [
        "minimum_width_pixels", "maximum_width_pixels", "minimum_height_pixels", "maximum_height_pixels",
        "dimension_multiple_pixels", "maximum_steps", "maximum_guidance_thousandths", "output_mime_types",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "minimum_width_pixels", value: .unsignedInteger(UInt64(self.minimumWidthPixels)));
        wireObject.appendEntry(key: "maximum_width_pixels", value: .unsignedInteger(UInt64(self.maximumWidthPixels)));
        wireObject.appendEntry(key: "minimum_height_pixels", value: .unsignedInteger(UInt64(self.minimumHeightPixels)));
        wireObject.appendEntry(key: "maximum_height_pixels", value: .unsignedInteger(UInt64(self.maximumHeightPixels)));
        wireObject.appendEntry(key: "dimension_multiple_pixels", value: .unsignedInteger(UInt64(self.dimensionMultiplePixels)));
        wireObject.appendEntry(key: "maximum_steps", value: .unsignedInteger(UInt64(self.maximumSteps)));
        wireObject.appendEntry(key: "maximum_guidance_thousandths", value: .unsignedInteger(UInt64(self.maximumGuidanceThousandths)));
        wireObject.appendEntry(key: "output_mime_types", value: JsonWireValue.mappedArray(self.outputMimeTypes, mappedWireValue: { (mimeType: String) -> JsonWireValue in return .string(mimeType); }));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ImageGenerationCapabilities {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedCapabilities = ImageGenerationCapabilities(
            minimumWidthPixels: try wireObject.decodeUInt32(fieldName: "minimum_width_pixels"),
            maximumWidthPixels: try wireObject.decodeUInt32(fieldName: "maximum_width_pixels"),
            minimumHeightPixels: try wireObject.decodeUInt32(fieldName: "minimum_height_pixels"),
            maximumHeightPixels: try wireObject.decodeUInt32(fieldName: "maximum_height_pixels"),
            dimensionMultiplePixels: try wireObject.decodeUInt32(fieldName: "dimension_multiple_pixels"),
            maximumSteps: try wireObject.decodeUInt16(fieldName: "maximum_steps"),
            maximumGuidanceThousandths: try wireObject.decodeUInt32(fieldName: "maximum_guidance_thousandths"),
            outputMimeTypes: try wireObject.decodeArray(fieldName: "output_mime_types", mappedElement: { (elementWireValue: JsonWireValue) throws -> String in
                return try JsonWireValue.extractString(elementWireValue);
            }));
        try wireObject.rejectUnknownFields(allowedFieldNames: ImageGenerationCapabilities.wireFieldNames);
        return parsedCapabilities;
    }
}

/// Embedding inference capability advertised by one loaded worker model.
public struct WorkerEmbeddingCapabilities: Equatable {
    /// Native vector width produced by the loaded artifact.
    public let vectorWidth: UInt32;
    /// Maximum accepted prompt tokens per embedding input.
    public let maxInputTokens: UInt32;

    public init(vectorWidth: UInt32, maxInputTokens: UInt32) {
        self.vectorWidth = vectorWidth;
        self.maxInputTokens = maxInputTokens;
    }

    internal static let wireFieldNames: Array<String> = ["vector_width", "max_input_tokens"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "vector_width", value: .unsignedInteger(UInt64(self.vectorWidth)));
        wireObject.appendEntry(key: "max_input_tokens", value: .unsignedInteger(UInt64(self.maxInputTokens)));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerEmbeddingCapabilities {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedCapabilities = WorkerEmbeddingCapabilities(
            vectorWidth: try wireObject.decodeUInt32(fieldName: "vector_width"),
            maxInputTokens: try wireObject.decodeUInt32(fieldName: "max_input_tokens"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerEmbeddingCapabilities.wireFieldNames);
        return parsedCapabilities;
    }
}

/// Independently represents the chat, image, and embedding surfaces exposed by
/// one model. Plain serde options: absent-or-null decodes to nil and nil
/// encodes as an explicit JSON null.
public struct WorkerModelCapabilities: Equatable {
    public let chat: ChatModelCapabilities?;
    public let imageGeneration: ImageGenerationCapabilities?;
    public let embeddings: WorkerEmbeddingCapabilities?;

    public init(
        chat: ChatModelCapabilities?,
        imageGeneration: ImageGenerationCapabilities?,
        embeddings: WorkerEmbeddingCapabilities?
    ) {
        self.chat = chat;
        self.imageGeneration = imageGeneration;
        self.embeddings = embeddings;
    }

    /// Rust `From<ChatModelCapabilities>` counterpart for chat-only models.
    public static func from(chatCapabilities: ChatModelCapabilities) -> WorkerModelCapabilities {
        return WorkerModelCapabilities(chat: chatCapabilities, imageGeneration: nil, embeddings: nil);
    }

    public static func embeddings(embedding: WorkerEmbeddingCapabilities) -> WorkerModelCapabilities {
        return WorkerModelCapabilities(chat: nil, imageGeneration: nil, embeddings: embedding);
    }

    public static func imageGeneration(imageGeneration: ImageGenerationCapabilities) -> WorkerModelCapabilities {
        return WorkerModelCapabilities(chat: nil, imageGeneration: imageGeneration, embeddings: nil);
    }

    public static func chatAndImage(chat: ChatModelCapabilities, imageGeneration: ImageGenerationCapabilities) -> WorkerModelCapabilities {
        return WorkerModelCapabilities(chat: chat, imageGeneration: imageGeneration, embeddings: nil);
    }

    internal static let wireFieldNames: Array<String> = ["chat", "image_generation", "embeddings"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "chat", value: WorkerModelCapabilities.optionalWireValue(self.chat, mappedWireValue: { (chatCapabilities: ChatModelCapabilities) -> JsonWireValue in
            return chatCapabilities.wireValue();
        }));
        wireObject.appendEntry(key: "image_generation", value: WorkerModelCapabilities.optionalWireValue(self.imageGeneration, mappedWireValue: { (imageCapabilities: ImageGenerationCapabilities) -> JsonWireValue in
            return imageCapabilities.wireValue();
        }));
        wireObject.appendEntry(key: "embeddings", value: WorkerModelCapabilities.optionalWireValue(self.embeddings, mappedWireValue: { (embeddingCapabilities: WorkerEmbeddingCapabilities) -> JsonWireValue in
            return embeddingCapabilities.wireValue();
        }));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerModelCapabilities {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedCapabilities = WorkerModelCapabilities(
            chat: try WorkerModelCapabilities.decodeOptionalMapped(wireObject, fieldName: "chat", { (chatWireValue: JsonWireValue) throws -> ChatModelCapabilities in
                return try ChatModelCapabilities.fromWireValue(chatWireValue);
            }),
            imageGeneration: try WorkerModelCapabilities.decodeOptionalMapped(wireObject, fieldName: "image_generation", { (imageWireValue: JsonWireValue) throws -> ImageGenerationCapabilities in
                return try ImageGenerationCapabilities.fromWireValue(imageWireValue);
            }),
            embeddings: try WorkerModelCapabilities.decodeOptionalMapped(wireObject, fieldName: "embeddings", { (embeddingWireValue: JsonWireValue) throws -> WorkerEmbeddingCapabilities in
                return try WorkerEmbeddingCapabilities.fromWireValue(embeddingWireValue);
            }));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerModelCapabilities.wireFieldNames);
        return parsedCapabilities;
    }

    private static func optionalWireValue<T>(_ optionalValue: T?, mappedWireValue: (T) -> JsonWireValue) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return mappedWireValue(unwrappedValue);
    }

    private static func decodeOptionalMapped<T>(_ wireObject: JsonWireObject, fieldName propertyName: String, _ mappedValue: (JsonWireValue) throws -> T) throws -> T? {
        let fieldValue = try wireObject.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try mappedValue(fieldValue);
    }
}
