import Foundation;

/// Complete autoregressive execution policy for one canonical requestable model.
public struct WorkerAutoregressiveModelConfiguration: Equatable, Sendable {
    public let modelId: String;
    /// Effective prompt-plus-output context capability.
    public let maximumContextTokens: UInt32;
    /// Independent output capability; request defaults remain supervisor-owned.
    public let maximumOutputTokens: UInt32;
    public let chunking: WorkerChunkingConfiguration;

    public init(
        modelId: String,
        maximumContextTokens: UInt32,
        maximumOutputTokens: UInt32,
        chunking: WorkerChunkingConfiguration
    ) {
        self.modelId = modelId;
        self.maximumContextTokens = maximumContextTokens;
        self.maximumOutputTokens = maximumOutputTokens;
        self.chunking = chunking;
    }

    internal static let wireFieldNames: Array<String> = [
        "model_id", "maximum_context_tokens", "maximum_output_tokens", "chunking",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId));
        wireObject.appendEntry(key: "maximum_context_tokens", value: .unsignedInteger(UInt64(self.maximumContextTokens)));
        wireObject.appendEntry(key: "maximum_output_tokens", value: .unsignedInteger(UInt64(self.maximumOutputTokens)));
        wireObject.appendEntry(key: "chunking", value: self.chunking.wireValue());
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerAutoregressiveModelConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedConfiguration = WorkerAutoregressiveModelConfiguration(
            modelId: try wireObject.decodeString(fieldName: "model_id"),
            maximumContextTokens: try wireObject.decodeUInt32(fieldName: "maximum_context_tokens"),
            maximumOutputTokens: try wireObject.decodeUInt32(fieldName: "maximum_output_tokens"),
            chunking: try WorkerChunkingConfiguration.fromWireValue(try wireObject.requireObjectValue(fieldName: "chunking")));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerAutoregressiveModelConfiguration.wireFieldNames);
        return parsedConfiguration;
    }
}

/// Path-free autoregressive policy acknowledged after model binding.
public struct WorkerLoadedAutoregressiveModelRuntimeConfiguration: Equatable, Sendable {
    public let modelId: String;
    /// Effective prompt-plus-output context capability.
    public let maximumContextTokens: UInt32;
    /// Independent output capability, not the configured request default.
    public let maximumOutputTokens: UInt32;
    public let chunking: WorkerChunkingConfiguration;

    public init(
        modelId: String,
        maximumContextTokens: UInt32,
        maximumOutputTokens: UInt32,
        chunking: WorkerChunkingConfiguration
    ) {
        self.modelId = modelId;
        self.maximumContextTokens = maximumContextTokens;
        self.maximumOutputTokens = maximumOutputTokens;
        self.chunking = chunking;
    }

    internal static let wireFieldNames: Array<String> = WorkerAutoregressiveModelConfiguration.wireFieldNames;

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId));
        wireObject.appendEntry(key: "maximum_context_tokens", value: .unsignedInteger(UInt64(self.maximumContextTokens)));
        wireObject.appendEntry(key: "maximum_output_tokens", value: .unsignedInteger(UInt64(self.maximumOutputTokens)));
        wireObject.appendEntry(key: "chunking", value: self.chunking.wireValue());
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerLoadedAutoregressiveModelRuntimeConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedConfiguration = WorkerLoadedAutoregressiveModelRuntimeConfiguration(
            modelId: try wireObject.decodeString(fieldName: "model_id"),
            maximumContextTokens: try wireObject.decodeUInt32(fieldName: "maximum_context_tokens"),
            maximumOutputTokens: try wireObject.decodeUInt32(fieldName: "maximum_output_tokens"),
            chunking: try WorkerChunkingConfiguration.fromWireValue(try wireObject.requireObjectValue(fieldName: "chunking")));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerLoadedAutoregressiveModelRuntimeConfiguration.wireFieldNames);
        return parsedConfiguration;
    }
}

/// Typed image profile identifier carried without autoregressive placeholders.
public enum WorkerImageGenerationModelFamily: Equatable, Sendable {
    case flux2Klein;
    case qwenImage21;

    private static let expectedVariantNames: Array<String> = ["flux_2_klein", "qwen_image_2_1"];

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case .flux2Klein: return .string("flux_2_klein");
        case .qwenImage21: return .string("qwen_image_2_1");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerImageGenerationModelFamily {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "flux_2_klein": return .flux2Klein;
        case "qwen_image_2_1": return .qwenImage21;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(WorkerImageGenerationModelFamily.expectedVariantNames))");
        }
    }
}

/// Typed embedding profile identifier carried without autoregressive placeholders.
public enum WorkerEmbeddingModelFamily: Equatable, Sendable {
    case modernBert;

    private static let expectedVariantNames: Array<String> = ["modern_bert"];

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case .modernBert: return .string("modern_bert");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerEmbeddingModelFamily {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "modern_bert": return .modernBert;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(WorkerEmbeddingModelFamily.expectedVariantNames))");
        }
    }
}

/// Exact embedding artifact identity required by the selected profile.
public struct WorkerEmbeddingModelConfiguration: Equatable, Sendable {
    public let modelId: String;
    public let modelFamily: WorkerEmbeddingModelFamily;
    public let artifactRevision: String;
    /// Native vector width produced by the loaded artifact.
    public let vectorWidth: UInt32;
    /// Maximum accepted prompt tokens per embedding input.
    public let maximumInputTokens: UInt32;

    public init(
        modelId: String,
        modelFamily: WorkerEmbeddingModelFamily,
        artifactRevision: String,
        vectorWidth: UInt32,
        maximumInputTokens: UInt32
    ) {
        self.modelId = modelId;
        self.modelFamily = modelFamily;
        self.artifactRevision = artifactRevision;
        self.vectorWidth = vectorWidth;
        self.maximumInputTokens = maximumInputTokens;
    }

    internal static let wireFieldNames: Array<String> = [
        "model_id", "model_family", "artifact_revision", "vector_width", "maximum_input_tokens",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId));
        wireObject.appendEntry(key: "model_family", value: self.modelFamily.wireValue());
        wireObject.appendEntry(key: "artifact_revision", value: .string(self.artifactRevision));
        wireObject.appendEntry(key: "vector_width", value: .unsignedInteger(UInt64(self.vectorWidth)));
        wireObject.appendEntry(key: "maximum_input_tokens", value: .unsignedInteger(UInt64(self.maximumInputTokens)));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerEmbeddingModelConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedConfiguration = WorkerEmbeddingModelConfiguration(
            modelId: try wireObject.decodeString(fieldName: "model_id"),
            modelFamily: try WorkerEmbeddingModelFamily.fromWireValue(try wireObject.requireObjectValue(fieldName: "model_family")),
            artifactRevision: try wireObject.decodeString(fieldName: "artifact_revision"),
            vectorWidth: try wireObject.decodeUInt32(fieldName: "vector_width"),
            maximumInputTokens: try wireObject.decodeUInt32(fieldName: "maximum_input_tokens"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerEmbeddingModelConfiguration.wireFieldNames);
        return parsedConfiguration;
    }
}

/// Exact FLUX artifact identity required by the selected image profile.
public struct WorkerFlux2KleinModelConfiguration: Equatable, Sendable {
    public let modelId: String;
    public let modelFamily: WorkerImageGenerationModelFamily;
    public let artifactRevision: String;

    public init(modelId: String, modelFamily: WorkerImageGenerationModelFamily, artifactRevision: String) {
        self.modelId = modelId;
        self.modelFamily = modelFamily;
        self.artifactRevision = artifactRevision;
    }

    internal static let wireFieldNames: Array<String> = ["model_id", "model_family", "artifact_revision"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId));
        wireObject.appendEntry(key: "model_family", value: self.modelFamily.wireValue());
        wireObject.appendEntry(key: "artifact_revision", value: .string(self.artifactRevision));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerFlux2KleinModelConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedConfiguration = WorkerFlux2KleinModelConfiguration(
            modelId: try wireObject.decodeString(fieldName: "model_id"),
            modelFamily: try WorkerImageGenerationModelFamily.fromWireValue(try wireObject.requireObjectValue(fieldName: "model_family")),
            artifactRevision: try wireObject.decodeString(fieldName: "artifact_revision"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerFlux2KleinModelConfiguration.wireFieldNames);
        return parsedConfiguration;
    }
}

/// Exact Qwen-Image-2.1 artifact identity required by the selected image profile.
public struct WorkerQwenImage21ModelConfiguration: Equatable, Sendable {
    public let modelId: String;
    public let modelFamily: WorkerImageGenerationModelFamily;
    public let artifactRevision: String;

    public init(modelId: String, modelFamily: WorkerImageGenerationModelFamily, artifactRevision: String) {
        self.modelId = modelId;
        self.modelFamily = modelFamily;
        self.artifactRevision = artifactRevision;
    }

    internal static let wireFieldNames: Array<String> = ["model_id", "model_family", "artifact_revision"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId));
        wireObject.appendEntry(key: "model_family", value: self.modelFamily.wireValue());
        wireObject.appendEntry(key: "artifact_revision", value: .string(self.artifactRevision));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerQwenImage21ModelConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedConfiguration = WorkerQwenImage21ModelConfiguration(
            modelId: try wireObject.decodeString(fieldName: "model_id"),
            modelFamily: try WorkerImageGenerationModelFamily.fromWireValue(try wireObject.requireObjectValue(fieldName: "model_family")),
            artifactRevision: try wireObject.decodeString(fieldName: "artifact_revision"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerQwenImage21ModelConfiguration.wireFieldNames);
        return parsedConfiguration;
    }
}

/// Complete effective execution policy for one canonical requestable model.
/// Wire shape is adjacently tagged: `kind` names the variant, `configuration`
/// carries its payload.
public enum WorkerModelConfiguration: Equatable, Sendable {

    /// Serializes the exact wire form this policy sends to the worker; the
    /// resolved-configuration generation digests these bytes.
    public func serializedJsonBytes() throws -> Data {
        var wireWriter = JsonWireWriter();
        try wireWriter.appendValue(self.wireValue());
        return wireWriter.serializedUtf8Bytes;
    }

    case autoregressive(WorkerAutoregressiveModelConfiguration);
    case flux2Klein(WorkerFlux2KleinModelConfiguration);
    case qwenImage21(WorkerQwenImage21ModelConfiguration);
    case embeddings(WorkerEmbeddingModelConfiguration);

    private static let expectedVariantNames: Array<String> = [
        "autoregressive", "flux_2_klein", "qwen_image_2_1", "embeddings",
    ];

    /// Returns the canonical requestable model identity for either execution family.
    public func modelId() -> String {
        switch (self) {
        case let .autoregressive(configuration): return configuration.modelId;
        case let .flux2Klein(configuration): return configuration.modelId;
        case let .qwenImage21(configuration): return configuration.modelId;
        case let .embeddings(configuration): return configuration.modelId;
        }
    }

    /// Returns chat policy only when this model owns autoregressive execution.
    public func autoregressive() -> WorkerAutoregressiveModelConfiguration? {
        switch (self) {
        case let .autoregressive(configuration): return configuration;
        case .flux2Klein, .qwenImage21, .embeddings: return nil;
        }
    }

    /// Removes local auxiliary paths while retaining the exact policy bound by the worker.
    public func runtimeConfiguration() -> WorkerLoadedModelRuntimeConfiguration {
        switch (self) {
        case let .autoregressive(configuration):
            return .autoregressive(WorkerLoadedAutoregressiveModelRuntimeConfiguration(
                modelId: configuration.modelId,
                maximumContextTokens: configuration.maximumContextTokens,
                maximumOutputTokens: configuration.maximumOutputTokens,
                chunking: configuration.chunking));
        case let .flux2Klein(configuration):
            return .flux2Klein(configuration);
        case let .qwenImage21(configuration):
            return .qwenImage21(configuration);
        case let .embeddings(configuration):
            return .embeddings(configuration);
        }
    }

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case let .autoregressive(configuration):
            return WorkerModelConfiguration.taggedWireObject(variantName: "autoregressive", payloadWireValue: configuration.wireValue());
        case let .flux2Klein(configuration):
            return WorkerModelConfiguration.taggedWireObject(variantName: "flux_2_klein", payloadWireValue: configuration.wireValue());
        case let .qwenImage21(configuration):
            return WorkerModelConfiguration.taggedWireObject(variantName: "qwen_image_2_1", payloadWireValue: configuration.wireValue());
        case let .embeddings(configuration):
            return WorkerModelConfiguration.taggedWireObject(variantName: "embeddings", payloadWireValue: configuration.wireValue());
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerModelConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: WorkerModelConfiguration.expectedVariantNames) {
        case "autoregressive":
            let parsedConfiguration = WorkerModelConfiguration.autoregressive(
                try WorkerAutoregressiveModelConfiguration.fromWireValue(.object(try wireObject.decodeObject(fieldName: "configuration"))));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["configuration"]);
            return parsedConfiguration;
        case "flux_2_klein":
            let parsedConfiguration = WorkerModelConfiguration.flux2Klein(
                try WorkerFlux2KleinModelConfiguration.fromWireValue(.object(try wireObject.decodeObject(fieldName: "configuration"))));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["configuration"]);
            return parsedConfiguration;
        case "qwen_image_2_1":
            let parsedConfiguration = WorkerModelConfiguration.qwenImage21(
                try WorkerQwenImage21ModelConfiguration.fromWireValue(.object(try wireObject.decodeObject(fieldName: "configuration"))));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["configuration"]);
            return parsedConfiguration;
        default:
            let parsedConfiguration = WorkerModelConfiguration.embeddings(
                try WorkerEmbeddingModelConfiguration.fromWireValue(.object(try wireObject.decodeObject(fieldName: "configuration"))));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["configuration"]);
            return parsedConfiguration;
        }
    }

    private static func taggedWireObject(variantName: String, payloadWireValue: JsonWireValue) -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "kind", value: .string(variantName));
        wireObject.appendEntry(key: "configuration", value: payloadWireValue);
        return .object(wireObject);
    }
}

/// Path-free loaded-model policy acknowledged to the supervisor and local status API.
/// Wire shape is adjacently tagged: `kind` names the variant, `configuration`
/// carries its payload.
public enum WorkerLoadedModelRuntimeConfiguration: Equatable, Sendable {
    case autoregressive(WorkerLoadedAutoregressiveModelRuntimeConfiguration);
    case flux2Klein(WorkerFlux2KleinModelConfiguration);
    case qwenImage21(WorkerQwenImage21ModelConfiguration);
    case embeddings(WorkerEmbeddingModelConfiguration);

    private static let expectedVariantNames: Array<String> = [
        "autoregressive", "flux_2_klein", "qwen_image_2_1", "embeddings",
    ];

    /// Returns the canonical requestable model identity for either execution family.
    public func modelId() -> String {
        switch (self) {
        case let .autoregressive(configuration): return configuration.modelId;
        case let .flux2Klein(configuration): return configuration.modelId;
        case let .qwenImage21(configuration): return configuration.modelId;
        case let .embeddings(configuration): return configuration.modelId;
        }
    }

    /// Returns chat runtime policy only when this model owns autoregressive execution.
    public func autoregressive() -> WorkerLoadedAutoregressiveModelRuntimeConfiguration? {
        switch (self) {
        case let .autoregressive(configuration): return configuration;
        case .flux2Klein, .qwenImage21, .embeddings: return nil;
        }
    }

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case let .autoregressive(configuration):
            return WorkerLoadedModelRuntimeConfiguration.taggedWireObject(variantName: "autoregressive", payloadWireValue: configuration.wireValue());
        case let .flux2Klein(configuration):
            return WorkerLoadedModelRuntimeConfiguration.taggedWireObject(variantName: "flux_2_klein", payloadWireValue: configuration.wireValue());
        case let .qwenImage21(configuration):
            return WorkerLoadedModelRuntimeConfiguration.taggedWireObject(variantName: "qwen_image_2_1", payloadWireValue: configuration.wireValue());
        case let .embeddings(configuration):
            return WorkerLoadedModelRuntimeConfiguration.taggedWireObject(variantName: "embeddings", payloadWireValue: configuration.wireValue());
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerLoadedModelRuntimeConfiguration {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: WorkerLoadedModelRuntimeConfiguration.expectedVariantNames) {
        case "autoregressive":
            let parsedConfiguration = WorkerLoadedModelRuntimeConfiguration.autoregressive(
                try WorkerLoadedAutoregressiveModelRuntimeConfiguration.fromWireValue(.object(try wireObject.decodeObject(fieldName: "configuration"))));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["configuration"]);
            return parsedConfiguration;
        case "flux_2_klein":
            let parsedConfiguration = WorkerLoadedModelRuntimeConfiguration.flux2Klein(
                try WorkerFlux2KleinModelConfiguration.fromWireValue(.object(try wireObject.decodeObject(fieldName: "configuration"))));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["configuration"]);
            return parsedConfiguration;
        case "qwen_image_2_1":
            let parsedConfiguration = WorkerLoadedModelRuntimeConfiguration.qwenImage21(
                try WorkerQwenImage21ModelConfiguration.fromWireValue(.object(try wireObject.decodeObject(fieldName: "configuration"))));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["configuration"]);
            return parsedConfiguration;
        default:
            let parsedConfiguration = WorkerLoadedModelRuntimeConfiguration.embeddings(
                try WorkerEmbeddingModelConfiguration.fromWireValue(.object(try wireObject.decodeObject(fieldName: "configuration"))));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["configuration"]);
            return parsedConfiguration;
        }
    }

    private static func taggedWireObject(variantName: String, payloadWireValue: JsonWireValue) -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "kind", value: .string(variantName));
        wireObject.appendEntry(key: "configuration", value: payloadWireValue);
        return .object(wireObject);
    }
}
