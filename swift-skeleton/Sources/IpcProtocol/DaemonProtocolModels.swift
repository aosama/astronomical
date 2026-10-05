import Foundation;

/// Handshake facts the daemon and its local CLI clients agree on at connect.
public enum DaemonProtocol {
    /// Protocol version the daemon and its local CLI clients negotiate on connect.
    public static let protocolVersion: UInt32 = 1;
    /// Application name the daemon reports during the handshake so a CLI client
    /// can confirm the socket belongs to Astronomical and not to a stale file.
    public static let applicationName: String = "Astronomical";
}

/// Coarse daemon availability reported for status requests.
public enum DaemonWorkerStatus: Equatable {
    /// The inference engine is still loading.
    case loading;
    /// The daemon can accept chat generation requests.
    case ready;
    /// The inference engine is absent or otherwise unavailable.
    case unavailable;

    private static let expectedVariantNames: Array<String> = ["loading", "ready", "unavailable"];

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case .loading: return .string("loading");
        case .ready: return .string("ready");
        case .unavailable: return .string("unavailable");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> DaemonWorkerStatus {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "loading": return .loading;
        case "ready": return .ready;
        case "unavailable": return .unavailable;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(DaemonWorkerStatus.expectedVariantNames))");
        }
    }
}

/// One model discovered on this Mac, any capability class.
public struct DaemonListedModel: Equatable {
    /// Requestable model id (leaf of the huggingface id).
    public let modelId: String;
    /// Model family label, e.g. `qwen`.
    public let family: String;
    /// Effective context window in tokens after policy clamping.
    /// `nil` for models that are not chat-capable (image, embeddings).
    public let contextWindow: UInt32?;
    /// True when this model can produce embeddings for `embed`.
    public let supportsEmbeddings: Bool;
    /// True when this model is currently resident in the worker.
    public let isResident: Bool;
    /// Artifact size in bytes; the CLI renders decimal SI gigabytes.
    public let sizeBytes: UInt64;

    public init(
        modelId: String,
        family: String,
        contextWindow: UInt32?,
        supportsEmbeddings: Bool,
        isResident: Bool,
        sizeBytes: UInt64
    ) {
        self.modelId = modelId;
        self.family = family;
        self.contextWindow = contextWindow;
        self.supportsEmbeddings = supportsEmbeddings;
        self.isResident = isResident;
        self.sizeBytes = sizeBytes;
    }

    internal static let wireFieldNames: Array<String> = [
        "model_id", "family", "context_window", "supports_embeddings", "is_resident", "size_bytes",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId));
        wireObject.appendEntry(key: "family", value: .string(self.family));
        wireObject.appendEntry(key: "context_window", value: DaemonListedModel.optionalUInt32WireValue(self.contextWindow));
        wireObject.appendEntry(key: "supports_embeddings", value: .boolean(self.supportsEmbeddings));
        wireObject.appendEntry(key: "is_resident", value: .boolean(self.isResident));
        wireObject.appendEntry(key: "size_bytes", value: .unsignedInteger(self.sizeBytes));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> DaemonListedModel {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedModel = DaemonListedModel(
            modelId: try wireObject.decodeString(fieldName: "model_id"),
            family: try wireObject.decodeString(fieldName: "family"),
            contextWindow: try wireObject.decodeOptionalUInt32(fieldName: "context_window"),
            supportsEmbeddings: try wireObject.decodeBool(fieldName: "supports_embeddings"),
            isResident: try wireObject.decodeBool(fieldName: "is_resident"),
            sizeBytes: try wireObject.decodeUInt64(fieldName: "size_bytes"));
        try wireObject.rejectUnknownFields(allowedFieldNames: DaemonListedModel.wireFieldNames);
        return parsedModel;
    }

    private static func optionalUInt32WireValue(_ optionalValue: UInt32?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }
}

/// One release download-catalog entry with its local readiness.
public struct DaemonCatalogEntry: Equatable {
    public let huggingfaceId: String;
    public let displayName: String;
    public let family: String;
    /// Approximate download size in bytes; the CLI renders decimal SI GB.
    public let approximateSizeBytes: UInt64;
    /// True when the entry is discovered or has a validated publication here.
    public let readyOnThisMac: Bool;
    /// Requestable model id once ready; `nil` while not downloaded.
    public let requestableModelId: String?;
    /// Active download job state for this entry, if one runs.
    public let downloadState: String?;
    public let contextWindow: UInt32?;
    public let supportsReasoning: Bool;
    public let supportsVision: Bool;
    public let supportsToolCalls: Bool;
    public let supportsImageGeneration: Bool;
    public let supportsEmbeddings: Bool;

    public init(
        huggingfaceId: String,
        displayName: String,
        family: String,
        approximateSizeBytes: UInt64,
        readyOnThisMac: Bool,
        requestableModelId: String?,
        downloadState: String?,
        contextWindow: UInt32?,
        supportsReasoning: Bool,
        supportsVision: Bool,
        supportsToolCalls: Bool,
        supportsImageGeneration: Bool,
        supportsEmbeddings: Bool
    ) {
        self.huggingfaceId = huggingfaceId;
        self.displayName = displayName;
        self.family = family;
        self.approximateSizeBytes = approximateSizeBytes;
        self.readyOnThisMac = readyOnThisMac;
        self.requestableModelId = requestableModelId;
        self.downloadState = downloadState;
        self.contextWindow = contextWindow;
        self.supportsReasoning = supportsReasoning;
        self.supportsVision = supportsVision;
        self.supportsToolCalls = supportsToolCalls;
        self.supportsImageGeneration = supportsImageGeneration;
        self.supportsEmbeddings = supportsEmbeddings;
    }

    internal static let wireFieldNames: Array<String> = [
        "huggingface_id", "display_name", "family", "approximate_size_bytes", "ready_on_this_mac",
        "requestable_model_id", "download_state", "context_window", "supports_reasoning",
        "supports_vision", "supports_tool_calls", "supports_image_generation", "supports_embeddings",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "huggingface_id", value: .string(self.huggingfaceId));
        wireObject.appendEntry(key: "display_name", value: .string(self.displayName));
        wireObject.appendEntry(key: "family", value: .string(self.family));
        wireObject.appendEntry(key: "approximate_size_bytes", value: .unsignedInteger(self.approximateSizeBytes));
        wireObject.appendEntry(key: "ready_on_this_mac", value: .boolean(self.readyOnThisMac));
        wireObject.appendEntry(key: "requestable_model_id", value: DaemonCatalogEntry.optionalStringWireValue(self.requestableModelId));
        wireObject.appendEntry(key: "download_state", value: DaemonCatalogEntry.optionalStringWireValue(self.downloadState));
        wireObject.appendEntry(key: "context_window", value: DaemonCatalogEntry.optionalUInt32WireValue(self.contextWindow));
        wireObject.appendEntry(key: "supports_reasoning", value: .boolean(self.supportsReasoning));
        wireObject.appendEntry(key: "supports_vision", value: .boolean(self.supportsVision));
        wireObject.appendEntry(key: "supports_tool_calls", value: .boolean(self.supportsToolCalls));
        wireObject.appendEntry(key: "supports_image_generation", value: .boolean(self.supportsImageGeneration));
        wireObject.appendEntry(key: "supports_embeddings", value: .boolean(self.supportsEmbeddings));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> DaemonCatalogEntry {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedEntry = DaemonCatalogEntry(
            huggingfaceId: try wireObject.decodeString(fieldName: "huggingface_id"),
            displayName: try wireObject.decodeString(fieldName: "display_name"),
            family: try wireObject.decodeString(fieldName: "family"),
            approximateSizeBytes: try wireObject.decodeUInt64(fieldName: "approximate_size_bytes"),
            readyOnThisMac: try wireObject.decodeBool(fieldName: "ready_on_this_mac"),
            requestableModelId: try wireObject.decodeOptionalString(fieldName: "requestable_model_id"),
            downloadState: try wireObject.decodeOptionalString(fieldName: "download_state"),
            contextWindow: try wireObject.decodeOptionalUInt32(fieldName: "context_window"),
            supportsReasoning: try wireObject.decodeBool(fieldName: "supports_reasoning"),
            supportsVision: try wireObject.decodeBool(fieldName: "supports_vision"),
            supportsToolCalls: try wireObject.decodeBool(fieldName: "supports_tool_calls"),
            supportsImageGeneration: try wireObject.decodeBool(fieldName: "supports_image_generation"),
            supportsEmbeddings: try wireObject.decodeBool(fieldName: "supports_embeddings"));
        try wireObject.rejectUnknownFields(allowedFieldNames: DaemonCatalogEntry.wireFieldNames);
        return parsedEntry;
    }

    private static func optionalStringWireValue(_ optionalValue: String?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }

    private static func optionalUInt32WireValue(_ optionalValue: UInt32?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }
}

/// The active library download job, if any.
public struct DaemonDownloadJob: Equatable {
    public let huggingfaceId: String;
    /// Durable job state, e.g. `downloading`, `verifying`, `publishing`.
    public let state: String;
    public let bytesCompleted: UInt64;
    public let bytesTotal: UInt64;
    /// Public error code when the job failed.
    public let error: String?;

    public init(huggingfaceId: String, state: String, bytesCompleted: UInt64, bytesTotal: UInt64, error: String?) {
        self.huggingfaceId = huggingfaceId;
        self.state = state;
        self.bytesCompleted = bytesCompleted;
        self.bytesTotal = bytesTotal;
        self.error = error;
    }

    internal static let wireFieldNames: Array<String> = [
        "huggingface_id", "state", "bytes_completed", "bytes_total", "error",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "huggingface_id", value: .string(self.huggingfaceId));
        wireObject.appendEntry(key: "state", value: .string(self.state));
        wireObject.appendEntry(key: "bytes_completed", value: .unsignedInteger(self.bytesCompleted));
        wireObject.appendEntry(key: "bytes_total", value: .unsignedInteger(self.bytesTotal));
        wireObject.appendEntry(key: "error", value: DaemonDownloadJob.optionalStringWireValue(self.error));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> DaemonDownloadJob {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedJob = DaemonDownloadJob(
            huggingfaceId: try wireObject.decodeString(fieldName: "huggingface_id"),
            state: try wireObject.decodeString(fieldName: "state"),
            bytesCompleted: try wireObject.decodeUInt64(fieldName: "bytes_completed"),
            bytesTotal: try wireObject.decodeUInt64(fieldName: "bytes_total"),
            error: try wireObject.decodeOptionalString(fieldName: "error"));
        try wireObject.rejectUnknownFields(allowedFieldNames: DaemonDownloadJob.wireFieldNames);
        return parsedJob;
    }

    private static func optionalStringWireValue(_ optionalValue: String?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }
}
