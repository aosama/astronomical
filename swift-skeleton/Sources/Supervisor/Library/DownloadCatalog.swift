import Foundation

/// Strict immutable contract for release-authored model metadata, migrating
/// apps/supervisor/src/library/download_catalog.rs: one validated catalog in
/// authored presentation order, parsed from the release-bundled document or
/// supplied by tests. Explicit metadata bounds keep release packaging
/// mistakes from consuming arbitrary laptop resources, and every catalog
/// entry must be a public Hugging Face artifact at an immutable revision.
public struct DownloadCatalog: Equatable, Sendable {

    public let schemaVersion: UInt32
    public let entries: Array<DownloadCatalogEntry>

    /// Parses and validates one complete catalog document.
    public static func parseJson(_ catalogJson: String) throws -> DownloadCatalog {
        guard catalogJson.utf8.count <= DownloadCatalog.MAXIMUM_DOWNLOAD_CATALOG_BYTES else {
            throw DownloadCatalogError.documentTooLarge
        }
        let catalogDocument: Any
        do {
            catalogDocument = try JSONSerialization.jsonObject(with: Data(catalogJson.utf8), options: [])
        } catch {
            throw DownloadCatalogError.parse
        }
        guard let catalogObject: [String: Any] = catalogDocument as? [String: Any] else {
            throw DownloadCatalogError.parse
        }
        guard let rawSchemaVersion: Any = catalogObject["schema_version"],
            let schemaVersion: UInt32 = rawSchemaVersion as? UInt32 else {
            throw DownloadCatalogError.parse
        }
        guard schemaVersion == DownloadCatalog.DOWNLOAD_CATALOG_SCHEMA_VERSION else {
            throw DownloadCatalogError.unsupportedSchemaVersion(schemaVersion: schemaVersion)
        }
        guard let rawEntries: Any = catalogObject["entries"],
            let entryDocuments: Array<Any> = rawEntries as? Array<Any> else {
            throw DownloadCatalogError.parse
        }
        guard entryDocuments.count <= DownloadCatalog.MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT else {
            throw DownloadCatalogError.tooManyEntries
        }

        var normalizedHuggingFaceIds: Set<String> = Set<String>()
        var entries: Array<DownloadCatalogEntry> = Array<DownloadCatalogEntry>()
        for (entryIndex, rawEntryDocument) in entryDocuments.enumerated() {
            guard let entryObject: [String: Any] = rawEntryDocument as? [String: Any] else {
                throw DownloadCatalogError.parse
            }
            let entryDocument: DownloadCatalogEntryDocument = try DownloadCatalogEntryDocument.decode(
                entryObject,
                entryIndex: entryIndex)
            guard DownloadCatalog.isValidHuggingFaceId(entryDocument.huggingfaceId) else {
                throw DownloadCatalogError.invalidHuggingFaceId(entryIndex: entryIndex)
            }
            guard normalizedHuggingFaceIds.insert(entryDocument.huggingfaceId.lowercased()).inserted else {
                throw DownloadCatalogError.duplicateHuggingFaceId
            }
            guard DownloadCatalog.isValidImmutableRevision(entryDocument.revision) else {
                throw DownloadCatalogError.invalidRevision(entryIndex: entryIndex)
            }
            guard DownloadCatalog.isValidTextValue(
                entryDocument.displayName,
                maximumBytes: DownloadCatalog.MAXIMUM_DISPLAY_NAME_BYTES) else {
                throw DownloadCatalogError.invalidDisplayName(entryIndex: entryIndex)
            }
            guard entryDocument.approximateSizeBytes >= 1,
                entryDocument.approximateSizeBytes <= DownloadCatalog.MAXIMUM_JAVASCRIPT_SAFE_INTEGER else {
                throw DownloadCatalogError.invalidApproximateSize(entryIndex: entryIndex)
            }
            guard entryDocument.isPublic else {
                throw DownloadCatalogError.modelNotPublic(entryIndex: entryIndex)
            }
            let description: String? = try DownloadCatalog.validatedOptionalText(
                entryDocument.description,
                maximumBytes: DownloadCatalog.MAXIMUM_DESCRIPTION_BYTES,
                entryIndex: entryIndex,
                invalidError: DownloadCatalogError.invalidDescription(entryIndex: entryIndex))
            let quantizationLabel: String? = try DownloadCatalog.validatedOptionalText(
                entryDocument.quantizationLabel,
                maximumBytes: DownloadCatalog.MAXIMUM_QUANTIZATION_LABEL_BYTES,
                entryIndex: entryIndex,
                invalidError: DownloadCatalogError.invalidQuantizationLabel(entryIndex: entryIndex))
            let architectureSummary: String? = try DownloadCatalog.validatedOptionalText(
                entryDocument.architectureSummary,
                maximumBytes: DownloadCatalog.MAXIMUM_ARCHITECTURE_SUMMARY_BYTES,
                entryIndex: entryIndex,
                invalidError: DownloadCatalogError.invalidArchitectureSummary(entryIndex: entryIndex))
            let upstreamLicense: String? = try DownloadCatalog.validatedOptionalText(
                entryDocument.upstreamLicense,
                maximumBytes: DownloadCatalog.MAXIMUM_UPSTREAM_LICENSE_BYTES,
                entryIndex: entryIndex,
                invalidError: DownloadCatalogError.invalidUpstreamLicense(entryIndex: entryIndex))
            let capabilities: DownloadCatalogCapabilities
            if let rawCapabilities: [String: Any] = entryDocument.capabilities {
                let decodedCapabilities: DownloadCatalogCapabilitiesDocument =
                    try DownloadCatalogCapabilitiesDocument.decode(rawCapabilities)
                guard decodedCapabilities.declaresAnyCapability else {
                    throw DownloadCatalogError.invalidCapabilities(entryIndex: entryIndex)
                }
                capabilities = DownloadCatalogCapabilities(
                    supportsReasoning: decodedCapabilities.supportsReasoning,
                    supportsVision: decodedCapabilities.supportsVision,
                    supportsToolCalls: decodedCapabilities.supportsToolCalls,
                    contextWindow: decodedCapabilities.contextWindow,
                    maxOutputTokens: decodedCapabilities.maxOutputTokens,
                    supportsImageGeneration: decodedCapabilities.supportsImageGeneration,
                    supportsEmbeddings: decodedCapabilities.supportsEmbeddings)
            } else {
                capabilities = DownloadCatalogCapabilities()
            }
            let downloadPathSelection: DownloadPathSelection
            do {
                downloadPathSelection = try DownloadPathSelection(includedPaths: entryDocument.includedPaths)
            } catch {
                throw DownloadCatalogError.invalidIncludedPaths(entryIndex: entryIndex)
            }
            entries.append(DownloadCatalogEntry(
                huggingfaceId: entryDocument.huggingfaceId,
                revision: entryDocument.revision,
                displayName: entryDocument.displayName,
                family: entryDocument.family,
                approximateSizeBytes: entryDocument.approximateSizeBytes,
                description: description,
                capabilities: capabilities,
                quantizationLabel: quantizationLabel,
                architectureSummary: architectureSummary,
                upstreamLicense: upstreamLicense,
                downloadPathSelection: downloadPathSelection))
        }
        return DownloadCatalog(schemaVersion: schemaVersion, entries: entries)
    }

    /// Validates the catalog embedded into the current daemon binary.
    public static func loadBundled() throws -> DownloadCatalog {
        guard let bundledCatalogUrl: URL = Bundle.module.url(
            forResource: "download_catalog",
            withExtension: "json") else {
            throw DownloadCatalogError.bundledCatalogMissing
        }
        let bundledCatalogJson: String
        do {
            bundledCatalogJson = try String(contentsOf: bundledCatalogUrl, encoding: .utf8)
        } catch {
            throw DownloadCatalogError.bundledCatalogMissing
        }
        return try DownloadCatalog.parseJson(bundledCatalogJson)
    }

    public var entryCount: Int {
        return self.entries.count
    }

    private static let DOWNLOAD_CATALOG_SCHEMA_VERSION: UInt32 = 2
    private static let MAXIMUM_DOWNLOAD_CATALOG_BYTES: Int = 1_000_000
    private static let MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT: Int = 1_024
    private static let MAXIMUM_HUGGING_FACE_COMPONENT_LENGTH: Int = 96
    private static let MAXIMUM_DISPLAY_NAME_BYTES: Int = 256
    private static let MAXIMUM_DESCRIPTION_BYTES: Int = 512
    private static let MAXIMUM_QUANTIZATION_LABEL_BYTES: Int = 64
    private static let MAXIMUM_ARCHITECTURE_SUMMARY_BYTES: Int = 256
    private static let MAXIMUM_UPSTREAM_LICENSE_BYTES: Int = 128
    private static let MAXIMUM_JAVASCRIPT_SAFE_INTEGER: UInt64 = 9_007_199_254_740_991

    private static func validatedOptionalText(
        _ candidateText: String?,
        maximumBytes: Int,
        entryIndex: Int,
        invalidError: DownloadCatalogError
    ) throws -> String? {
        guard let candidateText: String = candidateText,
            !candidateText.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty else {
            return nil
        }
        guard DownloadCatalog.isValidTextValue(candidateText, maximumBytes: maximumBytes) else {
            throw invalidError
        }
        return candidateText
    }

    private static func isValidTextValue(_ textValue: String, maximumBytes: Int) -> Bool {
        return textValue.utf8.count <= maximumBytes
            && !textValue.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty
            && !textValue.contains(where: { (textCharacter: Character) -> Bool in
                return textCharacter.isControlCharacter
            })
    }

    public static func isValidHuggingFaceId(_ huggingFaceId: String) -> Bool {
        let identityComponents: Array<Substring> = huggingFaceId.split(separator: "/", omittingEmptySubsequences: false)
        guard identityComponents.count == 2 else {
            return false
        }
        let organization: Substring = identityComponents[0]
        let modelName: Substring = identityComponents[1]
        return DownloadCatalog.isValidHuggingFaceComponent(String(organization))
            && DownloadCatalog.isValidHuggingFaceComponent(String(modelName))
            && !modelName.hasSuffix(".git")
    }

    public static func isValidImmutableRevision(_ revision: String) -> Bool {
        return revision.count == 40
            && revision.allSatisfy({ (revisionCharacter: Character) -> Bool in
                return revisionCharacter.isHexDigit && (revisionCharacter.isNumber || revisionCharacter.isLowercase)
            })
    }

    private static func isValidHuggingFaceComponent(_ component: String) -> Bool {
        guard !component.isEmpty,
            component.utf8.count <= DownloadCatalog.MAXIMUM_HUGGING_FACE_COMPONENT_LENGTH,
            !(component.hasPrefix("-") || component.hasPrefix(".")),
            !(component.hasSuffix("-") || component.hasSuffix(".")),
            !component.contains("--"),
            !component.contains("..") else {
            return false
        }
        return component.allSatisfy({ (componentCharacter: Character) -> Bool in
            return componentCharacter.isASCII
                && (componentCharacter.isLetter || componentCharacter.isNumber
                    || componentCharacter == "-" || componentCharacter == "_" || componentCharacter == ".")
        })
    }
}

/// One immutable Hugging Face artifact declared public by the release catalog.
public struct DownloadCatalogEntry: Equatable, Sendable {

    public let huggingfaceId: String
    public let revision: String
    public let displayName: String
    public let family: DownloadCatalogFamily
    public let approximateSizeBytes: UInt64
    public let description: String?
    public let capabilities: DownloadCatalogCapabilities
    public let quantizationLabel: String?
    public let architectureSummary: String?
    public let upstreamLicense: String?
    public let downloadPathSelection: DownloadPathSelection
}

/// Human-facing capability badges surfaced from the catalog so users can
/// compare models.
public struct DownloadCatalogCapabilities: Equatable, Sendable {

    public var supportsReasoning: Bool = false
    public var supportsVision: Bool = false
    public var supportsToolCalls: Bool = false
    public var contextWindow: UInt32? = nil
    public var maxOutputTokens: UInt32? = nil
    public var supportsImageGeneration: Bool = false
    public var supportsEmbeddings: Bool = false

    public init() {
    }

    public init(
        supportsReasoning: Bool,
        supportsVision: Bool,
        supportsToolCalls: Bool,
        contextWindow: UInt32?,
        maxOutputTokens: UInt32?,
        supportsImageGeneration: Bool,
        supportsEmbeddings: Bool
    ) {
        self.supportsReasoning = supportsReasoning
        self.supportsVision = supportsVision
        self.supportsToolCalls = supportsToolCalls
        self.contextWindow = contextWindow
        self.maxOutputTokens = maxOutputTokens
        self.supportsImageGeneration = supportsImageGeneration
        self.supportsEmbeddings = supportsEmbeddings
    }
}

/// Executable model families intentionally supported by catalog version 2.
/// The wire names match the discovery spellings, including qwen_image_21's
/// underscore and modernbert's missing one.
public enum DownloadCatalogFamily: String, Equatable, Sendable {
    case qwen3_5 = "qwen3_5"
    case flux2Klein = "flux2_klein"
    case qwenImage21 = "qwen_image_21"
    case modernbert = "modernbert"
    case k2HorizonMoVA = "k2_horizon_mova"
}

/// Catalog syntax or semantic validation failure.
public enum DownloadCatalogError: Error, Equatable, CustomStringConvertible {

    case documentTooLarge
    case parse
    case unsupportedSchemaVersion(schemaVersion: UInt32)
    case tooManyEntries
    case invalidHuggingFaceId(entryIndex: Int)
    case invalidRevision(entryIndex: Int)
    case invalidDisplayName(entryIndex: Int)
    case invalidDescription(entryIndex: Int)
    case invalidCapabilities(entryIndex: Int)
    case invalidQuantizationLabel(entryIndex: Int)
    case invalidArchitectureSummary(entryIndex: Int)
    case invalidUpstreamLicense(entryIndex: Int)
    case invalidApproximateSize(entryIndex: Int)
    case modelNotPublic(entryIndex: Int)
    case duplicateHuggingFaceId
    case invalidIncludedPaths(entryIndex: Int)
    case bundledCatalogMissing

    public var description: String {
        switch (self) {
        case .documentTooLarge:
            return "download catalog exceeds the 1000000-byte metadata limit"
        case .parse:
            return "download catalog is not valid JSON"
        case let .unsupportedSchemaVersion(schemaVersion):
            return "unsupported download catalog schema version \(schemaVersion)"
        case .tooManyEntries:
            return "download catalog exceeds the 1024-entry metadata limit"
        case let .invalidHuggingFaceId(entryIndex):
            return "download catalog entry \(entryIndex) has an invalid Hugging Face identity"
        case let .invalidRevision(entryIndex):
            return "download catalog entry \(entryIndex) has an invalid immutable revision"
        case let .invalidDisplayName(entryIndex):
            return "download catalog entry \(entryIndex) has an invalid display name"
        case let .invalidDescription(entryIndex):
            return "download catalog entry \(entryIndex) has an invalid description"
        case let .invalidCapabilities(entryIndex):
            return "download catalog entry \(entryIndex) has invalid capabilities"
        case let .invalidQuantizationLabel(entryIndex):
            return "download catalog entry \(entryIndex) has an invalid quantization label"
        case let .invalidArchitectureSummary(entryIndex):
            return "download catalog entry \(entryIndex) has an invalid architecture summary"
        case let .invalidUpstreamLicense(entryIndex):
            return "download catalog entry \(entryIndex) has an invalid upstream license"
        case let .invalidApproximateSize(entryIndex):
            return "download catalog entry \(entryIndex) must declare approximate_size_bytes between 1 and 9007199254740991"
        case let .modelNotPublic(entryIndex):
            return "download catalog entry \(entryIndex) must declare public: true"
        case .duplicateHuggingFaceId:
            return "download catalog contains a duplicate or case-colliding Hugging Face identity"
        case let .invalidIncludedPaths(entryIndex):
            return "download catalog entry \(entryIndex) has invalid or overlapping included paths"
        case .bundledCatalogMissing:
            return "the bundled download catalog is missing from the daemon binary"
        }
    }
}

/// One raw entry document field set, decoded strictly: unknown fields are
/// rejected the way serde's deny_unknown_fields does.
private struct DownloadCatalogEntryDocument {

    let huggingfaceId: String
    let revision: String
    let displayName: String
    let family: DownloadCatalogFamily
    let approximateSizeBytes: UInt64
    let isPublic: Bool
    let description: String?
    let capabilities: [String: Any]?
    let quantizationLabel: String?
    let architectureSummary: String?
    let upstreamLicense: String?
    let includedPaths: Array<String>?

    static func decode(_ entryObject: [String: Any], entryIndex: Int) throws -> DownloadCatalogEntryDocument {
        let knownFieldNames: Set<String> = [
            "huggingface_id", "revision", "display_name", "family", "approximate_size_bytes",
            "public", "description", "capabilities", "quantization_label", "architecture_summary",
            "upstream_license", "included_paths"
        ]
        for entryFieldName: String in entryObject.keys where !knownFieldNames.contains(entryFieldName) {
            throw DownloadCatalogError.parse
        }
        guard let huggingFaceId: String = entryObject["huggingface_id"] as? String,
            let revision: String = entryObject["revision"] as? String,
            let displayName: String = entryObject["display_name"] as? String,
            let familyWireName: String = entryObject["family"] as? String,
            let family: DownloadCatalogFamily = DownloadCatalogFamily(rawValue: familyWireName),
            let approximateSizeBytes: UInt64 = entryObject["approximate_size_bytes"] as? UInt64,
            let isPublic: Bool = entryObject["public"] as? Bool else {
            throw DownloadCatalogError.parse
        }
        let description: String? = DownloadCatalogEntryDocument.optionalString(entryObject, fieldName: "description")
        let quantizationLabel: String? = DownloadCatalogEntryDocument.optionalString(entryObject, fieldName: "quantization_label")
        let architectureSummary: String? = DownloadCatalogEntryDocument.optionalString(entryObject, fieldName: "architecture_summary")
        let upstreamLicense: String? = DownloadCatalogEntryDocument.optionalString(entryObject, fieldName: "upstream_license")
        var includedPaths: Array<String>? = nil
        if let rawIncludedPaths: Any = entryObject["included_paths"] {
            guard let includedPathStrings: Array<String> = rawIncludedPaths as? Array<String> else {
                throw DownloadCatalogError.invalidIncludedPaths(entryIndex: entryIndex)
            }
            includedPaths = includedPathStrings
        }
        var capabilities: [String: Any]? = nil
        if let rawCapabilities: Any = entryObject["capabilities"] {
            guard let capabilitiesObject: [String: Any] = rawCapabilities as? [String: Any] else {
                throw DownloadCatalogError.invalidCapabilities(entryIndex: entryIndex)
            }
            capabilities = capabilitiesObject
        }
        return DownloadCatalogEntryDocument(
            huggingfaceId: huggingFaceId,
            revision: revision,
            displayName: displayName,
            family: family,
            approximateSizeBytes: approximateSizeBytes,
            isPublic: isPublic,
            description: description,
            capabilities: capabilities,
            quantizationLabel: quantizationLabel,
            architectureSummary: architectureSummary,
            upstreamLicense: upstreamLicense,
            includedPaths: includedPaths)
    }

    private static func optionalString(_ entryObject: [String: Any], fieldName: String) -> String? {
        guard let rawValue: Any = entryObject[fieldName], !(rawValue is NSNull) else {
            return nil
        }
        return rawValue as? String
    }
}

private struct DownloadCatalogCapabilitiesDocument {

    let supportsReasoning: Bool
    let supportsVision: Bool
    let supportsToolCalls: Bool
    let contextWindow: UInt32?
    let maxOutputTokens: UInt32?
    let supportsImageGeneration: Bool
    let supportsEmbeddings: Bool

    var declaresAnyCapability: Bool {
        return self.supportsReasoning || self.supportsVision || self.supportsToolCalls
            || self.contextWindow != nil || self.maxOutputTokens != nil
            || self.supportsImageGeneration || self.supportsEmbeddings
    }

    static func decode(_ capabilitiesObject: [String: Any]) throws -> DownloadCatalogCapabilitiesDocument {
        let knownFieldNames: Set<String> = [
            "supports_reasoning", "supports_vision", "supports_tool_calls", "context_window",
            "max_output_tokens", "supports_image_generation", "supports_embeddings"
        ]
        for capabilitiesFieldName: String in capabilitiesObject.keys where !knownFieldNames.contains(capabilitiesFieldName) {
            throw DownloadCatalogError.parse
        }
        return DownloadCatalogCapabilitiesDocument(
            supportsReasoning: (capabilitiesObject["supports_reasoning"] as? Bool) ?? false,
            supportsVision: (capabilitiesObject["supports_vision"] as? Bool) ?? false,
            supportsToolCalls: (capabilitiesObject["supports_tool_calls"] as? Bool) ?? false,
            contextWindow: capabilitiesObject["context_window"] as? UInt32,
            maxOutputTokens: capabilitiesObject["max_output_tokens"] as? UInt32,
            supportsImageGeneration: (capabilitiesObject["supports_image_generation"] as? Bool) ?? false,
            supportsEmbeddings: (capabilitiesObject["supports_embeddings"] as? Bool) ?? false)
    }
}
