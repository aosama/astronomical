import Foundation;
import Darwin;

/**
 * Family-owned shallow discovery rules for executable Laguna artifacts,
 * porting crates/config/src/model_discovery/laguna.rs. The caseless enum is
 * the Swift equivalent of the Rust module of free functions and constants.
 */
internal enum Laguna {

    private static let MAXIMUM_INDEX_BYTES: UInt64 = 32 * 1024 * 1024;
    private static let MAXIMUM_TEXT_DOCUMENT_BYTES: UInt64 = 32 * 1024 * 1024;
    private static let SUPPORTED_PARSER_ID: String = "poolside_v1";
    private static let STANDALONE_CHAT_TEMPLATE_FILE_NAME: String = "chat_template.jinja";
    private static let MAXIMUM_REVISION_METADATA_BYTES: UInt64 = 4_096;

    /**
     * Family-derived metadata returned to neutral discovery orchestration.
     */
    internal struct DiscoveredModelMetadata: Equatable, Sendable {
        internal let revision: String;
        internal let contextWindowTokens: UInt32;
        internal let maximumInputTokens: UInt32;
        internal let maximumOutputTokens: UInt32;
        internal let hasVision: Bool;
        internal let supportsReasoning: Bool;
        internal let supportsToolCalls: Bool;
        internal let modelSizeBytes: UInt64;

        internal init(
            revision: String,
            contextWindowTokens: UInt32,
            maximumInputTokens: UInt32,
            maximumOutputTokens: UInt32,
            hasVision: Bool,
            supportsReasoning: Bool,
            supportsToolCalls: Bool,
            modelSizeBytes: UInt64
        ) {
            self.revision = revision;
            self.contextWindowTokens = contextWindowTokens;
            self.maximumInputTokens = maximumInputTokens;
            self.maximumOutputTokens = maximumOutputTokens;
            self.hasVision = hasVision;
            self.supportsReasoning = supportsReasoning;
            self.supportsToolCalls = supportsToolCalls;
            self.modelSizeBytes = modelSizeBytes;
        }
    }

    /** Recognizes the authoritative Laguna family marker. */
    internal static func recognizesModelType(_ modelType: String?) -> Bool {
        guard let presentModelType: String = modelType else {
            return false;
        }
        return presentModelType == "laguna";
    }

    /**
     * Predicts whether startup can execute one Laguna artifact without
     * reading weight payloads. Nil means the directory is not a complete
     * immutable executable Laguna artifact.
     */
    internal static func discoverModelMetadata(
        modelDirectory: FilePath,
        configBytes: Data
    ) -> Laguna.DiscoveredModelMetadata? {
        let configDocument: ConfigDocument;
        do {
            let configRootValue: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: configBytes);
            guard let configRootObject: Dictionary<String, Any> = configRootValue as? Dictionary<String, Any> else {
                return nil;
            }
            configDocument = try ConfigDocument.decode(object: configRootObject);
        } catch {
            return nil;
        }
        if (configDocument.modelType != "laguna") {
            return nil;
        }
        if (Laguna.hasExecutableLagunaStorage(configDocument: configDocument) == false) {
            return nil;
        }
        let servingLanguageFields: LanguageFields;
        if let textConfigFields: LanguageFields = configDocument.textConfig {
            servingLanguageFields = textConfigFields;
        } else {
            servingLanguageFields = configDocument.languageFields;
        }
        guard let contextWindowTokens: UInt32 = servingLanguageFields.maximumPositionEmbeddings else {
            return nil;
        }
        if (contextWindowTokens < 2) {
            return nil;
        }
        let maximumOutputTokens: UInt32 = Swift.min(UInt32(UInt16.max), contextWindowTokens - 1);

        // Startup requires all three text sidecars. Discovery reads only
        // bounded metadata needed to predict the supported public text
        // contract.
        if (Laguna.readBoundedFile(
            filePath: modelDirectory.appending(component: "tokenizer.json"),
            maximumBytes: Laguna.MAXIMUM_TEXT_DOCUMENT_BYTES
        ) == nil) {
            return nil;
        }
        guard let tokenizerConfigBytes: Data = Laguna.readBoundedFile(
            filePath: modelDirectory.appending(component: "tokenizer_config.json"),
            maximumBytes: Laguna.MAXIMUM_TEXT_DOCUMENT_BYTES
        ) else {
            return nil;
        }
        guard let generationConfigBytes: Data = Laguna.readBoundedFile(
            filePath: modelDirectory.appending(component: "generation_config.json"),
            maximumBytes: Laguna.MAXIMUM_TEXT_DOCUMENT_BYTES
        ) else {
            return nil;
        }
        guard
            let standaloneTemplateState: LagunaTemplateSource.StandaloneChatTemplateState =
                Laguna.standaloneTemplateState(modelDirectory: modelDirectory)
        else {
            return nil;
        }
        let selectedRootTemplate: LagunaTemplateSource.RootChatTemplateSource;
        do {
            selectedRootTemplate = try LagunaTemplateSource.selectRootChatTemplate(
                tokenizerConfigBytes: tokenizerConfigBytes,
                standaloneTemplateState: standaloneTemplateState
            );
        } catch {
            return nil;
        }
        let rootTemplateSource: String;
        switch (selectedRootTemplate) {
        case .embedded(let templateSource, _):
            rootTemplateSource = templateSource;
        case .standalone:
            guard let standaloneTemplateBytes: Data = Laguna.readBoundedFile(
                filePath: modelDirectory.appending(component: Laguna.STANDALONE_CHAT_TEMPLATE_FILE_NAME),
                maximumBytes: LagunaTemplateIncludes.MAXIMUM_TEMPLATE_BYTES
            ) else {
                return nil;
            }
            guard
                let standaloneTemplateSource: String = String(bytes: standaloneTemplateBytes, encoding: .utf8)
            else {
                return nil;
            }
            rootTemplateSource = standaloneTemplateSource;
        }
        var standaloneRootFileName: String?;
        switch (selectedRootTemplate) {
        case .standalone: standaloneRootFileName = Laguna.STANDALONE_CHAT_TEMPLATE_FILE_NAME;
        case .embedded(_, _): standaloneRootFileName = nil;
        }
        guard let selectedTemplateIncludeNames: Set<String> = LagunaTemplateIncludes.validateTemplateSources(
            modelDirectory: modelDirectory,
            rootTemplateSource: rootTemplateSource,
            standaloneRootFileName: standaloneRootFileName
        ) else {
            return nil;
        }
        do {
            try LagunaTemplateSource.validateStandaloneChatTemplateRole(
                rootTemplateSource: selectedRootTemplate,
                standaloneTemplateIsSelectedInclude: selectedTemplateIncludeNames.contains(
                    Laguna.STANDALONE_CHAT_TEMPLATE_FILE_NAME
                )
            );
        } catch {
            return nil;
        }
        let generationConfigDocument: GenerationConfigDocument;
        do {
            let generationConfigRootValue: Any = try DiscoveryStrictJsonDocument.parseDocument(
                bytes: generationConfigBytes
            );
            guard
                let generationConfigRootObject: Dictionary<String, Any> = generationConfigRootValue
                    as? Dictionary<String, Any>
            else {
                return nil;
            }
            generationConfigDocument = try GenerationConfigDocument.decode(object: generationConfigRootObject);
        } catch {
            return nil;
        }
        let supportsReasoning: Bool = generationConfigDocument.reasoningParser == Laguna.SUPPORTED_PARSER_ID;
        let supportsToolCalls: Bool = generationConfigDocument.toolCallParser == Laguna.SUPPORTED_PARSER_ID;
        if (supportsReasoning == false || supportsToolCalls == false) {
            return nil;
        }

        guard let modelSizeBytes: UInt64 = Laguna.validateIndexedPayload(modelDirectory: modelDirectory) else {
            return nil;
        }
        guard let revision: String = Laguna.immutableModelRevision(modelDirectory: modelDirectory) else {
            return nil;
        }
        if (Laguna.isImmutableRevision(revision: revision) == false) {
            return nil;
        }

        return Laguna.DiscoveredModelMetadata(
            revision: revision,
            contextWindowTokens: contextWindowTokens,
            maximumInputTokens: contextWindowTokens - 1,
            maximumOutputTokens: maximumOutputTokens,
            hasVision: false,
            supportsReasoning: supportsReasoning,
            supportsToolCalls: supportsToolCalls,
            modelSizeBytes: modelSizeBytes
        );
    }

    private static func validateIndexedPayload(modelDirectory: FilePath) -> UInt64? {
        guard let indexBytes: Data = Laguna.readBoundedFile(
            filePath: modelDirectory.appending(component: "model.safetensors.index.json"),
            maximumBytes: Laguna.MAXIMUM_INDEX_BYTES
        ) else {
            return nil;
        }
        let indexDocument: ShardIndexDocument;
        do {
            let indexRootValue: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: indexBytes);
            guard let indexRootObject: Dictionary<String, Any> = indexRootValue as? Dictionary<String, Any> else {
                return nil;
            }
            indexDocument = try ShardIndexDocument.decode(object: indexRootObject);
        } catch {
            return nil;
        }
        if (indexDocument.totalSizeBytes == 0 || indexDocument.tensorNameToShardFileName.isEmpty) {
            return nil;
        }
        var shardFileNames: Set<String> = Set<String>();
        for shardFileName: String in indexDocument.tensorNameToShardFileName.values {
            if (Laguna.isSafeSafetensorsFileName(shardFileName: shardFileName) == false) {
                return nil;
            }
            shardFileNames.insert(shardFileName);
        }
        var modelSizeBytes: UInt64 = 0;
        for shardFileName: String in shardFileNames {
            guard let shardLengthBytes: UInt64 = Laguna.regularFileLengthBytes(
                filePath: modelDirectory.appending(component: shardFileName)
            ) else {
                return nil;
            }
            if (shardLengthBytes == 0) {
                return nil;
            }
            let (partialValue: combinedSizeBytes, overflow: didOverflowSize) = modelSizeBytes
                .addingReportingOverflow(shardLengthBytes);
            if (didOverflowSize) {
                return nil;
            }
            modelSizeBytes = combinedSizeBytes;
        }
        // Evinced indexes count either tensor payload bytes or serialized
        // shard bytes. Payload bytes cannot exceed their complete serialized
        // files.
        if (indexDocument.totalSizeBytes > modelSizeBytes) {
            return nil;
        }
        return modelSizeBytes;
    }

    private static func isSafeSafetensorsFileName(shardFileName: String) -> Bool {
        if (shardFileName.isEmpty || shardFileName.contains("\\")) {
            return false;
        }
        let shardFilePath: FilePath = FilePath(string: shardFileName);
        if (shardFilePath.isAbsolute) {
            return false;
        }
        let shardPathSegments: Array<Substring> = shardFileName.split(
            omittingEmptySubsequences: true,
            whereSeparator: { (separatorCharacter: Character) in return separatorCharacter == "/"; }
        );
        // Rust normalizes "." components away except at the path start, so
        // "a/./b" is safe while "./a" is not.
        if (shardPathSegments.first == ".") {
            return false;
        }
        for shardPathSegment: Substring in shardPathSegments {
            if (shardPathSegment == "..") {
                return false;
            }
        }
        guard let lastSegment: Substring = shardPathSegments.last else {
            return false;
        }
        guard let extensionDotIndex: String.Index = lastSegment.lastIndex(of: ".") else {
            return false;
        }
        if (extensionDotIndex == lastSegment.startIndex) {
            return false;
        }
        let shardFileExtension: String = String(lastSegment[lastSegment.index(after: extensionDotIndex)...]);
        return shardFileExtension == "safetensors";
    }

    private static func standaloneTemplateState(
        modelDirectory: FilePath
    ) -> LagunaTemplateSource.StandaloneChatTemplateState? {
        let templatePath: FilePath = modelDirectory.appending(component: Laguna.STANDALONE_CHAT_TEMPLATE_FILE_NAME);
        var templateStatus: stat = stat();
        if (Darwin.fstatat(Darwin.AT_FDCWD, templatePath.string, &templateStatus, 0) != 0) {
            if (errno == ENOENT) {
                return LagunaTemplateSource.StandaloneChatTemplateState.missing;
            }
            return nil;
        }
        if ((templateStatus.st_mode & S_IFMT) != S_IFREG) {
            return nil;
        }
        if (UInt64(templateStatus.st_size) == 0) {
            return LagunaTemplateSource.StandaloneChatTemplateState.empty;
        }
        return LagunaTemplateSource.StandaloneChatTemplateState.nonEmpty;
    }

    private static func regularFileLengthBytes(filePath: FilePath) -> UInt64? {
        var pathStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, filePath.string, &pathStatus, 0) == 0 else {
            return nil;
        }
        if ((pathStatus.st_mode & S_IFMT) != S_IFREG) {
            return nil;
        }
        return UInt64(pathStatus.st_size);
    }

    internal static func readBoundedFile(filePath: FilePath, maximumBytes: UInt64) -> Data? {
        guard let fileLengthBytes: UInt64 = Laguna.regularFileLengthBytes(filePath: filePath),
            fileLengthBytes > 0,
            fileLengthBytes <= maximumBytes
        else {
            return nil;
        }
        do {
            let fileBytes: Data = try BoundedArtifactFile.readBoundedNonemptyFile(
                filePath: filePath,
                maximumBytes: maximumBytes
            );
            if (UInt64(fileBytes.count) > maximumBytes) {
                return nil;
            }
            return fileBytes;
        } catch {
            return nil;
        }
    }

    private static func isImmutableRevision(revision: String) -> Bool {
        if (revision.utf8.count != 40) {
            return false;
        }
        for revisionByte: UInt8 in revision.utf8 {
            let isHexadecimalByte: Bool = (revisionByte >= UInt8(ascii: "0") && revisionByte <= UInt8(ascii: "9"))
                || (revisionByte >= UInt8(ascii: "a") && revisionByte <= UInt8(ascii: "f"))
                || (revisionByte >= UInt8(ascii: "A") && revisionByte <= UInt8(ascii: "F"));
            if (isHexadecimalByte == false) {
                return false;
            }
        }
        return true;
    }

    private static func hasExecutableLagunaStorage(configDocument: ConfigDocument) -> Bool {
        // compressed-tensors is a Transformers GPU packaging, not an
        // executable MLX affine or native-floating Laguna artifact.
        var quantizationHints: Array<QuantizationHint> = Array<QuantizationHint>();
        if let declaredQuantizationHint: QuantizationHint = configDocument.quantization {
            quantizationHints.append(declaredQuantizationHint);
        }
        if let declaredQuantizationConfigHint: QuantizationHint = configDocument.quantizationConfig {
            quantizationHints.append(declaredQuantizationConfigHint);
        }
        for quantizationHint: QuantizationHint in quantizationHints {
            if (quantizationHint.quantMethod == "compressed-tensors") {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns artifact provenance only when its source records an immutable
     * revision candidate. Local port of
     * classified_artifacts::immutable_model_revision; a future shared port of
     * the classified-artifacts module should replace this helper.
     */
    private static func immutableModelRevision(modelDirectory: FilePath) -> String? {
        let authoritativeFileName: String;
        if (DiscoveryPathNavigation.isExistingRegularFile(
            path: modelDirectory.appending(component: "model_index.json")
        )) {
            authoritativeFileName = "model_index.json";
        } else {
            authoritativeFileName = "config.json";
        }
        let localMetadataPath: FilePath = modelDirectory.appending(
            component: ".cache/huggingface/download/" + authoritativeFileName + ".metadata"
        );
        var metadataStatus: stat = stat();
        if (Darwin.fstatat(Darwin.AT_FDCWD, localMetadataPath.string, &metadataStatus, 0) == 0
            && UInt64(metadataStatus.st_size) <= Laguna.MAXIMUM_REVISION_METADATA_BYTES)
        {
            guard let metadataBytes: Data = FileManager.default.contents(atPath: localMetadataPath.string) else {
                return nil;
            }
            guard let metadataText: String = String(bytes: metadataBytes, encoding: .utf8) else {
                return nil;
            }
            // str::lines yields nothing for an empty document, so an empty
            // metadata file records no revision at all.
            if (metadataText.isEmpty) {
                return nil;
            }
            var firstLineText: String;
            if let firstNewlineIndex: String.Index = metadataText.firstIndex(of: "\n") {
                firstLineText = String(metadataText[..<firstNewlineIndex]);
            } else {
                firstLineText = metadataText;
            }
            if (firstLineText.hasSuffix("\r")) {
                firstLineText.removeLast();
            }
            return firstLineText;
        }
        var hasHuggingfaceCacheAncestor: Bool = false;
        for ancestorPath: FilePath in DiscoveryPathNavigation.ancestorDirectoryPaths(startingFrom: modelDirectory) {
            guard let ancestorName: String = DiscoveryPathNavigation.lastComponentName(of: ancestorPath) else {
                continue;
            }
            if (ModelIdentity.decodeHuggingfaceCacheDirectoryName(directoryName: ancestorName) != nil) {
                hasHuggingfaceCacheAncestor = true;
                break;
            }
        }
        if (hasHuggingfaceCacheAncestor) {
            return DiscoveryPathNavigation.lastComponentName(of: modelDirectory);
        }
        return nil;
    }

    /**
     * The Laguna configuration document, decoded with serde's field
     * semantics: unknown keys ignored, Option fields absent-or-null, and the
     * flattened language fields read from the document root.
     */
    private struct ConfigDocument {
        let modelType: String;
        let textConfig: LanguageFields?;
        let quantization: QuantizationHint?;
        let quantizationConfig: QuantizationHint?;
        let languageFields: LanguageFields;

        fileprivate static func decode(object: Dictionary<String, Any>) throws -> ConfigDocument {
            let modelType: String = try StrictJson.requiredString(object: object, fieldName: "model_type");
            let textConfig: LanguageFields? = try StrictJson.decodeOptional(
                try StrictJson.optionalObject(object: object, fieldName: "text_config"),
                decode: { (sectionObject: Dictionary<String, Any>) throws -> LanguageFields in
                    return try LanguageFields.decode(object: sectionObject);
                }
            );
            let quantization: QuantizationHint? = try StrictJson.decodeOptional(
                try StrictJson.optionalObject(object: object, fieldName: "quantization"),
                decode: { (sectionObject: Dictionary<String, Any>) throws -> QuantizationHint in
                    return try QuantizationHint.decode(object: sectionObject);
                }
            );
            let quantizationConfig: QuantizationHint? = try StrictJson.decodeOptional(
                try StrictJson.optionalObject(object: object, fieldName: "quantization_config"),
                decode: { (sectionObject: Dictionary<String, Any>) throws -> QuantizationHint in
                    return try QuantizationHint.decode(object: sectionObject);
                }
            );
            let languageFields: LanguageFields = try LanguageFields.decode(object: object);
            return ConfigDocument(
                modelType: modelType,
                textConfig: textConfig,
                quantization: quantization,
                quantizationConfig: quantizationConfig,
                languageFields: languageFields
            );
        }
    }

    private struct QuantizationHint {
        let quantMethod: String?;

        fileprivate static func decode(object: Dictionary<String, Any>) throws -> QuantizationHint {
            let quantMethod: String? = try StrictJson.optionalString(object: object, fieldName: "quant_method");
            return QuantizationHint(quantMethod: quantMethod);
        }
    }

    private struct LanguageFields {
        let maximumPositionEmbeddings: UInt32?;

        fileprivate static func decode(object: Dictionary<String, Any>) throws -> LanguageFields {
            let maximumPositionEmbeddings: UInt32? = try StrictJson.optionalUnsignedInteger(
                object: object,
                fieldName: "max_position_embeddings"
            );
            return LanguageFields(maximumPositionEmbeddings: maximumPositionEmbeddings);
        }
    }

    private struct GenerationConfigDocument {
        let reasoningParser: String;
        let toolCallParser: String;

        fileprivate static func decode(object: Dictionary<String, Any>) throws -> GenerationConfigDocument {
            let reasoningParser: String = try StrictJson.requiredString(object: object, fieldName: "reasoning_parser");
            let toolCallParser: String = try StrictJson.requiredString(object: object, fieldName: "tool_call_parser");
            return GenerationConfigDocument(reasoningParser: reasoningParser, toolCallParser: toolCallParser);
        }
    }

    private struct ShardIndexDocument {
        let totalSizeBytes: UInt64;
        let tensorNameToShardFileName: Dictionary<String, String>;

        fileprivate static func decode(object: Dictionary<String, Any>) throws -> ShardIndexDocument {
            let metadataObject: Dictionary<String, Any> = try StrictJson.objectValue(
                object: object,
                fieldName: "metadata"
            );
            let totalSizeBytes: UInt64 = try StrictJson.requiredUnsignedInteger(
                object: metadataObject,
                fieldName: "total_size"
            );
            let weightMapObject: Dictionary<String, Any> = try StrictJson.objectValue(
                object: object,
                fieldName: "weight_map"
            );
            var tensorNameToShardFileName: Dictionary<String, String> = Dictionary<String, String>();
            for (key: tensorName, value: tensorFileNameValue) in weightMapObject {
                guard let tensorFileName: String = tensorFileNameValue as? String else {
                    throw StrictJsonError(fieldName: tensorName, problem: "must be a string");
                }
                tensorNameToShardFileName[tensorName] = tensorFileName;
            }
            return ShardIndexDocument(
                totalSizeBytes: totalSizeBytes,
                tensorNameToShardFileName: tensorNameToShardFileName
            );
        }
    }
}
