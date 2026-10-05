import Foundation;

/**
 * Shallow discovery for a distilled FLUX.2 Klein 4B Diffusers pipeline,
 * porting crates/config/src/model_discovery/flux2_klein.rs. Geometry,
 * Apache-2.0 license, and required modular files prove the family. An
 * immutable revision is recorded when present; serving does not pin one Hub
 * SHA.
 */
internal enum Flux2Klein {
    internal static let CANONICAL_MODEL_ID: String = "FLUX.2-klein-4B";
    internal static let PROVIDER_MODEL_ID: String = "black-forest-labs/FLUX.2-klein-4B";
    private static let MAXIMUM_JSON_BYTES: UInt64 = 4 * 1024 * 1024;
    private static let MAXIMUM_TEXT_ENCODER_INDEX_BYTES: UInt64 = 32 * 1024 * 1024;
    private static let MAXIMUM_SIDECAR_BYTES: UInt64 = 64 * 1024 * 1024;
    private static let MAXIMUM_LICENSE_BYTES: UInt64 = 64 * 1024;
    private static let PIPELINE_CLASS_NAME: String = "Flux2KleinPipeline";
    private static let REQUIRED_SIDECARS: Array<String> = Array<String>([
        "text_encoder/generation_config.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
    ]);

    /** Trusted shallow evidence reread from one selected FLUX artifact directory. */
    internal struct DirectoryEvidence: Equatable, Sendable {
        internal let canonicalModelId: String;
        internal let providerModelId: String;
        internal let revision: String;
        internal let license: ModelLicense;
        internal let capabilities: DiscoveryImageGenerationCapabilities;
        internal let modelSizeBytes: UInt64;

        internal init(
            canonicalModelId: String,
            providerModelId: String,
            revision: String,
            license: ModelLicense,
            capabilities: DiscoveryImageGenerationCapabilities,
            modelSizeBytes: UInt64
        ) {
            self.canonicalModelId = canonicalModelId;
            self.providerModelId = providerModelId;
            self.revision = revision;
            self.license = license;
            self.capabilities = capabilities;
            self.modelSizeBytes = modelSizeBytes;
        }
    }

    /** Bounded path-free rejection for a directory that does not prove the reviewed profile. */
    internal enum DirectoryVerificationError: Error, Equatable, CustomStringConvertible {
        case invalidPipelineIndex;
        case invalidTransformerConfiguration;
        case invalidTextEncoderConfiguration;
        case invalidVaeConfiguration;
        case invalidSchedulerConfiguration;
        case invalidLicense;
        case missingOrInvalidSidecar(sidecar: String);
        case invalidTextEncoderWeightIndex;
        case missingOrInvalidWeightFile(component: String);
        case modelSizeOverflow;
        case missingRevision;

        internal var description: String {
            switch (self) {
            case .invalidPipelineIndex: return "FLUX.2 Klein pipeline index is missing, malformed, oversized, or unsupported";
            case .invalidTransformerConfiguration: return "FLUX.2 Klein transformer configuration does not match the reviewed profile";
            case .invalidTextEncoderConfiguration: return "FLUX.2 Klein text encoder configuration does not match the reviewed profile";
            case .invalidVaeConfiguration: return "FLUX.2 Klein VAE configuration does not match the reviewed profile";
            case .invalidSchedulerConfiguration: return "FLUX.2 Klein scheduler configuration does not match the reviewed profile";
            case .invalidLicense: return "FLUX.2 Klein license evidence is missing or invalid";
            case .missingOrInvalidSidecar(let sidecar): return "FLUX.2 Klein required sidecar \(sidecar) is missing, empty, or oversized";
            case .invalidTextEncoderWeightIndex: return "FLUX.2 Klein text encoder weight index is invalid";
            case .missingOrInvalidWeightFile(let component): return "FLUX.2 Klein \(component) weight file is missing, empty, or invalid";
            case .modelSizeOverflow: return "FLUX.2 Klein weight size exceeds the supported integer range";
            case .missingRevision: return "FLUX.2 Klein immutable revision evidence is missing";
            }
        }
    }

    internal static func classifiesPipelineIndex(_ pipelineIndexBytes: Data) throws -> Bool {
        let pipelineClassDocumentValue: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: pipelineIndexBytes);
        guard let pipelineClassObject: Dictionary<String, Any> = pipelineClassDocumentValue as? Dictionary<String, Any> else {
            throw DiscoveryStrictJsonDocument.ParseError.malformedJson(description: "expected a JSON object");
        }
        let pipelineClass: PipelineClass = try PipelineClass.fromJsonObject(pipelineClassObject);
        guard pipelineClass.className == Flux2Klein.PIPELINE_CLASS_NAME else {
            return false;
        }
        let pipelineIndexDocumentValue: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: pipelineIndexBytes);
        guard let pipelineIndexObject: Dictionary<String, Any> = pipelineIndexDocumentValue as? Dictionary<String, Any> else {
            throw DiscoveryStrictJsonDocument.ParseError.malformedJson(description: "expected a JSON object");
        }
        let pipelineIndex: PipelineIndex = try PipelineIndex.fromJsonObject(pipelineIndexObject);
        return Flux2Klein.isExactDistilledPipeline(pipelineIndex: pipelineIndex);
    }

    /** Rereads one exact directory without walking its parents, children, or configured scan roots. */
    internal static func verifyModelDirectory(modelDirectory: FilePath) throws -> Flux2Klein.DirectoryEvidence {
        return try Flux2Klein.verifyModelDirectoryEvidence(modelDirectory: modelDirectory);
    }

    /** Requires the authoritative modular package rather than its duplicate single-file export. */
    private static func verifyModelDirectoryEvidence(modelDirectory: FilePath) throws -> Flux2Klein.DirectoryEvidence {
        try Flux2Klein.validatePipelineIndex(modelDirectory: modelDirectory);
        try Flux2Klein.validateTransformerGeometry(modelDirectory: modelDirectory);
        try Flux2Klein.validateTextEncoderGeometry(modelDirectory: modelDirectory);
        try Flux2Klein.validateVaeGeometry(modelDirectory: modelDirectory);
        try Flux2Klein.validateSchedulerGeometry(modelDirectory: modelDirectory);
        try Flux2Klein.validateApache2License(modelDirectory: modelDirectory);
        for requiredSidecar: String in Flux2Klein.REQUIRED_SIDECARS {
            do {
                _ = try BoundedArtifactFile.readBoundedNonemptyFile(
                    filePath: modelDirectory.appending(component: requiredSidecar),
                    maximumBytes: Flux2Klein.MAXIMUM_SIDECAR_BYTES
                );
            } catch {
                throw Flux2Klein.DirectoryVerificationError.missingOrInvalidSidecar(sidecar: requiredSidecar);
            }
        }
        let modelSizeBytes: UInt64 = try Flux2Klein.measureModularWeightBytes(modelDirectory: modelDirectory);
        let libraryProvenance: (providerModelId: String, revision: String)? = Flux2KleinProvenance.immutableModelProvenance(
            modelDirectory: modelDirectory
        );
        var revision: String? = nil;
        if let provenanceEvidence: (providerModelId: String, revision: String) = libraryProvenance {
            revision = provenanceEvidence.revision;
        }
        if (revision == nil) {
            revision = Flux2KleinProvenance.immutableFileRevision(
                modelDirectory: modelDirectory,
                authoritativeFileName: "model_index.json"
            );
        }
        guard let immutableRevision: String = revision else {
            throw Flux2Klein.DirectoryVerificationError.missingRevision;
        }
        var recordedProviderModelId: String = Flux2Klein.PROVIDER_MODEL_ID;
        if let provenanceEvidence: (providerModelId: String, revision: String) = libraryProvenance {
            recordedProviderModelId = provenanceEvidence.providerModelId;
        }
        return Flux2Klein.DirectoryEvidence(
            canonicalModelId: Flux2Klein.CANONICAL_MODEL_ID,
            providerModelId: recordedProviderModelId,
            revision: immutableRevision,
            license: ModelLicense.apache20,
            capabilities: DiscoveryImageGenerationCapabilities(
                supportsTextToImage: true,
                supportsImageEditing: false,
                supportsMultipleReferenceImages: false,
                // FLUX.2 Klein is a distilled turbo model whose reference schedule is four steps.
                defaultSteps: 4,
                // The serving profile's reviewed envelope: 64-pixel minimum side, 1024-pixel
                // maximum, and the 16-pixel latent alignment.
                minimumDimensionPixels: 64,
                maximumDimensionPixels: 1_024,
                dimensionMultiplePixels: 16
            ),
            modelSizeBytes: modelSizeBytes
        );
    }

    private static func validateApache2License(modelDirectory: FilePath) throws -> Void {
        let licenseText: String;
        do {
            let licenseBytes: Data = try BoundedArtifactFile.readBoundedNonemptyFile(
                filePath: modelDirectory.appending(component: "LICENSE.md"),
                maximumBytes: Flux2Klein.MAXIMUM_LICENSE_BYTES
            );
            guard let decodedLicenseText: String = String(data: licenseBytes, encoding: .utf8) else {
                throw Flux2Klein.DirectoryVerificationError.invalidLicense;
            }
            licenseText = decodedLicenseText;
        } catch {
            throw Flux2Klein.DirectoryVerificationError.invalidLicense;
        }
        let hasApacheNotice: Bool = licenseText.contains("Apache License")
            && licenseText.contains("Version 2.0, January 2004")
            && licenseText.contains("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION")
            && licenseText.contains("END OF TERMS AND CONDITIONS");
        guard hasApacheNotice else {
            throw Flux2Klein.DirectoryVerificationError.invalidLicense;
        }
    }

    private static func validatePipelineIndex(modelDirectory: FilePath) throws -> Void {
        let isExactPipeline: Bool;
        do {
            let pipelineIndexObject: Dictionary<String, Any> = try Flux2Klein.readJsonObject(
                documentPath: modelDirectory.appending(component: "model_index.json"),
                maximumBytes: Flux2Klein.MAXIMUM_JSON_BYTES
            );
            let pipelineIndex: PipelineIndex = try PipelineIndex.fromJsonObject(pipelineIndexObject);
            isExactPipeline = Flux2Klein.isExactDistilledPipeline(pipelineIndex: pipelineIndex);
        } catch {
            throw Flux2Klein.DirectoryVerificationError.invalidPipelineIndex;
        }
        guard isExactPipeline else {
            throw Flux2Klein.DirectoryVerificationError.invalidPipelineIndex;
        }
    }

    private static func isExactDistilledPipeline(pipelineIndex: PipelineIndex) -> Bool {
        let expectedScheduler: Array<String> = Array<String>(["diffusers", "FlowMatchEulerDiscreteScheduler"]);
        let expectedTextEncoder: Array<String> = Array<String>(["transformers", "Qwen3ForCausalLM"]);
        let expectedTokenizer: Array<String> = Array<String>(["transformers", "Qwen2TokenizerFast"]);
        let expectedTransformer: Array<String> = Array<String>(["diffusers", "Flux2Transformer2DModel"]);
        let expectedVae: Array<String> = Array<String>(["diffusers", "AutoencoderKLFlux2"]);
        return pipelineIndex.className == Flux2Klein.PIPELINE_CLASS_NAME
            && pipelineIndex.isDistilled
            && pipelineIndex.scheduler == expectedScheduler
            && pipelineIndex.textEncoder == expectedTextEncoder
            && pipelineIndex.tokenizer == expectedTokenizer
            && pipelineIndex.transformer == expectedTransformer
            && pipelineIndex.vae == expectedVae;
    }

    private static func validateTransformerGeometry(modelDirectory: FilePath) throws -> Void {
        let isReviewedProfile: Bool;
        do {
            let geometryObject: Dictionary<String, Any> = try Flux2Klein.readJsonObject(
                documentPath: modelDirectory.appending(component: "transformer/config.json"),
                maximumBytes: Flux2Klein.MAXIMUM_JSON_BYTES
            );
            let geometry: TransformerGeometry = try TransformerGeometry.fromJsonObject(geometryObject);
            isReviewedProfile = geometry.className == "Flux2Transformer2DModel"
                && geometry.attentionHeadDim == 128
                && geometry.axesDimsRope == Array<UInt32>([32, 32, 32, 32])
                && geometry.eps == 0.000001
                && !geometry.guidanceEmbeds
                && geometry.inChannels == 128
                && geometry.jointAttentionDim == 7_680
                && geometry.mlpRatio == 3.0
                && geometry.numAttentionHeads == 24
                && geometry.numLayers == 5
                && geometry.numSingleLayers == 20
                && geometry.outChannels == nil
                && geometry.patchSize == 1
                && geometry.ropeTheta == 2_000.0
                && geometry.timestepGuidanceChannels == 256;
        } catch {
            throw Flux2Klein.DirectoryVerificationError.invalidTransformerConfiguration;
        }
        guard isReviewedProfile else {
            throw Flux2Klein.DirectoryVerificationError.invalidTransformerConfiguration;
        }
    }

    private static func validateTextEncoderGeometry(modelDirectory: FilePath) throws -> Void {
        let isReviewedProfile: Bool;
        do {
            let geometryObject: Dictionary<String, Any> = try Flux2Klein.readJsonObject(
                documentPath: modelDirectory.appending(component: "text_encoder/config.json"),
                maximumBytes: Flux2Klein.MAXIMUM_JSON_BYTES
            );
            let geometry: TextEncoderGeometry = try TextEncoderGeometry.fromJsonObject(geometryObject);
            let isFullAttentionLayerTypes: Bool = geometry.layerTypes.count == 36
                && geometry.layerTypes.allSatisfy({ (layerType: String) -> Bool in return layerType == "full_attention"; });
            isReviewedProfile = geometry.architectures == Array<String>(["Qwen3ForCausalLM"])
                && !geometry.attentionBias
                && geometry.attentionDropout == 0.0
                && geometry.dtype == "bfloat16"
                && geometry.headDim == 128
                && geometry.hiddenAct == "silu"
                && geometry.hiddenSize == 2_560
                && geometry.intermediateSize == 9_728
                && isFullAttentionLayerTypes
                && geometry.maxPositionEmbeddings == 40_960
                && geometry.maxWindowLayers == 36
                && geometry.modelType == "qwen3"
                && geometry.numAttentionHeads == 32
                && geometry.numHiddenLayers == 36
                && geometry.numKeyValueHeads == 8
                && geometry.rmsNormEps == 0.000001
                && geometry.ropeScaling == nil
                && geometry.ropeTheta == 1_000_000.0
                && geometry.slidingWindow == nil
                && geometry.tieWordEmbeddings
                && geometry.useCache
                && !geometry.useSlidingWindow
                && geometry.vocabSize == 151_936;
        } catch {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderConfiguration;
        }
        guard isReviewedProfile else {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderConfiguration;
        }
    }

    private static func validateVaeGeometry(modelDirectory: FilePath) throws -> Void {
        let isReviewedProfile: Bool;
        do {
            let geometryObject: Dictionary<String, Any> = try Flux2Klein.readJsonObject(
                documentPath: modelDirectory.appending(component: "vae/config.json"),
                maximumBytes: Flux2Klein.MAXIMUM_JSON_BYTES
            );
            let geometry: VaeGeometry = try VaeGeometry.fromJsonObject(geometryObject);
            let fourDownBlocks: Array<String> = Array<String>([
                "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",
            ]);
            let fourUpBlocks: Array<String> = Array<String>([
                "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",
            ]);
            isReviewedProfile = geometry.className == "AutoencoderKLFlux2"
                && geometry.actFn == "silu"
                && geometry.batchNormEps == 0.0001
                && geometry.batchNormMomentum == 0.1
                && geometry.blockOutChannels == Array<UInt32>([128, 256, 512, 512])
                && geometry.downBlockTypes == fourDownBlocks
                && geometry.forceUpcast
                && geometry.inChannels == 3
                && geometry.latentChannels == 32
                && geometry.layersPerBlock == 2
                && geometry.midBlockAddAttention
                && geometry.normNumGroups == 32
                && geometry.outChannels == 3
                && geometry.patchSize == Array<UInt32>([2, 2])
                && geometry.sampleSize == 1_024
                && geometry.upBlockTypes == fourUpBlocks
                && geometry.usePostQuantConv
                && geometry.useQuantConv;
        } catch {
            throw Flux2Klein.DirectoryVerificationError.invalidVaeConfiguration;
        }
        guard isReviewedProfile else {
            throw Flux2Klein.DirectoryVerificationError.invalidVaeConfiguration;
        }
    }

    private static func validateSchedulerGeometry(modelDirectory: FilePath) throws -> Void {
        let isReviewedProfile: Bool;
        do {
            let geometryObject: Dictionary<String, Any> = try Flux2Klein.readJsonObject(
                documentPath: modelDirectory.appending(component: "scheduler/scheduler_config.json"),
                maximumBytes: Flux2Klein.MAXIMUM_JSON_BYTES
            );
            let geometry: SchedulerGeometry = try SchedulerGeometry.fromJsonObject(geometryObject);
            isReviewedProfile = geometry.className == "FlowMatchEulerDiscreteScheduler"
                && geometry.baseImageSeqLength == 256
                && geometry.baseShift == 0.5
                && !geometry.invertSigmas
                && geometry.maxImageSeqLength == 4_096
                && geometry.maxShift == 1.15
                && geometry.numTrainTimesteps == 1_000
                && geometry.shift == 3.0
                && geometry.shiftTerminal == nil
                && !geometry.stochasticSampling
                && geometry.timeShiftType == "exponential"
                && !geometry.useBetaSigmas
                && geometry.useDynamicShifting
                && !geometry.useExponentialSigmas
                && !geometry.useKarrasSigmas;
        } catch {
            throw Flux2Klein.DirectoryVerificationError.invalidSchedulerConfiguration;
        }
        guard isReviewedProfile else {
            throw Flux2Klein.DirectoryVerificationError.invalidSchedulerConfiguration;
        }
    }

    private static func measureModularWeightBytes(modelDirectory: FilePath) throws -> UInt64 {
        let textEncoderIndex: TextEncoderIndex;
        do {
            let indexObject: Dictionary<String, Any> = try Flux2Klein.readJsonObject(
                documentPath: modelDirectory.appending(component: "text_encoder/model.safetensors.index.json"),
                maximumBytes: Flux2Klein.MAXIMUM_TEXT_ENCODER_INDEX_BYTES
            );
            textEncoderIndex = try TextEncoderIndex.fromJsonObject(indexObject);
        } catch {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        guard textEncoderIndex.metadata.totalSize != 0 else {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        var indexedShardPaths: Set<String> = Set<String>();
        for shardPath: String in textEncoderIndex.weightMap.values {
            guard Flux2Klein.isSafeSafetensorsPath(shardPath: shardPath) else {
                throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
            }
            indexedShardPaths.insert(shardPath);
        }
        guard !indexedShardPaths.isEmpty else {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        var textEncoderWeightSizeBytes: UInt64 = 0;
        var textEncoderPayloadSizeBytes: UInt64 = 0;
        // Sorted iteration mirrors the Rust BTreeSet's deterministic shard order.
        for shardPath: String in indexedShardPaths.sorted() {
            let indexedWeightPath: FilePath = modelDirectory
                .appending(component: "text_encoder")
                .appending(component: shardPath);
            let shardWeightSizeBytes: UInt64 = try Flux2Klein.requiredWeightSize(
                weightPath: indexedWeightPath,
                componentName: "text encoder"
            );
            let weightOverflowResult: (partialValue: UInt64, overflow: Bool) = textEncoderWeightSizeBytes
                .addingReportingOverflow(shardWeightSizeBytes);
            if (weightOverflowResult.overflow) {
                throw Flux2Klein.DirectoryVerificationError.modelSizeOverflow;
            }
            textEncoderWeightSizeBytes = weightOverflowResult.partialValue;
            let shardPayloadSizeBytes: UInt64 = try Flux2Klein.requiredSafetensorsPayloadSize(weightPath: indexedWeightPath);
            let payloadOverflowResult: (partialValue: UInt64, overflow: Bool) = textEncoderPayloadSizeBytes
                .addingReportingOverflow(shardPayloadSizeBytes);
            if (payloadOverflowResult.overflow) {
                throw Flux2Klein.DirectoryVerificationError.modelSizeOverflow;
            }
            textEncoderPayloadSizeBytes = payloadOverflowResult.partialValue;
        }
        guard textEncoderIndex.metadata.totalSize == textEncoderPayloadSizeBytes else {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        var modularWeightSizeBytes: UInt64 = textEncoderWeightSizeBytes;
        let modularWeights: Array<(componentName: String, weightPath: FilePath)> = Array<(componentName: String, weightPath: FilePath)>([
            (
                componentName: "transformer",
                weightPath: modelDirectory.appending(component: "transformer/diffusion_pytorch_model.safetensors")
            ),
            (
                componentName: "vae",
                weightPath: modelDirectory.appending(component: "vae/diffusion_pytorch_model.safetensors")
            ),
        ]);
        for modularWeight: (componentName: String, weightPath: FilePath) in modularWeights {
            let componentWeightSizeBytes: UInt64 = try Flux2Klein.requiredWeightSize(
                weightPath: modularWeight.weightPath,
                componentName: modularWeight.componentName
            );
            let componentOverflowResult: (partialValue: UInt64, overflow: Bool) = modularWeightSizeBytes
                .addingReportingOverflow(componentWeightSizeBytes);
            if (componentOverflowResult.overflow) {
                throw Flux2Klein.DirectoryVerificationError.modelSizeOverflow;
            }
            modularWeightSizeBytes = componentOverflowResult.partialValue;
        }
        return modularWeightSizeBytes;
    }

    private static func requiredSafetensorsPayloadSize(weightPath: FilePath) throws -> UInt64 {
        var weightStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, weightPath.string, &weightStatus, 0) == 0 else {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        let weightFileSizeBytes: UInt64 = UInt64(weightStatus.st_size);
        let weightFileHandle: FileHandle;
        do {
            weightFileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: weightPath.string));
        } catch {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        defer {
            weightFileHandle.closeFile();
        }
        let headerLengthBytes: Data = weightFileHandle.readData(ofLength: 8);
        guard headerLengthBytes.count == 8 else {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        let headerSizeBytes: UInt64 = Flux2Klein.littleEndianUInt64(of: headerLengthBytes);
        let headerRemovalResult: (partialValue: UInt64, overflow: Bool) = weightFileSizeBytes.subtractingReportingOverflow(8);
        if (headerRemovalResult.overflow) {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        let payloadRemovalResult: (partialValue: UInt64, overflow: Bool) = headerRemovalResult.partialValue
            .subtractingReportingOverflow(headerSizeBytes);
        if (payloadRemovalResult.overflow) {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        guard payloadRemovalResult.partialValue > 0 else {
            throw Flux2Klein.DirectoryVerificationError.invalidTextEncoderWeightIndex;
        }
        return payloadRemovalResult.partialValue;
    }

    private static func requiredWeightSize(weightPath: FilePath, componentName: String) throws -> UInt64 {
        var weightStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, weightPath.string, &weightStatus, 0) == 0 else {
            throw Flux2Klein.DirectoryVerificationError.missingOrInvalidWeightFile(component: componentName);
        }
        guard (weightStatus.st_mode & S_IFMT) == S_IFREG else {
            throw Flux2Klein.DirectoryVerificationError.missingOrInvalidWeightFile(component: componentName);
        }
        let weightSizeBytes: UInt64 = UInt64(weightStatus.st_size);
        guard weightSizeBytes > 0 else {
            throw Flux2Klein.DirectoryVerificationError.missingOrInvalidWeightFile(component: componentName);
        }
        return weightSizeBytes;
    }

    private static func isSafeSafetensorsPath(shardPath: String) -> Bool {
        if shardPath.isEmpty || shardPath.contains("\\") {
            return false;
        }
        let shardFilePath: FilePath = FilePath(string: shardPath);
        if (shardFilePath.isAbsolute) {
            return false;
        }
        // Rust keeps only Normal components: a leading "." is CurDir and any
        // ".." is ParentDir, so both reject; interior "." normalizes away.
        let shardPathComponents: Array<String> = shardPath.split(
            omittingEmptySubsequences: true,
            whereSeparator: { (pathCharacter: Character) -> Bool in return pathCharacter == "/"; }
        ).map({ (pathComponent: Substring) -> String in return String(pathComponent); });
        for (offset: componentIndex, element: pathComponent) in shardPathComponents.enumerated() {
            if (componentIndex == 0 && pathComponent == ".") {
                return false;
            }
            if (pathComponent == "..") {
                return false;
            }
        }
        guard let shardFileName: String = DiscoveryPathNavigation.lastComponentName(of: shardFilePath) else {
            return false;
        }
        guard let extensionDotIndex: String.Index = shardFileName.lastIndex(of: ".") else {
            return false;
        }
        guard extensionDotIndex > shardFileName.startIndex else {
            return false;
        }
        let shardPathExtension: String = String(shardFileName[shardFileName.index(after: extensionDotIndex)...]);
        return shardPathExtension == "safetensors";
    }

    private static func littleEndianUInt64(of headerBytes: Data) -> UInt64 {
        var decodedSize: UInt64 = 0;
        for (offset: byteIndex, element: headerByte) in headerBytes.enumerated() {
            decodedSize = decodedSize | (UInt64(headerByte) << (8 * UInt64(byteIndex)));
        }
        return decodedSize;
    }

    private static func readJsonObject(documentPath: FilePath, maximumBytes: UInt64) throws -> Dictionary<String, Any> {
        let documentValue: Any = try BoundedArtifactFile.readJsonDocument(fileAtPath: documentPath, maximumBytes: maximumBytes);
        guard let documentObject: Dictionary<String, Any> = documentValue as? Dictionary<String, Any> else {
            throw DiscoveryStrictJsonDocument.ParseError.malformedJson(description: "expected a JSON object");
        }
        return documentObject;
    }

}
