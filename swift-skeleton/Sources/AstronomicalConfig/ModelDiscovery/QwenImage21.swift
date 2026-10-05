import Foundation;

/**
 * Shallow discovery for the reviewed Qwen-Image-2.1 MLX 4-bit pipeline,
 * porting crates/config/src/model_discovery/qwen_image_21.rs. Geometry,
 * the Qwen research license provenance, processor sidecars, and safetensors
 * weight indices prove the family. An immutable revision is recorded when
 * present; serving does not pin one Hub SHA.
 */
internal enum QwenImage21 {
    internal static let CANONICAL_MODEL_ID: String = "Qwen-Image-2.1-MLX-4bit";
    internal static let PROVIDER_MODEL_ID: String = "mlx-community/Qwen-Image-2.1-MLX-4bit";
    private static let MAXIMUM_JSON_BYTES: UInt64 = 4 * 1024 * 1024;
    internal static let MAXIMUM_COMPONENT_INDEX_BYTES: UInt64 = 32 * 1024 * 1024;
    private static let MAXIMUM_SIDECAR_BYTES: UInt64 = 64 * 1024 * 1024;
    private static let MAXIMUM_README_BYTES: UInt64 = 256 * 1024;
    private static let PIPELINE_CLASS_NAME: String = "QwenImage21Pipeline";
    private static let REQUIRED_PROCESSOR_FILES: Array<String> = Array<String>([
        "processor/added_tokens.json",
        "processor/chat_template.jinja",
        "processor/merges.txt",
        "processor/preprocessor_config.json",
        "processor/special_tokens_map.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "processor/vocab.json",
    ]);
    internal static let REVIEWED_COMPONENTS: Array<(componentDirectory: String, expectedClassName: String, componentErrorName: String)> = Array<(componentDirectory: String, expectedClassName: String, componentErrorName: String)>([
        (
            componentDirectory: "text_encoder",
            expectedClassName: "Qwen3VLForConditionalGeneration",
            componentErrorName: "text encoder"
        ),
        (
            componentDirectory: "transformer",
            expectedClassName: "QwenImage21Transformer2DModel",
            componentErrorName: "transformer"
        ),
        (
            componentDirectory: "vae",
            expectedClassName: "AutoencoderKLQwenImage21",
            componentErrorName: "VAE"
        ),
    ]);

    /** Trusted shallow evidence reread from one selected Qwen-Image-2.1 artifact directory. */
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
        case invalidLicenseProvenance;
        case missingOrInvalidProcessorFile(processorFile: String);
        case invalidComponentWeightIndex(component: String);
        case missingOrInvalidWeightFile(component: String);
        case modelSizeOverflow;
        case missingRevision;

        internal var description: String {
            switch (self) {
            case .invalidPipelineIndex: return "Qwen-Image-2.1 pipeline index is missing, malformed, oversized, or unsupported";
            case .invalidTransformerConfiguration: return "Qwen-Image-2.1 transformer configuration does not match the reviewed profile";
            case .invalidTextEncoderConfiguration: return "Qwen-Image-2.1 text encoder configuration does not match the reviewed profile";
            case .invalidVaeConfiguration: return "Qwen-Image-2.1 VAE configuration does not match the reviewed profile";
            case .invalidSchedulerConfiguration: return "Qwen-Image-2.1 scheduler configuration does not match the reviewed profile";
            case .invalidLicenseProvenance: return "Qwen-Image-2.1 license provenance is missing or invalid";
            case .missingOrInvalidProcessorFile(let processorFile): return "Qwen-Image-2.1 processor file \(processorFile) is missing, empty, or oversized";
            case .invalidComponentWeightIndex(let component): return "Qwen-Image-2.1 \(component) safetensors index is invalid";
            case .missingOrInvalidWeightFile(let component): return "Qwen-Image-2.1 \(component) weight file is missing, empty, or invalid";
            case .modelSizeOverflow: return "Qwen-Image-2.1 weight size exceeds the supported integer range";
            case .missingRevision: return "Qwen-Image-2.1 immutable revision evidence is missing";
            }
        }
    }

    internal static func classifiesPipelineIndex(_ pipelineIndexBytes: Data) throws -> Bool {
        let pipelineClassDocumentValue: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: pipelineIndexBytes);
        guard let pipelineClassObject: Dictionary<String, Any> = pipelineClassDocumentValue as? Dictionary<String, Any> else {
            throw DiscoveryStrictJsonDocument.ParseError.malformedJson(description: "expected a JSON object");
        }
        let pipelineClass: QwenImage21PipelineClass = try QwenImage21PipelineClass.fromJsonObject(pipelineClassObject);
        guard pipelineClass.className == QwenImage21.PIPELINE_CLASS_NAME else {
            return false;
        }
        let pipelineIndexDocumentValue: Any = try DiscoveryStrictJsonDocument.parseDocument(bytes: pipelineIndexBytes);
        guard let pipelineIndexObject: Dictionary<String, Any> = pipelineIndexDocumentValue as? Dictionary<String, Any> else {
            throw DiscoveryStrictJsonDocument.ParseError.malformedJson(description: "expected a JSON object");
        }
        let pipelineIndex: QwenImage21PipelineIndex = try QwenImage21PipelineIndex.fromJsonObject(pipelineIndexObject);
        return QwenImage21.isReviewedPipeline(pipelineIndex: pipelineIndex);
    }

    /** Rereads one exact directory without walking its parents, children, or configured scan roots. */
    internal static func verifyModelDirectory(modelDirectory: FilePath) throws -> QwenImage21.DirectoryEvidence {
        try QwenImage21.validatePipelineIndex(modelDirectory: modelDirectory);
        try QwenImage21.validateTransformerGeometry(modelDirectory: modelDirectory);
        try QwenImage21.validateTextEncoderGeometry(modelDirectory: modelDirectory);
        try QwenImage21.validateVaeGeometry(modelDirectory: modelDirectory);
        try QwenImage21.validateSchedulerGeometry(modelDirectory: modelDirectory);
        try QwenImage21.validateProcessorFiles(modelDirectory: modelDirectory);
        try QwenImage21.validateLicenseProvenance(modelDirectory: modelDirectory);
        let modelSizeBytes: UInt64 = try QwenImage21Weights.measureReviewedWeightBytes(modelDirectory: modelDirectory);
        let libraryProvenance: (providerModelId: String, revision: String)? = QwenImage21Provenance.immutableModelProvenance(
            modelDirectory: modelDirectory
        );
        var revision: String? = nil;
        if let provenanceEvidence: (providerModelId: String, revision: String) = libraryProvenance {
            revision = provenanceEvidence.revision;
        }
        if (revision == nil) {
            revision = QwenImage21Provenance.immutableFileRevision(
                modelDirectory: modelDirectory,
                authoritativeFileName: "model_index.json"
            );
        }
        guard let immutableRevision: String = revision else {
            throw QwenImage21.DirectoryVerificationError.missingRevision;
        }
        var recordedProviderModelId: String = QwenImage21.PROVIDER_MODEL_ID;
        if let provenanceEvidence: (providerModelId: String, revision: String) = libraryProvenance {
            recordedProviderModelId = provenanceEvidence.providerModelId;
        }
        return QwenImage21.DirectoryEvidence(
            canonicalModelId: QwenImage21.CANONICAL_MODEL_ID,
            providerModelId: recordedProviderModelId,
            revision: immutableRevision,
            license: ModelLicense.qwenResearch,
            capabilities: DiscoveryImageGenerationCapabilities(
                supportsTextToImage: true,
                supportsImageEditing: false,
                supportsMultipleReferenceImages: false,
                // The reference pipeline's own default; lower counts quarter-denoise the render.
                defaultSteps: 40,
                // The serving profile's reviewed envelope: 256-pixel minimum side, 1024-pixel
                // maximum, and the VAE's 32-pixel spatial multiple.
                minimumDimensionPixels: 256,
                maximumDimensionPixels: 1_024,
                dimensionMultiplePixels: 32
            ),
            modelSizeBytes: modelSizeBytes
        );
    }

    private static func validatePipelineIndex(modelDirectory: FilePath) throws -> Void {
        let isExactPipeline: Bool;
        do {
            let pipelineIndexObject: Dictionary<String, Any> = try QwenImage21.readJsonObject(
                documentPath: modelDirectory.appending(component: "model_index.json"),
                maximumBytes: QwenImage21.MAXIMUM_JSON_BYTES
            );
            let pipelineIndex: QwenImage21PipelineIndex = try QwenImage21PipelineIndex.fromJsonObject(pipelineIndexObject);
            isExactPipeline = QwenImage21.isReviewedPipeline(pipelineIndex: pipelineIndex);
        } catch {
            throw QwenImage21.DirectoryVerificationError.invalidPipelineIndex;
        }
        guard isExactPipeline else {
            throw QwenImage21.DirectoryVerificationError.invalidPipelineIndex;
        }
    }

    private static func isReviewedPipeline(pipelineIndex: QwenImage21PipelineIndex) -> Bool {
        let expectedProcessor: Array<String> = Array<String>(["transformers", "Qwen3VLProcessor"]);
        let expectedScheduler: Array<String> = Array<String>(["diffusers", "FlowMatchEulerDiscreteScheduler"]);
        let expectedTextEncoder: Array<String> = Array<String>(["transformers", "Qwen3VLForConditionalGeneration"]);
        let expectedTransformer: Array<String> = Array<String>(["diffusers", "QwenImage21Transformer2DModel"]);
        let expectedVae: Array<String> = Array<String>(["diffusers", "AutoencoderKLQwenImage21"]);
        return pipelineIndex.className == QwenImage21.PIPELINE_CLASS_NAME
            && pipelineIndex.processor == expectedProcessor
            && pipelineIndex.scheduler == expectedScheduler
            && pipelineIndex.textEncoder == expectedTextEncoder
            && pipelineIndex.transformer == expectedTransformer
            && pipelineIndex.vae == expectedVae;
    }

    /** README front matter must carry the Qwen research license triad verbatim. */
    private static func validateLicenseProvenance(modelDirectory: FilePath) throws -> Void {
        let readmeText: String;
        do {
            let readmeBytes: Data = try BoundedArtifactFile.readBoundedNonemptyFile(
                filePath: modelDirectory.appending(component: "README.md"),
                maximumBytes: QwenImage21.MAXIMUM_README_BYTES
            );
            guard let decodedReadmeText: String = String(data: readmeBytes, encoding: .utf8) else {
                throw QwenImage21.DirectoryVerificationError.invalidLicenseProvenance;
            }
            readmeText = decodedReadmeText;
        } catch {
            throw QwenImage21.DirectoryVerificationError.invalidLicenseProvenance;
        }
        let hasQwenLicense: Bool = QwenImage21.hasFrontMatterValue(readme: readmeText, key: "license", expectedValue: "other")
            && QwenImage21.hasFrontMatterValue(readme: readmeText, key: "license_name", expectedValue: "qwen-research")
            && QwenImage21.hasFrontMatterValue(readme: readmeText, key: "base_model", expectedValue: "Qwen/Qwen-Image-2.1");
        guard hasQwenLicense else {
            throw QwenImage21.DirectoryVerificationError.invalidLicenseProvenance;
        }
    }

    private static func hasFrontMatterValue(readme: String, key: String, expectedValue: String) -> Bool {
        let expectedLine: String = key + ": " + expectedValue;
        // Splitting on "\n" and trimming each line covers CRLF endings the way
        // Rust's lines() plus trim does.
        let readmeLines: Array<Substring> = readme.split(
            omittingEmptySubsequences: false,
            whereSeparator: { (readmeCharacter: Character) -> Bool in return readmeCharacter == "\n"; }
        );
        var lineIterator: Array<Substring>.Iterator = readmeLines.makeIterator();
        guard let firstLine: Substring = lineIterator.next() else {
            return false;
        }
        guard String(firstLine).trimmingCharacters(in: CharacterSet.whitespacesAndNewlines) == "---" else {
            return false;
        }
        while let rawLine: Substring = lineIterator.next() {
            let lineText: String = String(rawLine).trimmingCharacters(in: CharacterSet.whitespacesAndNewlines);
            if (lineText == "---") {
                return false;
            }
            if (lineText == expectedLine) {
                return true;
            }
        }
        return false;
    }

    private static func validateProcessorFiles(modelDirectory: FilePath) throws -> Void {
        for requiredProcessorFile: String in QwenImage21.REQUIRED_PROCESSOR_FILES {
            do {
                _ = try BoundedArtifactFile.readBoundedNonemptyFile(
                    filePath: modelDirectory.appending(component: requiredProcessorFile),
                    maximumBytes: QwenImage21.MAXIMUM_SIDECAR_BYTES
                );
            } catch {
                throw QwenImage21.DirectoryVerificationError.missingOrInvalidProcessorFile(processorFile: requiredProcessorFile);
            }
        }
    }

    private static func validateTransformerGeometry(modelDirectory: FilePath) throws -> Void {
        let isReviewedProfile: Bool;
        do {
            let geometryObject: Dictionary<String, Any> = try QwenImage21.readJsonObject(
                documentPath: modelDirectory.appending(component: "transformer/config.json"),
                maximumBytes: QwenImage21.MAXIMUM_JSON_BYTES
            );
            let geometry: QwenImage21TransformerGeometry = try QwenImage21TransformerGeometry.fromJsonObject(geometryObject);
            isReviewedProfile = geometry.className == "QwenImage21Transformer2DModel"
                && geometry.attentionHeadDim == 128
                && geometry.axesDimsRope == Array<UInt32>([16, 56, 56])
                && geometry.causalCondition
                && geometry.contextInDim == 4_096
                && geometry.eps == 0.000001
                && geometry.inChannels == 64
                && geometry.mlxFormat
                && geometry.mlpRatio == 3
                && geometry.numAttentionHeads == 32
                && geometry.numLayers == 32
                && geometry.outChannels == 64
                && geometry.patchSize == 1
                && geometry.quantization.bits == 4
                && geometry.quantization.groupSize == 64
                && geometry.quantization.mode == "affine";
        } catch {
            throw QwenImage21.DirectoryVerificationError.invalidTransformerConfiguration;
        }
        guard isReviewedProfile else {
            throw QwenImage21.DirectoryVerificationError.invalidTransformerConfiguration;
        }
    }

    private static func validateTextEncoderGeometry(modelDirectory: FilePath) throws -> Void {
        let isReviewedProfile: Bool;
        do {
            let geometryObject: Dictionary<String, Any> = try QwenImage21.readJsonObject(
                documentPath: modelDirectory.appending(component: "text_encoder/config.json"),
                maximumBytes: QwenImage21.MAXIMUM_JSON_BYTES
            );
            let geometry: QwenImage21TextEncoderGeometry = try QwenImage21TextEncoderGeometry.fromJsonObject(geometryObject);
            isReviewedProfile = geometry.architectures == Array<String>(["Qwen3VLForConditionalGeneration"])
                && geometry.dtype == "bfloat16"
                && geometry.mlxFormat
                && geometry.modelType == "qwen3_vl"
                && geometry.quantization.bits == 4
                && geometry.quantization.groupSize == 64
                && geometry.quantization.mode == "affine"
                && geometry.textConfig.dtype == "bfloat16"
                && !geometry.textConfig.attentionBias
                && geometry.textConfig.attentionDropout == 0.0
                && geometry.textConfig.headDim == 128
                && geometry.textConfig.hiddenAct == "silu"
                && geometry.textConfig.hiddenSize == 4_096
                && geometry.textConfig.intermediateSize == 12_288
                && geometry.textConfig.maxPositionEmbeddings == 262_144
                && geometry.textConfig.modelType == "qwen3_vl_text"
                && geometry.textConfig.numAttentionHeads == 32
                && geometry.textConfig.numHiddenLayers == 36
                && geometry.textConfig.numKeyValueHeads == 8
                && geometry.textConfig.rmsNormEps == 0.000001
                && geometry.textConfig.ropeScaling.mropeInterleaved
                && geometry.textConfig.ropeScaling.mropeSection == Array<UInt32>([24, 20, 20])
                && geometry.textConfig.ropeScaling.ropeType == "default"
                && geometry.textConfig.ropeTheta == 5_000_000
                && geometry.textConfig.useCache
                && geometry.textConfig.vocabSize == 151_936
                && !geometry.tieWordEmbeddings;
        } catch {
            throw QwenImage21.DirectoryVerificationError.invalidTextEncoderConfiguration;
        }
        guard isReviewedProfile else {
            throw QwenImage21.DirectoryVerificationError.invalidTextEncoderConfiguration;
        }
    }

    private static func validateVaeGeometry(modelDirectory: FilePath) throws -> Void {
        let isReviewedProfile: Bool;
        do {
            let geometryObject: Dictionary<String, Any> = try QwenImage21.readJsonObject(
                documentPath: modelDirectory.appending(component: "vae/config.json"),
                maximumBytes: QwenImage21.MAXIMUM_JSON_BYTES
            );
            let geometry: QwenImage21VaeGeometry = try QwenImage21VaeGeometry.fromJsonObject(geometryObject);
            // Latent statistics must describe every latent channel, and every
            // standard deviation must be positive so renormalization cannot
            // divide by zero at serving time.
            let latentChannelsAreDescribed: Bool = geometry.latentsMean.count == Int(geometry.zDim)
                && geometry.latentsStd.count == Int(geometry.zDim);
            let latentStandardDeviationsArePositive: Bool = geometry.latentsStd.allSatisfy(
                { (standardDeviation: Double) -> Bool in return standardDeviation > 0.0; }
            );
            isReviewedProfile = geometry.className == "AutoencoderKLQwenImage21"
                && geometry.attnScales.isEmpty
                && geometry.baseDim == 96
                && geometry.decoderBaseDim == 144
                && geometry.dimMult == Array<UInt32>([1, 2, 4, 8, 8])
                && geometry.dropout == 0.0
                && geometry.inChannels == 4
                && geometry.isResidual
                && latentChannelsAreDescribed
                && latentStandardDeviationsArePositive
                && geometry.mlxFormat
                && geometry.numResBlocks == 2
                && geometry.outChannels == 4
                && geometry.patchSize == nil
                && geometry.scaleFactorSpatial == 16
                && geometry.scaleFactorTemporal == 8
                && geometry.temporalDownsample == Array<Bool>([false, true, true, true])
                && geometry.zDim == 64;
        } catch {
            throw QwenImage21.DirectoryVerificationError.invalidVaeConfiguration;
        }
        guard isReviewedProfile else {
            throw QwenImage21.DirectoryVerificationError.invalidVaeConfiguration;
        }
    }

    private static func validateSchedulerGeometry(modelDirectory: FilePath) throws -> Void {
        let isReviewedProfile: Bool;
        do {
            let geometryObject: Dictionary<String, Any> = try QwenImage21.readJsonObject(
                documentPath: modelDirectory.appending(component: "scheduler/scheduler_config.json"),
                maximumBytes: QwenImage21.MAXIMUM_JSON_BYTES
            );
            let geometry: QwenImage21SchedulerGeometry = try QwenImage21SchedulerGeometry.fromJsonObject(geometryObject);
            isReviewedProfile = geometry.className == "FlowMatchEulerDiscreteScheduler"
                && geometry.baseImageSeqLen == 256
                && geometry.baseShift == 0.5
                && !geometry.invertSigmas
                && geometry.maxImageSeqLen == 8_192
                && geometry.maxShift == 0.9
                && geometry.numTrainTimesteps == 1_000
                && geometry.shift == 1.0
                && geometry.shiftTerminal == 0.02
                && !geometry.stochasticSampling
                && geometry.timeShiftType == "exponential"
                && !geometry.useBetaSigmas
                && geometry.useDynamicShifting
                && !geometry.useExponentialSigmas
                && !geometry.useKarrasSigmas;
        } catch {
            throw QwenImage21.DirectoryVerificationError.invalidSchedulerConfiguration;
        }
        guard isReviewedProfile else {
            throw QwenImage21.DirectoryVerificationError.invalidSchedulerConfiguration;
        }
    }

    internal static func readJsonObject(documentPath: FilePath, maximumBytes: UInt64) throws -> Dictionary<String, Any> {
        let documentValue: Any = try BoundedArtifactFile.readJsonDocument(fileAtPath: documentPath, maximumBytes: maximumBytes);
        guard let documentObject: Dictionary<String, Any> = documentValue as? Dictionary<String, Any> else {
            throw DiscoveryStrictJsonDocument.ParseError.malformedJson(description: "expected a JSON object");
        }
        return documentObject;
    }

}
