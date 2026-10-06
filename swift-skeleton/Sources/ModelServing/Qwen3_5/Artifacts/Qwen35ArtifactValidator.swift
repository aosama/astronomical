import CryptoKit;
import Foundation;
import os;

/// Switchable timing pair for the Qwen3.5 artifact-validation path (model
/// loading from disk). Every load operation must be attributable to its
/// exact call site, so the start and end of each operation are captured
/// separately and only logged when the caller enables performance
/// attribution — the same contract as the daemon IPC log.
enum Qwen35ArtifactPerformanceAttribution {

    private static let performanceAttributionLog: Logger = Logger(subsystem: "dev.astronomical.model-loading", category: "performance");

    static func startedOperation(operationName: String, performanceAttributionEnabled: Bool) -> ContinuousClock.Instant? {
        if performanceAttributionEnabled == false {
            return nil;
        }
        return ContinuousClock.now;
    }

    static func finishedOperation(operationName: String, operationStart: ContinuousClock.Instant?, operationOutcome: String, performanceAttributionEnabled: Bool) -> Void {
        if performanceAttributionEnabled == false {
            return;
        }
        guard let unwrappedOperationStart: ContinuousClock.Instant = operationStart else {
            return;
        }
        let elapsedDuration: ContinuousClock.Duration = ContinuousClock.now - unwrappedOperationStart;
        let elapsedMilliseconds: Double = Double(elapsedDuration.components.seconds) * 1000.0
            + Double(elapsedDuration.components.attoseconds) / 1_000_000_000_000_000.0;
        performanceAttributionLog.info("operation=\(operationName, privacy: .public) outcome=\(operationOutcome, privacy: .public) elapsed_ms=\(elapsedMilliseconds, format: .fixed(precision: 3))");
    }
}

/// Validates the complete Qwen3.5 artifact before any native allocation.
///
/// Everything is discovered from the model directory. The config, shard
/// index, and tokenizer are the sole sources of truth. Port of
/// Qwen3_5ArtifactValidator from
/// crates/model-serving/src/qwen3_5/artifacts/artifact.rs.
public struct Qwen35ArtifactValidator {

    public init() {
    }

    /// Validates required file structure, config, index, and bounded model
    /// shard headers.
    ///
    /// Discovers everything from the model directory:
    /// - Required files (`config.json`, `tokenizer.json`,
    ///   `model.safetensors.index.json`, shards)
    /// - Shard names from the safetensors index
    /// - Vision sidecar presence from the index
    /// - Model ID from the leaf directory name
    /// - Revision from a SHA-256 hash of `config.json` bytes
    public func validate(
        modelDirectory: String, maxOutputTokens: UInt32,
        performanceAttributionEnabled: Bool = false) throws -> ValidatedQwen35Artifact {
        let validationOperationStart: ContinuousClock.Instant? = Qwen35ArtifactPerformanceAttribution
            .startedOperation(operationName: "qwen35_artifact_validate", performanceAttributionEnabled: performanceAttributionEnabled);
        do {
            let validatedArtifact: ValidatedQwen35Artifact = try self.validateBounded(
                modelDirectory: modelDirectory, maxOutputTokens: maxOutputTokens,
                performanceAttributionEnabled: performanceAttributionEnabled);
            Qwen35ArtifactPerformanceAttribution.finishedOperation(
                operationName: "qwen35_artifact_validate", operationStart: validationOperationStart,
                operationOutcome: "ok", performanceAttributionEnabled: performanceAttributionEnabled);
            return validatedArtifact;
        } catch let validationError as Qwen35ArtifactValidationError {
            Qwen35ArtifactPerformanceAttribution.finishedOperation(
                operationName: "qwen35_artifact_validate", operationStart: validationOperationStart,
                operationOutcome: "failed", performanceAttributionEnabled: performanceAttributionEnabled);
            throw validationError;
        } catch let artifactProblem as ArtifactValidationError {
            Qwen35ArtifactPerformanceAttribution.finishedOperation(
                operationName: "qwen35_artifact_validate", operationStart: validationOperationStart,
                operationOutcome: "failed", performanceAttributionEnabled: performanceAttributionEnabled);
            throw Qwen35ArtifactValidationError.artifact(artifactProblem);
        } catch let configProblem as Qwen3_5ConfigError {
            Qwen35ArtifactPerformanceAttribution.finishedOperation(
                operationName: "qwen35_artifact_validate", operationStart: validationOperationStart,
                operationOutcome: "failed", performanceAttributionEnabled: performanceAttributionEnabled);
            throw Qwen35ArtifactValidationError.config(configProblem);
        } catch let metadataProblem as OptiQMetadataError {
            Qwen35ArtifactPerformanceAttribution.finishedOperation(
                operationName: "qwen35_artifact_validate", operationStart: validationOperationStart,
                operationOutcome: "failed", performanceAttributionEnabled: performanceAttributionEnabled);
            throw Qwen35ArtifactValidationError.optiQMetadata(metadataProblem);
        } catch let shardIndexProblem as Qwen3_5ArtifactError {
            Qwen35ArtifactPerformanceAttribution.finishedOperation(
                operationName: "qwen35_artifact_validate", operationStart: validationOperationStart,
                operationOutcome: "failed", performanceAttributionEnabled: performanceAttributionEnabled);
            throw Qwen35ArtifactValidationError.shardIndex(shardIndexProblem);
        }
    }

    private func validateBounded(
        modelDirectory: String, maxOutputTokens: UInt32,
        performanceAttributionEnabled: Bool) throws -> ValidatedQwen35Artifact {
        var isDirectoryFlag: ObjCBool = false;
        guard FileManager.default.fileExists(atPath: modelDirectory, isDirectory: &isDirectoryFlag),
            isDirectoryFlag.boolValue else {
            throw Qwen35ArtifactValidationError.artifact(.modelDirectoryNotFound(modelDirectory: modelDirectory));
        }

        // A converted per-expert streaming revision declares itself with
        // manifest.json and carries no shard index; its validation is the SSD
        // streaming slice, so it fails closed here until that slice lands.
        if FileManager.default.fileExists(atPath: modelDirectory + "/manifest.json") {
            throw Qwen35ArtifactValidationError.artifact(.invalidStreamingManifest(
                problem: "per-expert streaming revision validation lands with SSD model streaming"));
        }

        // Build required file profiles for the core config files.
        // Shard files are discovered later from the safetensors index.
        var requiredFileProfiles: Array<RequiredFileProfile> = [
            Qwen35ArtifactHelpers.requiredFile(fileName: "config.json"),
            Qwen35ArtifactHelpers.requiredFile(fileName: "model.safetensors.index.json"),
            Qwen35ArtifactHelpers.requiredFile(fileName: "tokenizer.json"),
        ];

        // optiq_metadata.json is optional — validate it if present.
        if FileManager.default.fileExists(atPath: modelDirectory + "/optiq_metadata.json") {
            requiredFileProfiles.append(Qwen35ArtifactHelpers.requiredFile(fileName: "optiq_metadata.json"));
        }
        if FileManager.default.fileExists(atPath: modelDirectory + "/generation_config.json") {
            requiredFileProfiles.append(Qwen35ArtifactHelpers.requiredFile(fileName: "generation_config.json"));
        }

        let requiredFiles: Dictionary<String, ValidatedRequiredFile> = try RequiredFiles
            .validateRequiredFiles(modelDirectory: modelDirectory, requiredFileProfiles: requiredFileProfiles);

        // Read config.json and derive the revision hash from its bytes.
        let configBytes: Data = try Qwen35ArtifactHelpers.capturedRequiredFileBytes(
            requiredFiles: requiredFiles, fileName: "config.json");
        let revision: String = Self.deriveRevisionFromConfigBytes(configBytes: Array(configBytes));

        var config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(configBytes: Array(configBytes));
        let visionConfig: Qwen3_5VisionConfig? = try Qwen3_5VisionConfig
            .fromOptionalJsonBytes(configBytes: configBytes);

        // Validate optiq_metadata.json if present: the measured bit map must
        // bind exactly to the config's quantization overrides.
        if let optiQMetadataRequiredFile: ValidatedRequiredFile = requiredFiles["optiq_metadata.json"] {
            let optiQMetadataBytes: Data = try Qwen35ArtifactHelpers
                .readRequiredFileBytes(requiredFile: optiQMetadataRequiredFile);
            try OptiQMetadata.fromJsonBytes(metadataBytes: Array(optiQMetadataBytes))
                .validateAgainstConfig(qwen3_5Config: config);
        }

        // Read the shard index to discover shard names, tensor names, and
        // resolve which modules are quantized vs. stored as bfloat16.
        guard let shardIndexRequiredFile: ValidatedRequiredFile = requiredFiles["model.safetensors.index.json"] else {
            throw Qwen35ArtifactValidationError.artifact(.profileMissingRequiredFile(fileName: "model.safetensors.index.json"));
        }
        let shardIndexBytes: Data = try Qwen35ArtifactHelpers.readRequiredFileBytes(requiredFile: shardIndexRequiredFile);
        let canonicalTensorNames: Set<String> = try Qwen3_5ShardIndex
            .extractLanguageTensorNamesFromJson(indexBytes: Array(shardIndexBytes));
        config.resolveUnquantizedModulesFromShardIndex(shardTensorNames: canonicalTensorNames);
        let languageTensorProfiles: Array<TensorProfile> = Qwen3_5TensorSpec
            .qwen3_5LanguageTensorProfiles(qwen3_5Config: config);
        let shardIndex: Qwen3_5ShardIndex = try Qwen3_5ShardIndex.fromJsonBytes(
            indexBytes: Array(shardIndexBytes), languageTensorProfiles: languageTensorProfiles);
        let validatedVisionTowerStorage: ValidatedVisionTowerStorage = try Qwen35VisionValidation
            .validateVisionTowerInventory(shardIndex: shardIndex, visionConfig: visionConfig);
        var recognizedTensorProfiles: Array<TensorProfile> = languageTensorProfiles;
        if let resolvedVisionConfig: Qwen3_5VisionConfig = visionConfig {
            recognizedTensorProfiles.append(contentsOf: Qwen35VisionTensorSpec
                .visionTensorProfiles(visionConfig: resolvedVisionConfig));
        }
        let tensorInventory: TensorInventory = try Qwen35ArtifactInventory
            .buildIndexTensorInventory(shardIndex: shardIndex);
        let sourceIdByFileName: Dictionary<String, TensorSourceId> = try Qwen35ArtifactInventory
            .sourceIdByFileName(shardIndex: shardIndex);

        let shardParseOperationStart: ContinuousClock.Instant? = Qwen35ArtifactPerformanceAttribution
            .startedOperation(operationName: "qwen35_artifact_shard_parse", performanceAttributionEnabled: performanceAttributionEnabled);
        var safetensorsSources: Dictionary<TensorSourceId, ValidatedSafetensorsSource> = Dictionary();
        var totalPayloadBytes: UInt64 = 0;
        // Source ids are numbered in lexical file-name order, so ascending id
        // iteration keeps error surfacing deterministic exactly like the Rust
        // BTreeMap order.
        for shardFileEntry: (key: String, value: TensorSourceId) in sourceIdByFileName
            .sorted(by: { (leftEntry: (key: String, value: TensorSourceId), rightEntry: (key: String, value: TensorSourceId)) -> Bool in
                return leftEntry.value < rightEntry.value;
            }) {
            let validatedShardFile: ValidatedRequiredFile = try RequiredFiles.validateRequiredFile(
                modelDirectory: modelDirectory,
                requiredFileProfile: RequiredFileProfile(fileName: shardFileEntry.key, sizeBytes: 0));
            let shardSource: ValidatedSafetensorsSource = try ValidatedSafetensorsSource.parse(
                sourceId: shardFileEntry.value, requiredFile: validatedShardFile);
            try shardSource.validateInventoryProfiles(
                inventory: tensorInventory, canonicalProfiles: recognizedTensorProfiles);
            let (summedPayloadBytes, payloadOverflowed): (UInt64, Bool) = totalPayloadBytes
                .addingReportingOverflow(shardSource.payloadBytes);
            if payloadOverflowed {
                throw Qwen35ArtifactValidationError.artifact(.tensorPayloadSizeOverflow);
            }
            totalPayloadBytes = summedPayloadBytes;
            safetensorsSources[shardFileEntry.value] = shardSource;
        }
        Qwen35ArtifactPerformanceAttribution.finishedOperation(
            operationName: "qwen35_artifact_shard_parse", operationStart: shardParseOperationStart,
            operationOutcome: "ok", performanceAttributionEnabled: performanceAttributionEnabled);

        // Derive model_id from the leaf directory name.
        let modelId: String = RequiredFiles.huggingFaceSnapshotModelId(modelDirectory: modelDirectory)
            ?? URL(fileURLWithPath: modelDirectory).lastPathComponent;

        return ValidatedQwen35Artifact(
            config: config, visionConfig: visionConfig, requiredFiles: requiredFiles,
            shardIndex: shardIndex, totalPayloadBytes: totalPayloadBytes,
            hasSeparateVisionSidecar: validatedVisionTowerStorage.hasSeparateSidecar(),
            hasValidatedVisionTower: validatedVisionTowerStorage.hasValidatedVisionTower(),
            tensorInventory: tensorInventory, safetensorsSources: safetensorsSources,
            sourceIdByFileName: sourceIdByFileName, modelId: modelId, revision: revision,
            maxOutputTokens: maxOutputTokens);
    }

    /// Derives a 12-character hex revision string from the SHA-256 hash of
    /// config.json bytes. This ensures prompt cache blocks are invalidated
    /// when the model config changes.
    static func deriveRevisionFromConfigBytes(configBytes: Array<UInt8>) -> String {
        let configHash: SHA256Digest = SHA256.hash(data: Data(configBytes));
        var leadingQuadWord: UInt64 = 0;
        for digestByte: UInt8 in Array(configHash)[0..<8] {
            leadingQuadWord = (leadingQuadWord << 8) | UInt64(digestByte);
        }
        return String(format: "%012llx", leadingQuadWord);
    }
}
