import Foundation;

/// Descriptor-backed validated ownership of the complete Qwen3.5 artifact.
/// Port of ValidatedQwen3_5Artifact from
/// crates/model-serving/src/qwen3_5/artifacts/validated_artifact.rs.
public final class ValidatedQwen35Artifact {

    private let configValue: Qwen3_5Config;
    private let visionConfigValue: Qwen3_5VisionConfig?;
    private let requiredFiles: Dictionary<String, ValidatedRequiredFile>;
    private let shardIndexValue: Qwen3_5ShardIndex;
    private let totalPayloadBytesValue: UInt64;
    private let hasSeparateVisionSidecarValue: Bool;
    private let hasValidatedVisionTowerValue: Bool;
    private let tensorInventoryValue: TensorInventory;
    private var safetensorsSources: Dictionary<TensorSourceId, ValidatedSafetensorsSource>;
    private let sourceIdByFileName: Dictionary<String, TensorSourceId>;
    private let modelIdValue: String;
    private let revisionValue: String;
    private let maxOutputTokensValue: UInt32;

    public init(
        config: Qwen3_5Config, visionConfig: Qwen3_5VisionConfig?,
        requiredFiles: Dictionary<String, ValidatedRequiredFile>, shardIndex: Qwen3_5ShardIndex,
        totalPayloadBytes: UInt64, hasSeparateVisionSidecar: Bool, hasValidatedVisionTower: Bool,
        tensorInventory: TensorInventory,
        safetensorsSources: Dictionary<TensorSourceId, ValidatedSafetensorsSource>,
        sourceIdByFileName: Dictionary<String, TensorSourceId>, modelId: String,
        revision: String, maxOutputTokens: UInt32) {
        self.configValue = config;
        self.visionConfigValue = visionConfig;
        self.requiredFiles = requiredFiles;
        self.shardIndexValue = shardIndex;
        self.totalPayloadBytesValue = totalPayloadBytes;
        self.hasSeparateVisionSidecarValue = hasSeparateVisionSidecar;
        self.hasValidatedVisionTowerValue = hasValidatedVisionTower;
        self.tensorInventoryValue = tensorInventory;
        self.safetensorsSources = safetensorsSources;
        self.sourceIdByFileName = sourceIdByFileName;
        self.modelIdValue = modelId;
        self.revisionValue = revision;
        self.maxOutputTokensValue = maxOutputTokens;
    }

    public func config() -> Qwen3_5Config {
        return self.configValue;
    }

    public func visionConfig() -> Qwen3_5VisionConfig? {
        return self.visionConfigValue;
    }

    public func supportsImageInput() -> Bool {
        return self.hasValidatedVisionTowerValue;
    }

    public func shardIndex() -> Qwen3_5ShardIndex {
        return self.shardIndexValue;
    }

    public func hasSeparateVisionSidecar() -> Bool {
        return self.hasSeparateVisionSidecarValue;
    }

    public func tensorInventory() -> TensorInventory {
        return self.tensorInventoryValue;
    }

    public func modelId() -> String {
        return self.modelIdValue;
    }

    public func revision() -> String {
        return self.revisionValue;
    }

    public func maxOutputTokens() -> UInt32 {
        return self.maxOutputTokensValue;
    }

    public func shardCount() -> Int {
        return self.shardIndexValue.shardCount();
    }

    public func totalPayloadBytes() -> UInt64 {
        return self.totalPayloadBytesValue;
    }

    /// Sums actual payload byte ranges for a validated set of canonical
    /// tensor names; `nil` means a requested profile is missing or the
    /// arithmetic overflowed.
    public func payloadByteCount(canonicalTensorNames: Set<String>) -> UInt64? {
        var totalPayloadBytes: UInt64 = 0;
        for canonicalTensorName: String in canonicalTensorNames {
            guard let tensorLocation: TensorLocation = self.tensorInventoryValue
                .location(canonicalName: canonicalTensorName),
                let safetensorsSource: ValidatedSafetensorsSource = self.safetensorsSources[
                    tensorLocation.sourceId],
                let tensorView: SafetensorsFraming.TensorView = safetensorsSource
                    .storedTensorView(storedName: tensorLocation.storedName) else {
                return nil;
            }
            let (tensorPayloadBytes, tensorLengthOverflowed) = tensorView.dataEndOffset()
                .subtractingReportingOverflow(tensorView.dataStartOffset());
            if tensorLengthOverflowed {
                return nil;
            }
            let (summedPayloadBytes, totalOverflowed) = totalPayloadBytes
                .addingReportingOverflow(tensorPayloadBytes);
            if totalOverflowed {
                return nil;
            }
            totalPayloadBytes = summedPayloadBytes;
        }
        return totalPayloadBytes;
    }

    /// Returns routed-expert payload bytes and the largest gate/up fusion
    /// transient for a validated tensor-name set without reading payload
    /// bytes. Header ranges preserve mixed-bit OptiQ storage exactly.
    public func sparseExpertPayloadFootprint(
        canonicalTensorNames: Set<String>
    ) -> (totalPayloadBytes: UInt64, largestGateUpFusionTransientBytes: UInt64) {
        var payloadBytesByLayer: Dictionary<UInt32, UInt64> = Dictionary();
        var gateUpPayloadBytesByLayer: Dictionary<UInt32, UInt64> = Dictionary();
        for canonicalTensorName: String in canonicalTensorNames {
            guard Qwen3_5MoeTensorSpec.isSparseSelectedExpertTensorName(
                tensorName: canonicalTensorName),
                let tensorLocation: TensorLocation = self.tensorInventoryValue
                    .location(canonicalName: canonicalTensorName) else {
                return (UInt64.max, UInt64.max);
            }
            let layerIndex: UInt32? = ValidatedQwen35Artifact
                .sparseExpertLayerIndex(tensorName: canonicalTensorName);
            guard let layerIndex: UInt32 = layerIndex,
                let shardSource: ValidatedSafetensorsSource = self.safetensorsSources[
                    tensorLocation.sourceId],
                let tensorView: SafetensorsFraming.TensorView = shardSource.storedTensorView(
                    storedName: tensorLocation.storedName) else {
                return (UInt64.max, UInt64.max);
            }
            let (tensorPayloadBytes, tensorLengthOverflowed) = tensorView.dataEndOffset()
                .subtractingReportingOverflow(tensorView.dataStartOffset());
            if tensorLengthOverflowed {
                return (UInt64.max, UInt64.max);
            }
            let currentLayerBytes: UInt64 = payloadBytesByLayer[layerIndex] ?? 0;
            let (layerPayloadBytes, layerOverflowed) = currentLayerBytes
                .addingReportingOverflow(tensorPayloadBytes);
            if layerOverflowed {
                return (UInt64.max, UInt64.max);
            }
            payloadBytesByLayer[layerIndex] = layerPayloadBytes;
            if ValidatedQwen35Artifact.isGateOrUpProjectionTensor(
                tensorName: tensorLocation.canonicalName) {
                let currentGateUpBytes: UInt64 = gateUpPayloadBytesByLayer[layerIndex] ?? 0;
                let (gateUpPayloadBytes, gateUpOverflowed) = currentGateUpBytes
                    .addingReportingOverflow(tensorPayloadBytes);
                if gateUpOverflowed {
                    return (UInt64.max, UInt64.max);
                }
                gateUpPayloadBytesByLayer[layerIndex] = gateUpPayloadBytes;
            }
        }
        var totalExpertPayloadBytes: UInt64 = 0;
        for layerPayloadBytes: UInt64 in payloadBytesByLayer.values {
            let (summedPayloadBytes, totalOverflowed) = totalExpertPayloadBytes
                .addingReportingOverflow(layerPayloadBytes);
            if totalOverflowed {
                return (UInt64.max, UInt64.max);
            }
            totalExpertPayloadBytes = summedPayloadBytes;
        }
        let largestGateUpFusionTransientBytes: UInt64 = gateUpPayloadBytesByLayer.values
            .max() ?? 0;
        return (totalExpertPayloadBytes, largestGateUpFusionTransientBytes);
    }

    public func tokenizerBytes() -> Data? {
        return self.requiredFiles["tokenizer.json"]?.capturedBytes;
    }

    /// The validated `config.json` bytes captured during validation; the
    /// runtime uses them to construct the upstream model configuration
    /// without reopening the mutable pathname.
    public func configBytes() -> Data? {
        return self.requiredFiles["config.json"]?.capturedBytes;
    }

    public func generationConfigBytes() -> Data? {
        return self.requiredFiles["generation_config.json"]?.capturedBytes;
    }

    /// Resolves validated source IDs once while architecture-specific file
    /// groupings are intact.
    public func sourceIdsForFileNames(
        fileNames: Array<String>) throws -> Array<TensorSourceId> {
        return try fileNames.map { (fileName: String) -> TensorSourceId in
            return try self.sourceIdForFileName(fileName);
        };
    }

    /// Resolves one architecture declaration to the opaque source used by
    /// all later transfers.
    public func sourceIdForFileName(_ fileName: String) throws -> TensorSourceId {
        guard let sourceId: TensorSourceId = self.sourceIdByFileName[fileName] else {
            throw Qwen35ArtifactValidationError.artifact(.profileMissingRequiredFile(fileName: fileName));
        }
        return sourceId;
    }

    /// Transfers each already-open SafeTensors owner exactly once by opaque
    /// validated identity.
    public func takeSafetensorsSources(
        _ sourceIds: Array<TensorSourceId>) throws -> Array<ValidatedWeightsFile> {
        return try sourceIds.map { (sourceId: TensorSourceId) -> ValidatedWeightsFile in
            return try self.takeSafetensorsSource(sourceId);
        };
    }

    /// Transfers one source without reopening a model-directory path or
    /// reparsing its header.
    public func takeSafetensorsSource(
        _ sourceId: TensorSourceId) throws -> ValidatedWeightsFile {
        guard let source: ValidatedSafetensorsSource = self.safetensorsSources.removeValue(forKey: sourceId) else {
            let missingFileName: String = self.sourceIdByFileName
                .first(where: { (_: String, candidateSourceId: TensorSourceId) -> Bool in
                    return candidateSourceId == sourceId;
                })?.key ?? "unresolved SafeTensors source";
            throw Qwen35ArtifactValidationError.artifact(.profileMissingRequiredFile(fileName: missingFileName));
        }
        return try source.intoValidatedWeightsFile();
    }

    private static func sparseExpertLayerIndex(tensorName: String) -> UInt32? {
        let tensorNameComponents: Array<Substring> = tensorName.split(separator: ".");
        guard let layerMarkerIndex: Int = tensorNameComponents.firstIndex(of: "layers"),
            tensorNameComponents.indices.contains(layerMarkerIndex + 1) else {
            return nil;
        }
        return UInt32(tensorNameComponents[layerMarkerIndex + 1]);
    }

    private static func isGateOrUpProjectionTensor(tensorName: String) -> Bool {
        return tensorName.contains(".switch_mlp.gate_proj.")
            || tensorName.contains(".switch_mlp.up_proj.");
    }
}
