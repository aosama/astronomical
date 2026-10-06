import Foundation;

import Darwin;
import MLX;
import MLXLLM;
import MLXNN;

/// Streams the validated Qwen3.5 artifact's model weights into the upstream
/// dense model.
///
/// Mirrors the load half of the Rust `Qwen3_5Weights` binding
/// (crates/model-serving/src/qwen3_5/model/weights.rs) on the upstream
/// mlx-swift-lm model: the shard index resolves each validated shard
/// descriptor, the descriptors are memory-mapped read-only and parsed with
/// the MLX safetensors reader, the upstream `sanitize` pass normalizes the
/// names onto the module tree, the config's OptiQ profiles drive per-module
/// affine quantization, and `update(parameters:verify: .all)` binds every
/// tensor with shape and coverage verification.
///
/// The Swift seam loads through the upstream model directly instead of a
/// hand-written tensor-by-tensor binding: the module tree, quantized
/// modules, and derived-state preparation are upstream concerns, and the
/// loader's job is ownership (validated descriptors, take-once) plus
/// attribution of each load phase.
public enum Qwen35ArtifactWeightLoading {

    /// Loads and binds every language-model weight from the validated
    /// artifact's shard descriptors into a prepared upstream model.
    ///
    /// The artifact is consumed: shard sources transfer exactly once, so a
    /// second load of the same validated artifact fails closed.
    public static func loadArtifactBoundModel(
        validatedArtifact: ValidatedQwen35Artifact,
        attributionEnabled: Bool
    ) throws -> Qwen35Model {
        guard let configBytes: Data = validatedArtifact.configBytes() else {
            throw InferenceEngineError.modelLoad(
                reason: "the validated artifact carries no captured configuration bytes");
        }
        let upstreamConfiguration: Qwen35Configuration;
        do {
            upstreamConfiguration = try JSONDecoder().decode(
                Qwen35Configuration.self, from: configBytes);
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model configuration could not be decoded");
        }

        let shardTensors: Dictionary<String, MLXArray> = try Qwen35ArtifactWeightLoading
            .loadShardTensors(validatedArtifact: validatedArtifact, attributionEnabled: attributionEnabled);

        let boundModel: Qwen35Model = try Qwen35ArtifactWeightLoading.bindWeights(
            upstreamConfiguration: upstreamConfiguration,
            repositoryConfiguration: validatedArtifact.config(),
            shardTensors: shardTensors,
            attributionEnabled: attributionEnabled);
        return boundModel;
    }

    /// Reads every indexed model shard through its validated descriptor.
    ///
    /// Each descriptor is memory-mapped private and read-only — the mapping
    /// is the descriptor's own bytes, so a file swapped on the mutable
    /// pathname after validation cannot reach the loader. Tensors are
    /// materialized inside the mapping's lifetime, then the mapping is
    /// released.
    private static func loadShardTensors(
        validatedArtifact: ValidatedQwen35Artifact,
        attributionEnabled: Bool
    ) throws -> Dictionary<String, MLXArray> {
        let shardReadStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_artifact_shard_read", attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_shard_read",
                operationStart: shardReadStart, attributionEnabled: attributionEnabled);
        }
        let shardFileNames: Array<String> = validatedArtifact.shardIndex().modelShardFileNames();
        guard shardFileNames.isEmpty == false else {
            throw InferenceEngineError.modelLoad(
                reason: "the validated artifact names no model shards");
        }
        let shardSourceIds: Array<TensorSourceId> = try validatedArtifact
            .sourceIdsForFileNames(fileNames: shardFileNames);
        let shardWeightsFiles: Array<ValidatedWeightsFile>;
        do {
            shardWeightsFiles = try validatedArtifact.takeSafetensorsSources(shardSourceIds);
        } catch {
            // Take-once ownership: sources already consumed by a prior load
            // surface as the engine's typed load failure, never as the
            // validator's internal error type.
            throw InferenceEngineError.modelLoad(
                reason: "the validated artifact no longer holds its model shards");
        }

        var mergedTensors: Dictionary<String, MLXArray> = Dictionary();
        for shardWeightsFile: ValidatedWeightsFile in shardWeightsFiles {
            let shardTensors: Dictionary<String, MLXArray> = try Qwen35ArtifactWeightLoading
                .readMappedShardTensors(shardWeightsFile: shardWeightsFile);
            // Shards merge in index order; a later shard overwrites a
            // duplicate name, matching the serial loader.
            mergedTensors.merge(shardTensors) { (_: MLXArray, nextTensor: MLXArray) -> MLXArray in
                return nextTensor;
            };
        }
        return mergedTensors;
    }

    /// Memory-maps one validated descriptor and parses its tensors.
    private static func readMappedShardTensors(
        shardWeightsFile: ValidatedWeightsFile
    ) throws -> Dictionary<String, MLXArray> {
        let shardByteCount: UInt64 = shardWeightsFile.sizeBytes;
        guard shardByteCount > 0 else {
            throw InferenceEngineError.modelLoad(
                reason: "the validated shard \(shardWeightsFile.validatedRequiredFile.fileName) is empty");
        }
        let fileDescriptor: Int32 = shardWeightsFile.intoFile().fileDescriptor;
        guard let mappedBytes: UnsafeMutableRawPointer = mmap(
            nil, Int(shardByteCount), PROT_READ, MAP_PRIVATE, fileDescriptor, 0),
            mappedBytes != UnsafeMutableRawPointer(bitPattern: -1) else {
            throw InferenceEngineError.modelLoad(
                reason: "the validated shard \(shardWeightsFile.validatedRequiredFile.fileName) could not be mapped");
        }
        let mappedByteCount: Int = Int(shardByteCount);
        let mappedData: Data = Data(
            bytesNoCopy: mappedBytes, count: mappedByteCount,
            deallocator: .custom { (_: UnsafeMutableRawPointer, _: Int) -> Void in
                _ = munmap(mappedBytes, mappedByteCount);
            });
        let shardTensors: Dictionary<String, MLXArray>;
        do {
            (shardTensors, _) = try MLX.loadArraysAndMetadata(data: mappedData);
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the validated shard \(shardWeightsFile.validatedRequiredFile.fileName) could not be parsed");
        }
        // Materialize every tensor while the mapping is alive so the arrays
        // own their bytes before the descriptor mapping is released.
        MLX.eval(Array(shardTensors.values));
        return shardTensors;
    }

    /// Sanitizes, quantizes, binds, and prepares the upstream model.
    private static func bindWeights(
        upstreamConfiguration: Qwen35Configuration,
        repositoryConfiguration: Qwen3_5Config,
        shardTensors: Dictionary<String, MLXArray>,
        attributionEnabled: Bool
    ) throws -> Qwen35Model {
        let bindStart: ContinuousClock.Instant? = ServingPerformanceAttribution.startedOperation(
            operationName: "qwen35_artifact_weight_bind", attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_weight_bind",
                operationStart: bindStart, attributionEnabled: attributionEnabled);
        }
        let boundModel: Qwen35Model = Qwen35Model(upstreamConfiguration);
        let sanitizedTensors: Dictionary<String, MLXArray> = boundModel.sanitize(weights: shardTensors);
        try Qwen35ArtifactWeightLoading.validateQuantizationCoverage(
            repositoryConfiguration: repositoryConfiguration,
            sanitizedTensors: sanitizedTensors);
        MLXNN.quantize(model: boundModel) { (modulePath: String, _: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
            return Qwen35ArtifactWeightLoading.quantizationTupleForModule(
                repositoryConfiguration: repositoryConfiguration,
                sanitizedTensors: sanitizedTensors,
                modulePath: modulePath);
        };
        do {
            try boundModel.update(
                parameters: ModuleParameters.unflattened(sanitizedTensors), verify: [.all]);
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the validated tensors did not bind onto the dense module tree");
        }
        do {
            try boundModel.prepare();
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model's derived inference state could not be prepared");
        }
        // Derived state and bound parameters leave the loader fully
        // materialized; forward passes stay read-only.
        MLX.eval(boundModel);
        return boundModel;
    }

    /// Fails closed when a shard supplies scales for a module the config
    /// left unquantized (or vice versa): the quantize callback cannot throw,
    /// so coverage is proven before any module is replaced. Only the
    /// scales-present direction is checked here: the complementary direction
    /// (config-quantized module with no shard scales) already fails closed
    /// during validation, which resolves every profile from the shard
    /// inventory before this loader runs.
    private static func validateQuantizationCoverage(
        repositoryConfiguration: Qwen3_5Config,
        sanitizedTensors: Dictionary<String, MLXArray>
    ) throws -> Void {
        for tensorName: String in sanitizedTensors.keys where tensorName.hasSuffix(".scales") {
            let moduleName: String = String(tensorName.dropLast(".scales".count));
            let moduleProfile: OptiQQuantizationProfile? = repositoryConfiguration
                .quantizedModuleProfiles().profile(forKey: moduleName);
            if moduleProfile == nil || moduleProfile?.isUnquantized() == true {
                throw InferenceEngineError.modelLoad(
                    reason: "the artifact quantized \(moduleName) but the configuration did not");
            }
        }
    }

    /// Resolves one module's affine quantization tuple from the shard
    /// evidence and the config profile: a module is stored quantized when
    /// the shard carries scales for it and the config profile is quantized.
    private static func quantizationTupleForModule(
        repositoryConfiguration: Qwen3_5Config,
        sanitizedTensors: Dictionary<String, MLXArray>,
        modulePath: String
    ) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? {
        guard sanitizedTensors["\(modulePath).scales"] != nil,
            let moduleProfile: OptiQQuantizationProfile = repositoryConfiguration
                .quantizedModuleProfiles().profile(forKey: modulePath),
            moduleProfile.isUnquantized() == false else {
            return nil;
        }
        return (
            groupSize: Int(moduleProfile.groupSize),
            bits: Int(moduleProfile.bits),
            mode: QuantizationMode.affine);
    }
}
