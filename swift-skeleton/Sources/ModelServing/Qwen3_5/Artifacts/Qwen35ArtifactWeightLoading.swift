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
        let upstreamConfiguration: Qwen35Configuration = try Qwen35ArtifactWeightLoading
            .decodeUpstreamConfiguration(
                configBytes: Qwen35ArtifactWeightLoading.validatedConfigBytes(validatedArtifact));
        let shardTensors: Dictionary<String, MLXArray> = try Qwen35ArtifactWeightLoading
            .loadShardTensors(validatedArtifact: validatedArtifact, attributionEnabled: attributionEnabled);
        return try Qwen35ArtifactWeightLoading.bindWeights(
            model: Qwen35Model(upstreamConfiguration),
            repositoryConfiguration: validatedArtifact.config(),
            shardTensors: shardTensors,
            attributionEnabled: attributionEnabled);
    }

    /// Loads and binds the validated artifact into the upstream MoE model.
    ///
    /// The MoE subclass only overrides `sanitize`, so the shard streaming,
    /// quantization coverage, and verified bind are exactly the dense
    /// loader's; the subclass constructor is the sole difference. When
    /// `bindRoutedExperts` is false the routed expert tensors are never
    /// bound or evaluated — the paged runtime replaces those modules before
    /// any evaluation, so their payload bytes are never read from storage.
    public static func loadArtifactBoundMoeModel(
        validatedArtifact: ValidatedQwen35Artifact,
        attributionEnabled: Bool,
        bindRoutedExperts: Bool = true
    ) throws -> Qwen35MoEModel {
        let upstreamConfiguration: Qwen35Configuration = try Qwen35ArtifactWeightLoading
            .decodeUpstreamConfiguration(
                configBytes: Qwen35ArtifactWeightLoading.validatedConfigBytes(validatedArtifact));
        let shardTensors: Dictionary<String, MLXArray> = try Qwen35ArtifactWeightLoading
            .loadShardTensors(validatedArtifact: validatedArtifact, attributionEnabled: attributionEnabled);
        let boundModel: Qwen35Model = try Qwen35ArtifactWeightLoading.bindWeights(
            model: Qwen35MoEModel(upstreamConfiguration),
            repositoryConfiguration: validatedArtifact.config(),
            shardTensors: shardTensors,
            attributionEnabled: attributionEnabled,
            bindRoutedExperts: bindRoutedExperts);
        guard let moeModel: Qwen35MoEModel = boundModel as? Qwen35MoEModel else {
            throw InferenceEngineError.modelLoad(
                reason: "the MoE artifact did not bind onto the MoE module tree");
        }
        return moeModel;
    }

    private static func validatedConfigBytes(
        _ validatedArtifact: ValidatedQwen35Artifact
    ) throws -> Data {
        guard let configBytes: Data = validatedArtifact.configBytes() else {
            throw InferenceEngineError.modelLoad(
                reason: "the validated artifact carries no captured configuration bytes");
        }
        return configBytes;
    }

    private static func decodeUpstreamConfiguration(configBytes: Data) throws -> Qwen35Configuration {
        do {
            return try JSONDecoder().decode(Qwen35Configuration.self, from: configBytes);
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the model configuration could not be decoded");
        }
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
            let shardParseStart: ContinuousClock.Instant? = ServingPerformanceAttribution
                .startedOperation(
                    operationName: "qwen35_artifact_shard_array_parse",
                    attributionEnabled: attributionEnabled);
            defer {
                ServingPerformanceAttribution.endedOperation(
                    operationName: "qwen35_artifact_shard_array_parse",
                    operationStart: shardParseStart,
                    attributionEnabled: attributionEnabled);
            }
            let shardTensors: Dictionary<String, MLXArray> = try Qwen35ArtifactWeightLoading
                .readMappedShardTensors(
                    shardWeightsFile: shardWeightsFile,
                    attributionEnabled: attributionEnabled);
            // Shards merge in index order; a later shard overwrites a
            // duplicate name, matching the serial loader.
            mergedTensors.merge(shardTensors) { (_: MLXArray, nextTensor: MLXArray) -> MLXArray in
                return nextTensor;
            };
        }
        return mergedTensors;
    }

    /// Memory-maps one validated descriptor and parses its tensors.
    ///
    /// The arrays stay lazy: each tensor's load primitive retains the reader,
    /// which retains the mapping, so the mapping lives exactly as long as
    /// the arrays over it. Materialization happens once at the verified
    /// bind's final evaluation — and a paged bind that drops expert tensors
    /// never reads their payload bytes at all.
    private static func readMappedShardTensors(
        shardWeightsFile: ValidatedWeightsFile,
        attributionEnabled: Bool
    ) throws -> Dictionary<String, MLXArray> {
        let shardArrayLoadStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_artifact_shard_array_load",
                attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_shard_array_load",
                operationStart: shardArrayLoadStart,
                attributionEnabled: attributionEnabled);
        }
        let shardByteCount: UInt64 = shardWeightsFile.sizeBytes;
        guard shardByteCount > 0, shardByteCount <= UInt64(Int.max) else {
            throw InferenceEngineError.modelLoad(
                reason: "the validated shard \(shardWeightsFile.validatedRequiredFile.fileName) has an unsupported size");
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
        return shardTensors;
    }

    /// Leaves a bound model in evaluation mode — the upstream loader
    /// contract (`ModelFactory.train(false)`). `Module.training` defaults to
    /// true, and the gated-delta forward dispatches its fused inference
    /// kernel only outside training mode (`useKernel: !training`), so a
    /// model left in the default mode silently serves the serial
    /// gradient-oriented recurrence through every linear-attention layer.
    static func finalizeInEvaluationMode(_ model: Qwen35Model) -> Void {
        model.train(false);
    }

    /// Sanitizes, quantizes, binds, and prepares the upstream model. A
    /// paged bind (`bindRoutedExperts: false`) omits the routed expert
    /// tensors entirely: the upstream expert modules keep their
    /// never-evaluated random initialization until the paged runtime
    /// replaces them, and quantization setup skips those placeholder
    /// modules. Core evaluation is deferred until after that replacement.
    private static func bindWeights(
        model: Qwen35Model,
        repositoryConfiguration: Qwen3_5Config,
        shardTensors: Dictionary<String, MLXArray>,
        attributionEnabled: Bool,
        bindRoutedExperts: Bool = true
    ) throws -> Qwen35Model {
        let bindStart: ContinuousClock.Instant? = ServingPerformanceAttribution.startedOperation(
            operationName: "qwen35_artifact_weight_bind", attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_weight_bind",
                operationStart: bindStart, attributionEnabled: attributionEnabled);
        }
        let boundModel: Qwen35Model = model;
        let sanitizeStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_artifact_weight_sanitize",
                attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_weight_sanitize",
                operationStart: sanitizeStart,
                attributionEnabled: attributionEnabled);
        }
        let sanitizedTensors: Dictionary<String, MLXArray> = boundModel.sanitize(weights: shardTensors);
        let bindableTensors: Dictionary<String, MLXArray> = bindRoutedExperts
            ? sanitizedTensors
            : sanitizedTensors.filter { (tensorName: String, _: MLXArray) -> Bool in
                return Qwen3_5MoeTensorSpec.isSparseSelectedExpertTensorName(tensorName: tensorName) == false;
            };
        try Qwen35ArtifactWeightLoading.validateQuantizationCoverage(
            repositoryConfiguration: repositoryConfiguration,
            sanitizedTensors: bindableTensors);
        let quantizationSetupStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_artifact_quantization_setup",
                attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_quantization_setup",
                operationStart: quantizationSetupStart,
                attributionEnabled: attributionEnabled);
        }
        MLXNN.quantize(model: boundModel) { (modulePath: String, _: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
            if bindRoutedExperts == false
                && Qwen3_5MoeTensorSpec.isSparseSelectedExpertTensorName(
                    tensorName: modulePath + ".weight") {
                return nil;
            }
            return Qwen35ArtifactWeightLoading.quantizationTupleForModule(
                repositoryConfiguration: repositoryConfiguration,
                sanitizedTensors: bindableTensors,
                modulePath: modulePath);
        };
        let parameterBindStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_artifact_parameter_bind",
                attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_parameter_bind",
                operationStart: parameterBindStart,
                attributionEnabled: attributionEnabled);
        }
        do {
            try boundModel.update(
                parameters: ModuleParameters.unflattened(bindableTensors),
                verify: bindRoutedExperts ? [.all] : [.noUnusedKeys, .shapeMismatch]);
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the validated tensors did not bind onto the module tree");
        }
        let prepareStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_artifact_model_prepare",
                attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_model_prepare",
                operationStart: prepareStart,
                attributionEnabled: attributionEnabled);
        }
        do {
            try boundModel.prepare();
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the model's derived inference state could not be prepared");
        }
        Qwen35ArtifactWeightLoading.finalizeInEvaluationMode(boundModel);
        try Qwen35ArtifactWeightLoading.installLastPositionLogitsTrimming(model: boundModel);
        if bindRoutedExperts {
            // Resident gate/up fusion stays parked: the fused gathered
            // projection is about 15% faster on a warmed block but regressed
            // the full-model gate in every measured run (1,363 / 1,544 /
            // 1,520 versus 2,006 tokens/second unfused), so production keeps
            // the upstream expert path until the end-to-end cause is found.
            // Qwen35ResidentFusionInstall.install(
            //     model: boundModel, configuration: repositoryConfiguration,
            //     attributionEnabled: attributionEnabled);
        }
        let materializeStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: bindRoutedExperts
                    ? "qwen35_artifact_resident_materialization"
                    : "qwen35_artifact_core_materialization",
                attributionEnabled: attributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: bindRoutedExperts
                    ? "qwen35_artifact_resident_materialization"
                    : "qwen35_artifact_core_materialization",
                operationStart: materializeStart,
                attributionEnabled: attributionEnabled);
        }
        if bindRoutedExperts {
            // Derived state and bound parameters leave the loader fully
            // materialized; forward passes stay read-only.
            MLX.eval(boundModel);
        }
        return boundModel;
    }

    /// Materializes only resident model parameters after paging replaces the
    /// upstream MoE modules. Their random expert placeholders must never be
    /// evaluated or they recreate the entire discarded expert payload.
    public static func materializePagedMoeCore(
        _ moeModel: Qwen35MoEModel,
        attributionEnabled: Bool
    ) -> Void {
        let materializeStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_artifact_core_materialization",
                attributionEnabled: attributionEnabled);
        let residentParameters: Array<MLXArray> = moeModel.parameters()
            .flattened()
            .filter({ (parameterPath: String, _: MLXArray) -> Bool in
                return Qwen3_5MoeTensorSpec.isSparseSelectedExpertTensorName(
                    tensorName: parameterPath) == false;
            })
            .map({ (_: String, parameterArray: MLXArray) -> MLXArray in
                return parameterArray;
            });
        MLX.eval(residentParameters);
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_artifact_core_materialization",
            operationStart: materializeStart,
            attributionEnabled: attributionEnabled);
    }

    /// Replaces the bound model's lm_head with the last-position-trimming
    /// wrapper, matching the Rust forward graph's vocabulary-logits rule.
    /// A tied-embedding model exposes no lm_head module and keeps its full
    /// projection unchanged.
    public static func installLastPositionLogitsTrimming(model: Qwen35Model) throws -> Void {
        for (modulePath, childModule) in model.namedModules() {
            guard modulePath.hasSuffix(".lm_head") else {
                continue;
            }
            guard let linearHead: Linear = childModule as? Linear else {
                throw InferenceEngineError.modelLoad(
                    reason: "the lm_head at \(modulePath) is not a Linear projection");
            }
            try model.update(
                modules: ModuleChildren.unflattened([(
                    modulePath,
                    Qwen35LastPositionLogitsHead(linearHead) as Module)]),
                verify: []);
            return;
        }
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
