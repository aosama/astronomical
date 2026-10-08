import Foundation;

import MLX;
import MLXLMCommon;
import MLXNN;

/// The paged-expert installation seam of `Qwen35MoeEngine`: locates every
/// MoE layer's upstream expert primitive, swaps it for the paged primitive
/// whose projections quantize exactly as the artifact bound them, installs
/// the plan's retained payload, and exposes the introspection seams the
/// hermetic paging journeys seed their page sources from.
extension Qwen35MoeEngine {

    /**
     * Installs paged expert execution over the loaded model: every MoE
     * layer's upstream SwitchGLU is swapped for the paged primitive that
     * materializes routed-but-missing experts through the given page
     * source, and the plan's retained experts are installed at startup as
     * the layer's resident payload. Must run after `loadInMemoryModel` and
     * before the first generation; a resident install stays untouched when
     * paging is never requested.
     */
    public func installPagedExpertExecution(
        retainedExpertIdsPerLayer: Array<Array<Int>>,
        expertPageMaterializer: any Qwen35MoeExpertPageMaterializing
    ) throws {
        guard let moeModel = self.moeModel, let repositoryConfiguration = self.moeRepositoryConfiguration
        else {
            throw InferenceEngineError.modelLoad(reason: "no MoE model is loaded");
        }
        let pagedResidency: Qwen35MoePagedExpertResidency = try Qwen35MoePagedExpertResidency(
            repositoryConfiguration: repositoryConfiguration,
            retainedExpertIdsPerLayer: retainedExpertIdsPerLayer);
        let forwardContext: Qwen35MoePagedForwardContext = Qwen35MoePagedForwardContext(
            currentInputTokenId: 0);
        let pagingDecorator: Qwen35MoePagedExpertDecorator = Qwen35MoePagedExpertDecorator(
            expertPageMaterializer: expertPageMaterializer,
            routeObservationRing: RouteObservationRing(
                capacity: RouteObservationRing.defaultObservationCapacity),
            attributionEnabled: self.attributionEnabled);

        let switchMluPathsByLayerIndex: Array<String> = try
            Qwen35MoeEngine.switchGluInstallations(
                model: moeModel, expectedLayerCount: Int(pagedResidency.totalLayerCount));
        // Paged decode mutates the switch_mlp modules on every step
        // (materialization), and a compiled trace freezes the module graph it
        // was built from — upstream invalidates traces on module replacement,
        // and the re-trace would capture the decorator and abort on its
        // in-transform eval. Disabling MLX compile process-wide makes every
        // CompiledTrace evaluate eagerly (the gate sits at trace-build time
        // and CompiledTrace compiles lazily on first call), which is the only
        // execution mode that honors per-step module mutation. The SSM cache
        // marker in startGeneration handles the model-level decode schedule;
        // this handles the per-block MoE trace. Resident execution never
        // reaches here.
        MLX.compile(enable: false);
        var pagedSwitchGlusByLayerIndexByPath: Array<(String, Module)> = [];
        for (layerIndex, retainedExpertIds) in retainedExpertIdsPerLayer.enumerated() {
            let switchMluPath: String = switchMluPathsByLayerIndex[layerIndex];
            let pagedSwitchGlu: Qwen35MoePagedSwitchGLU = Qwen35MoePagedSwitchGLU(
                inputDims: Int(repositoryConfiguration.hiddenSize()),
                hiddenDims: Int(repositoryConfiguration.expertIntermediateSize()),
                numExperts: Int(repositoryConfiguration.expertCount()),
                decoderLayerIndex: layerIndex,
                pagingDecorator: pagingDecorator,
                forwardContext: forwardContext);
            // Quantize the primitive's projections exactly as the artifact
            // bound the resident module it replaces: an affine page carries
            // weight, scales, and biases slices, so the replaced module must
            // hold the same parameter names or the verified install fails
            // closed. The module path inside this standalone primitive is
            // the projection basename.
            MLXNN.quantize(model: pagedSwitchGlu) { (modulePath: String, _: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                let projectionModuleName: String =
                    "language_model.model.layers.\(layerIndex).mlp.switch_mlp.\(modulePath)";
                let projectionProfile: OptiQQuantizationProfile = repositoryConfiguration
                    .quantizationProfile(forModule: projectionModuleName);
                if projectionProfile.isUnquantized() {
                    return nil;
                }
                return (
                    groupSize: Int(projectionProfile.groupSize),
                    bits: Int(projectionProfile.bits),
                    mode: QuantizationMode.affine);
            };
            try pagingDecorator.installRetainedExperts(
                layerIndex: layerIndex,
                expertIds: retainedExpertIds,
                switchGlu: pagedSwitchGlu);
            pagedSwitchGlusByLayerIndexByPath.append((switchMluPath, pagedSwitchGlu));
        }
        // One update call over every layer at once: an unflattened single-layer
        // path leaves the sibling decoder-layer slots as `.none`, which the
        // upstream array traversal rejects as an unexpected structure.
        try moeModel.update(
            modules: ModuleChildren.unflattened(pagedSwitchGlusByLayerIndexByPath),
            verify: [.all]);

        self.pagedForwardContext = forwardContext;
        self.pagedExpertDecorator = pagingDecorator;
        self.expertResidency = pagedResidency;
    }

    /// Exposes one decoder layer's expert primitive parameter arrays by
    /// parameter basename — the introspection seam hermetic paging journeys
    /// use to seed a page source from a fully resident engine. A pre-install
    /// read observes the loaded resident weights.
    internal func switchGluParameterArrays(layerIndex: Int) throws -> Dictionary<String, MLXArray> {
        guard let moeModel = self.moeModel else {
            throw InferenceEngineError.modelLoad(reason: "no MoE model is loaded");
        }
        let module: Module = moeModel;
        for (modulePath, childModule) in module.namedModules() {
            guard modulePath.hasSuffix(".mlp.switch_mlp") else {
                continue;
            }
            let pathComponents: Array<String> = modulePath.split(separator: ".").map(String.init);
            // The decoder path ends `layers.<index>.mlp.switch_mlp`, so the
            // layer index sits three components from the end.
            guard pathComponents.count >= 3,
                pathComponents[pathComponents.count - 2] == "mlp",
                let componentLayerIndex: Int = Int(pathComponents[pathComponents.count - 3]),
                componentLayerIndex == layerIndex
            else {
                continue;
            }
            guard let switchGlu: SwitchGLU = childModule as? SwitchGLU else {
                throw InferenceEngineError.modelLoad(
                    reason: "the MoE layer primitive at \(modulePath) is not an expert SwitchGLU");
            }
            return Dictionary(uniqueKeysWithValues: switchGlu.parameters().flattened());
        }
        throw InferenceEngineError.modelLoad(
            reason: "the model exposes no MoE layer primitive for layer \(layerIndex)");
    }

    /// Locates every MoE layer's expert primitive by its structural path,
    /// in decoder order; a checkpoint whose MoE layer count disagrees with
    /// the residency plan fails closed instead of paging a partial install.
    /// The located upstream primitives are intentionally discarded — each
    /// is replaced by a fresh paged primitive whose weights come solely
    /// from the page source.
    private static func switchGluInstallations(
        model: any LanguageModel,
        expectedLayerCount: Int
    ) throws -> Array<String> {
        let module: Module = model;
        var switchMluPaths: Array<String> = [];
        for (modulePath, childModule) in module.namedModules() {
            guard modulePath.hasSuffix(".mlp.switch_mlp") else {
                continue;
            }
            guard childModule is SwitchGLU else {
                throw InferenceEngineError.modelLoad(
                    reason: "the MoE layer primitive at \(modulePath) is not an expert SwitchGLU");
            }
            switchMluPaths.append(modulePath);
        }
        guard switchMluPaths.count == expectedLayerCount else {
            throw InferenceEngineError.modelLoad(
                reason: "the model exposes \(switchMluPaths.count) MoE layers but the residency plan names \(expectedLayerCount)");
        }
        return switchMluPaths;
    }

}
