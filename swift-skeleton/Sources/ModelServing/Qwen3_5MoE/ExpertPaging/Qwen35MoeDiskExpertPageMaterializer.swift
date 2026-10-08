import Foundation

import MLX
import RuntimeIntegration

/**
 * The disk-backed resident-page seam: every routed-expert miss reads its
 * quantized slices straight from the model's SafeTensors shard ranges.
 * Port of the Rust bounded expert streaming path
 * (`rust_expert_streaming` + `paged_expert_weights`) in the shape the
 * migration's decision record fixed: shard ranges are the only source —
 * the experimental per-expert pack format is expunged and has no ported
 * branch here.
 *
 * Deliberate divergence from Rust: allocation admission against the
 * composed memory budget stays with the memory-governor track and is not
 * consulted inside this materializer; the decorator owns the read-once
 * property (it only asks for experts the layer does not already hold),
 * so one call is exactly one page read.
 */
public final class Qwen35MoeDiskExpertPageMaterializer {

    private static let AFFINE_PARAMETER_BASENAMES: [String] = ["weight", "scales", "biases"]
    private static let NATIVE_PARAMETER_BASENAMES: [String] = ["weight"]

    private let modelDirectory: URL
    private let layerPlans: [QuantizedExpertLayerPlan]
    private let expertFileReadMetrics: PositionalFileReadMetrics?
    private let attributionEnabled: Bool

    public init(
        modelDirectory: URL,
        layerPlans: [QuantizedExpertLayerPlan],
        expertFileReadMetrics: PositionalFileReadMetrics?,
        attributionEnabled: Bool
    ) {
        self.modelDirectory = modelDirectory
        self.layerPlans = layerPlans
        self.expertFileReadMetrics = expertFileReadMetrics
        self.attributionEnabled = attributionEnabled
    }

    /**
     * Materializes the quantized weight slices for exactly these experts
     * of one decoder layer, reading only what residency is missing.
     *
     * - Parameters:
     *   - layerIndex: The decoder layer whose experts are requested.
     *   - expertIds: The requested expert identifiers; validated and
     *     normalized to ascending unique order, which is also the key
     *     space of the returned per-expert slice maps.
     * - Returns: The materialized slices and the page-read count (one
     *   page per call).
     * - Throws: `ExpertPagingError.layerIndexOutOfRange` when the layer
     *   request lies outside the startup plans;
     *   `.manifestValidationFailure` when the expert ids or shard reads
     *   are invalid; `.pageTensorMissing` or `.pageTensorExpertCountMismatch`
     *   when the loaded page cannot serve the plan's full parameter set.
     */
    public func materializeExpertWeights(
        layerIndex: Int,
        expertIds: [Int]
    ) throws -> Qwen35MoeMaterializedExpertWeights {
        let materializeStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "rust_expert_streaming_layer_preparation",
                attributionEnabled: self.attributionEnabled)
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "rust_expert_streaming_layer_preparation",
                operationStart: materializeStart,
                attributionEnabled: self.attributionEnabled)
        }
        guard layerIndex >= 0, layerIndex < self.layerPlans.count else {
            throw ExpertPagingError.layerIndexOutOfRange(
                layerIndex: layerIndex, layerCount: self.layerPlans.count)
        }
        let layerPlan: QuantizedExpertLayerPlan = self.layerPlans[layerIndex]
        let pageManifest: QuantizedExpertPageManifest = try QuantizedExpertPageManifestBuilder
            .buildPageManifest(layerPlan: layerPlan, expertIds: expertIds)
        let loadedTensorsByName: [String: MLXArray] = try QuantizedExpertPageLoader.loadPage(
            modelDirectory: self.modelDirectory,
            pageManifest: pageManifest,
            expertFileReadMetrics: self.expertFileReadMetrics)
        let gateProjection: Qwen35MoeMaterializedProjectionSlices = try self.projectionSlices(
            projectionName: "gate_proj",
            layerPlan: layerPlan,
            pageManifest: pageManifest,
            loadedTensorsByName: loadedTensorsByName)
        let upProjection: Qwen35MoeMaterializedProjectionSlices = try self.projectionSlices(
            projectionName: "up_proj",
            layerPlan: layerPlan,
            pageManifest: pageManifest,
            loadedTensorsByName: loadedTensorsByName)
        let downProjection: Qwen35MoeMaterializedProjectionSlices = try self.projectionSlices(
            projectionName: "down_proj",
            layerPlan: layerPlan,
            pageManifest: pageManifest,
            loadedTensorsByName: loadedTensorsByName)
        return Qwen35MoeMaterializedExpertWeights(
            gateProjection: gateProjection,
            upProjection: upProjection,
            downProjection: downProjection,
            expertPageReadCount: 1)
    }

    private func projectionSlices(
        projectionName: String,
        layerPlan: QuantizedExpertLayerPlan,
        pageManifest: QuantizedExpertPageManifest,
        loadedTensorsByName: [String: MLXArray]
    ) throws -> Qwen35MoeMaterializedProjectionSlices {
        let projectionMode: ExpertLayerQuantizationMode = layerPlan
            .quantizationModeForProjection(projectionName: projectionName)
        let parameterBasenames: [String] = projectionMode == .nativeBfloat16
            ? Qwen35MoeDiskExpertPageMaterializer.NATIVE_PARAMETER_BASENAMES
            : Qwen35MoeDiskExpertPageMaterializer.AFFINE_PARAMETER_BASENAMES
        var slicesByParameterBasename: [String: [Int: MLXArray]] = [:]
        for parameterBasename: String in parameterBasenames {
            let pageTensorName: String = "\(projectionName).\(parameterBasename)"
            guard let pageTensor: MLXArray = loadedTensorsByName[pageTensorName] else {
                throw ExpertPagingError.pageTensorMissing(
                    tensorName: pageTensorName, layerPrefix: layerPlan.layerPrefix)
            }
            let seatedSlotCount: Int = pageManifest.expertIds.count
            guard pageTensor.dim(0) == seatedSlotCount else {
                throw ExpertPagingError.pageTensorExpertCountMismatch(
                    tensorName: pageTensorName,
                    expectedSlotCount: seatedSlotCount,
                    actualSlotCount: pageTensor.dim(0))
            }
            var expertSlicesByExpertId: [Int: MLXArray] = [:]
            expertSlicesByExpertId.reserveCapacity(seatedSlotCount)
            for (pageSlot, expertId): (Int, Int) in pageManifest.expertIds.enumerated() {
                expertSlicesByExpertId[expertId] = pageTensor[pageSlot]
            }
            slicesByParameterBasename[parameterBasename] = expertSlicesByExpertId
        }
        return Qwen35MoeMaterializedProjectionSlices(
            parametersByParameterBasename: slicesByParameterBasename)
    }
}
