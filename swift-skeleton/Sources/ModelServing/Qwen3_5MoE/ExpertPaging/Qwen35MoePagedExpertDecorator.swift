import Foundation

import MLX
import MLXLMCommon
import MLXNN

/**
 * Residency-aware decoration of one MoE layer's upstream SwitchGLU.
 *
 * A decorated layer forward materializes ONLY the experts the routing
 * indices select — through the page-materializer seam, never by touching
 * disk directly — installs their quantized slices into the layer's SwitchGLU,
 * then delegates to the untouched upstream execution so both the unsorted
 * and the gather/sort dispatch paths keep their upstream behavior. Every
 * forward records its true route as a route observation and accumulates
 * page-read evidence, which gives the issue #629 read-once property: a
 * repeated identical forward finds its experts installed and reads no more
 * pages than the first. Paging timings feed the switchable performance
 * attribution log on the serving critical path.
 */
public final class Qwen35MoePagedExpertDecorator {

    /// The residency seam consulted for experts missing from the layer.
    private let expertPageMaterializer: Qwen35MoeExpertPageMaterializing

    /// Bounded route-observation history fed with every decorated forward.
    public let routeObservationRing: RouteObservationRing

    /// Switchable attribution configured from the serving configuration.
    private let attributionEnabled: Bool

    /// Expert pages the backing store has read on behalf of this decorator.
    public private(set) var totalExpertPageReadCount: UInt64

    /// Experts whose quantized slices each layer's SwitchGLU already holds.
    private var installedExpertIdsByLayer: [Int: Set<Int>]

    public init(
        expertPageMaterializer: Qwen35MoeExpertPageMaterializing,
        routeObservationRing: RouteObservationRing,
        attributionEnabled: Bool
    ) {
        self.expertPageMaterializer = expertPageMaterializer
        self.routeObservationRing = routeObservationRing
        self.attributionEnabled = attributionEnabled
        self.totalExpertPageReadCount = 0
        self.installedExpertIdsByLayer = [:]
    }

    /**
     * Runs one paged layer forward: materialize missing routed experts,
     * install their slices, record the route observation, and delegate to
     * the upstream SwitchGLU.
     *
     * - Parameters:
     *   - layerIndex: The decoder layer this SwitchGLU belongs to.
     *   - inputTokenEmbeddings: The layer input shaped [tokens, inputDims].
     *   - routingIndices: The router selection shaped [tokens, topK].
     *   - switchGlu: The layer's upstream expert primitive.
     *   - inputTokenId: The token id whose forward produced the routes.
     * - Returns: The unweighted per-assignment expert outputs, shaped
     *   [tokens, topK, inputDims], exactly as the upstream returns them.
     * - Throws: `Qwen35MoePagedExpertDecoratorError` when the materialized
     *   slices target a parameter the SwitchGLU does not expose, or a
     *   materializer failure.
     */
    public func decoratedLayerOutputs(
        layerIndex: Int,
        inputTokenEmbeddings: MLXArray,
        routingIndices: MLXArray,
        switchGlu: SwitchGLU,
        inputTokenId: UInt32
    ) throws -> MLXArray {
        try self.prepareLayerForForward(
            layerIndex: layerIndex,
            inputTokenCount: inputTokenEmbeddings.dim(0),
            routingIndices: routingIndices,
            switchGlu: switchGlu,
            inputTokenId: inputTokenId);
        return switchGlu(inputTokenEmbeddings, routingIndices)
    }

    /**
     * Prepares one layer for a decorated forward: materialize the routed
     * experts missing from the layer, install their slices, and record the
     * route observation. The engine-facing paged SwitchGLU subclass calls
     * this before delegating to `super`, so the upstream weighting and
     * reduction paths — including the ones that bypass `callAsFunction` —
     * observe installed weights without this decorator wrapping their
     * dataflow.
     *
     * - Throws: `Qwen35MoePagedExpertDecoratorError` when the materialized
     *   slices target a parameter the SwitchGLU does not expose, or a
     *   materializer failure.
     */
    public func prepareLayerForForward(
        layerIndex: Int,
        inputTokenCount: Int,
        routingIndices: MLXArray,
        switchGlu: SwitchGLU,
        inputTokenId: UInt32
    ) throws {
        let forwardStart: ContinuousClock.Instant? = ServingPerformanceAttribution.startedOperation(
            operationName: "moe.paged_expert_layer_forward",
            attributionEnabled: self.attributionEnabled)
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "moe.paged_expert_layer_forward",
                operationStart: forwardStart,
                attributionEnabled: self.attributionEnabled)
        }

        let routingTokenCount: Int = routingIndices.dim(0)
        guard inputTokenCount == routingTokenCount else {
            throw Qwen35MoePagedExpertDecoratorError.mismatchedForwardShapes(
                tokenCount: inputTokenCount, routingTokenCount: routingTokenCount)
        }
        let routedExpertIds: [Int] = self.routedExpertIds(routingIndices: routingIndices)
        let missingExpertIds: [Int] = self.missingExpertIds(
            layerIndex: layerIndex, routedExpertIds: routedExpertIds)

        if missingExpertIds.isEmpty == false {
            try self.materializeAndInstallExpertWeights(
                layerIndex: layerIndex,
                missingExpertIds: missingExpertIds,
                switchGlu: switchGlu)
        }

        self.recordRouteObservation(
            layerIndex: layerIndex,
            routedExpertIds: routedExpertIds,
            inputTokenId: inputTokenId)
    }

    /**
     * Installs the plan's retained experts at paging setup: the same
     * materialize-and-install seam as a runtime miss, without recording a
     * route observation — retained pages are startup reads, not forwards.
     */
    public func installRetainedExperts(
        layerIndex: Int,
        expertIds: [Int],
        switchGlu: SwitchGLU
    ) throws {
        guard expertIds.isEmpty == false else {
            return;
        }
        try self.materializeAndInstallExpertWeights(
            layerIndex: layerIndex,
            missingExpertIds: expertIds,
            switchGlu: switchGlu)
    }

    /// The distinct routed expert identifiers in ascending order.
    private func routedExpertIds(routingIndices: MLXArray) -> [Int] {
        let flatRoutingIndices: [Int32] = routingIndices.asArray(Int32.self)
        let distinctExpertIds: Set<Int> = Set(flatRoutingIndices.map(
            { (expertIndex: Int32) -> Int in
                return Int(expertIndex)
            }))
        return distinctExpertIds.sorted()
    }

    /// Routed experts whose quantized slices the layer does not hold yet.
    private func missingExpertIds(
        layerIndex: Int,
        routedExpertIds: [Int]
    ) -> [Int] {
        let alreadyInstalledExpertIds: Set<Int> = self.installedExpertIdsByLayer[layerIndex] ?? []
        return routedExpertIds.filter({ (expertId: Int) -> Bool in
            return alreadyInstalledExpertIds.contains(expertId) == false
        })
    }

    /**
     * Consults the materializer for the missing experts and assembles their
     * slices into full parameter arrays. Non-routed experts stay zeroed so
     * an unselected expert leaking into execution surfaces as a nonzero
     * output difference instead of passing silently.
     */
    private func materializeAndInstallExpertWeights(
        layerIndex: Int,
        missingExpertIds: [Int],
        switchGlu: SwitchGLU
    ) throws {
        let materializeStart: ContinuousClock.Instant? = ServingPerformanceAttribution.startedOperation(
            operationName: "moe.paged_expert_materialize",
            attributionEnabled: self.attributionEnabled)

        let materializedWeights: Qwen35MoeMaterializedExpertWeights = try self.expertPageMaterializer
            .materializeExpertWeights(layerIndex: layerIndex, expertIds: missingExpertIds)
        self.totalExpertPageReadCount = SaturatingArithmetic.add(
            self.totalExpertPageReadCount, UInt64(materializedWeights.expertPageReadCount))

        try self.installMaterializedWeights(materializedWeights, into: switchGlu)
        var installedExpertIds: Set<Int> = self.installedExpertIdsByLayer[layerIndex] ?? []
        installedExpertIds.formUnion(missingExpertIds)
        self.installedExpertIdsByLayer[layerIndex] = installedExpertIds

        ServingPerformanceAttribution.endedOperation(
            operationName: "moe.paged_expert_materialize",
            operationStart: materializeStart,
            attributionEnabled: self.attributionEnabled)
    }

    /// Assembles each materialized projection parameter into a full array
    /// whose routed expert rows carry the served slices, then verifies the
    /// whole replacement into the SwitchGLU.
    private func installMaterializedWeights(
        _ materializedWeights: Qwen35MoeMaterializedExpertWeights,
        into switchGlu: SwitchGLU
    ) throws {
        let projectionSlicesByProjectionName: [String: Qwen35MoeMaterializedProjectionSlices] = [
            "gate_proj": materializedWeights.gateProjection,
            "up_proj": materializedWeights.upProjection,
            "down_proj": materializedWeights.downProjection,
        ]
        let currentParametersByName: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: switchGlu.parameters().flattened())

        var replacementParameters: [String: MLXArray] = [:]
        for (projectionName, projectionSlices) in projectionSlicesByProjectionName {
            for (parameterBasename, expertSlicesByExpertId) in projectionSlices.parametersByParameterBasename {
                let parameterName: String = "\(projectionName).\(parameterBasename)"
                guard let currentArray: MLXArray = currentParametersByName[parameterName] else {
                    throw Qwen35MoePagedExpertDecoratorError.missingLayerParameter(parameterName: parameterName)
                }
                // Assembly base carries over every already-installed expert's
                // rows: a runtime miss must only add experts, never wipe the
                // retained payload installed by an earlier materialization.
                // Experts nobody has served yet stay zeroed, so an unselected
                // expert leaking into execution still surfaces as a nonzero
                // output difference instead of passing silently.
                let assembledArray: MLXArray = currentArray
                for (expertId, expertSlice) in expertSlicesByExpertId {
                    assembledArray[expertId] = expertSlice
                }
                replacementParameters[parameterName] = assembledArray
            }
        }

        try switchGlu.update(
            parameters: ModuleParameters.unflattened(replacementParameters), verify: [.all])
    }

    /// Records this forward's true route at its layer position; earlier
    /// layers are recorded by their own decorators, so the slot stays nil
    /// here and the engine composes the full per-token route.
    private func recordRouteObservation(
        layerIndex: Int,
        routedExpertIds: [Int],
        inputTokenId: UInt32
    ) {
        var tokenRoute: ObservedExpertRoute = Array(repeating: nil, count: layerIndex + 1)
        tokenRoute[layerIndex] = routedExpertIds.map({ (expertId: Int) -> UInt16 in
            return UInt16(expertId)
        })
        self.routeObservationRing.recordObservation(RouteObservationRecord(
            inputTokenId: inputTokenId,
            previousTokenRoute: nil,
            tokenRoute: tokenRoute))
    }
}
