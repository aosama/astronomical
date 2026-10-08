import Foundation

import JourneyCategories;
import MLX;
import MLXLMCommon;
import MLXNN;
import ModelServingTestSupport;
import Testing;

@testable import ModelServing;

/**
 * Engine epic #983 track E3 substrate evidence: the paged expert decorator
 * materializes ONLY the experts a layer's routes select, produces outputs
 * bit-identical to the resident all-experts reference through both
 * upstream dispatch paths, reads no more expert pages on a repeated
 * identical forward (issue #629 hermetic seed), and records the route
 * observations the on-device route predictor consumes.
 */
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class Qwen35MoePagedExpertDecoratorTests {

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    @Test
    func shouldMaterializeExactlyTheRoutedExpertsAndMatchTheResidentReference() throws {
        let routingIndices: MLXArray = MLXArray(
            (0..<16).map({ (assignmentIndex: Int) -> Int32 in
                return (assignmentIndex % 2 == 0) ? 0 : 2;
            }), [8, 2]);
        let pagingFixture = try DeterministicPagedSwitchGluFixture(
            expertCount: 3, inputDimensions: 32, hiddenDimensions: 64);
        let decorator = Qwen35MoePagedExpertDecorator(
            expertPageMaterializer: pagingFixture.expertPageMaterializer,
            routeObservationRing: RouteObservationRing(
                capacity: RouteObservationRing.defaultObservationCapacity),
            attributionEnabled: false);

        let residentOutputs: MLXArray = pagingFixture.residentOutputs(routingIndices: routingIndices);
        let decoratedOutputs: MLXArray = try decorator.decoratedLayerOutputs(
            layerIndex: 0,
            inputTokenEmbeddings: pagingFixture.tokenInputs,
            routingIndices: routingIndices,
            switchGlu: pagingFixture.pagedSwitchGlu,
            inputTokenId: 11);

        #expect(
            pagingFixture.expertPageMaterializer.requestedExpertIdsByCall == [[0, 2]],
            "the decorator must request exactly the routed experts, sorted, never an unselected one");
        #expect(
            pagingFixture.expertPageMaterializer.servedExpertPageTotal == 2,
            "one expert page per distinct routed expert must be read");
        let maximumDifference: Float = maximumAbsoluteDifference(decoratedOutputs, residentOutputs);
        #expect(
            maximumDifference == 0,
            "paged execution must be bit-identical to the resident all-experts reference (max diff \(maximumDifference))");
    }

    @Test
    func shouldMatchTheResidentReferenceThroughTheSortedPagedPath() throws {
        // 32 tokens x 2 assignments reaches the upstream gather/sort
        // threshold of 64 and exercises the sorted path under paging.
        let routingIndices: MLXArray = MLXArray(
            (0..<32).map({ (tokenIndex: Int) -> Int32 in
                return Int32(tokenIndex % 3);
            }) + (0..<32).map({ (tokenIndex: Int) -> Int32 in
                return Int32((tokenIndex + 1) % 3);
            }), [32, 2]);
        let pagingFixture = try DeterministicPagedSwitchGluFixture(
            expertCount: 3, inputDimensions: 32, hiddenDimensions: 64, tokenCount: 32);
        let decorator = Qwen35MoePagedExpertDecorator(
            expertPageMaterializer: pagingFixture.expertPageMaterializer,
            routeObservationRing: RouteObservationRing(
                capacity: RouteObservationRing.defaultObservationCapacity),
            attributionEnabled: false);

        // MLX's first evaluation of a freshly quantized layer takes a lazy
        // compile path whose numerics are not the steady-state kernel, so
        // both modules are warmed once before the asserted comparison; the
        // paging contract is about steady-state execution.
        _ = pagingFixture.residentOutputs(routingIndices: routingIndices);
        _ = try decorator.decoratedLayerOutputs(
            layerIndex: 0,
            inputTokenEmbeddings: pagingFixture.tokenInputs,
            routingIndices: routingIndices,
            switchGlu: pagingFixture.pagedSwitchGlu,
            inputTokenId: 12);

        let residentOutputs: MLXArray = pagingFixture.residentOutputs(routingIndices: routingIndices);
        let decoratedOutputs: MLXArray = try decorator.decoratedLayerOutputs(
            layerIndex: 0,
            inputTokenEmbeddings: pagingFixture.tokenInputs,
            routingIndices: routingIndices,
            switchGlu: pagingFixture.pagedSwitchGlu,
            inputTokenId: 12);

        #expect(
            pagingFixture.expertPageMaterializer.requestedExpertIdsByCall == [[0, 1, 2]],
            "the sorted path must materialize exactly the distinct routed experts");
        let maximumDifference: Float = maximumAbsoluteDifference(decoratedOutputs, residentOutputs);
        #expect(
            maximumDifference == 0,
            "paged execution through the gather/sort path must stay bit-identical to the resident reference (max diff \(maximumDifference))");
    }

    @Test
    func shouldReadNoMoreExpertPagesOnASecondIdenticalForward() throws {
        let routingIndices: MLXArray = MLXArray(
            (0..<16).map({ (assignmentIndex: Int) -> Int32 in
                return (assignmentIndex % 2 == 0) ? 0 : 2;
            }), [8, 2]);
        let pagingFixture = try DeterministicPagedSwitchGluFixture(
            expertCount: 3, inputDimensions: 32, hiddenDimensions: 64);
        let decorator = Qwen35MoePagedExpertDecorator(
            expertPageMaterializer: pagingFixture.expertPageMaterializer,
            routeObservationRing: RouteObservationRing(
                capacity: RouteObservationRing.defaultObservationCapacity),
            attributionEnabled: false);

        let firstOutputs: MLXArray = try decorator.decoratedLayerOutputs(
            layerIndex: 0,
            inputTokenEmbeddings: pagingFixture.tokenInputs,
            routingIndices: routingIndices,
            switchGlu: pagingFixture.pagedSwitchGlu,
            inputTokenId: 11);
        let firstForwardPageReadCount: UInt64 = decorator.totalExpertPageReadCount;

        let secondOutputs: MLXArray = try decorator.decoratedLayerOutputs(
            layerIndex: 0,
            inputTokenEmbeddings: pagingFixture.tokenInputs,
            routingIndices: routingIndices,
            switchGlu: pagingFixture.pagedSwitchGlu,
            inputTokenId: 11);
        let secondForwardPageReadCount: UInt64 = decorator.totalExpertPageReadCount;

        #expect(
            firstForwardPageReadCount == 2,
            "the first forward must read one page per distinct routed expert");
        #expect(
            secondForwardPageReadCount == firstForwardPageReadCount,
            "issue #629: a repeated identical forward must read no additional expert pages");
        #expect(
            pagingFixture.expertPageMaterializer.requestedExpertIdsByCall.count == 1,
            "the materializer must not be consulted again once the routed pages are installed");
        #expect(
            maximumAbsoluteDifference(firstOutputs, secondOutputs) == 0,
            "both forwards must produce identical outputs");
    }

    @Test
    func shouldRecordRouteObservationsForTheDecoratedLayer() throws {
        let firstRoutingIndices: MLXArray = MLXArray(
            (0..<16).map({ (assignmentIndex: Int) -> Int32 in
                return (assignmentIndex % 2 == 0) ? 0 : 2;
            }), [8, 2]);
        let secondRoutingIndices: MLXArray = MLXArray(
            [Int32](repeating: 1, count: 16), [8, 2]);
        let pagingFixture = try DeterministicPagedSwitchGluFixture(
            expertCount: 3, inputDimensions: 32, hiddenDimensions: 64);
        let routeObservationRing: RouteObservationRing = RouteObservationRing(
            capacity: RouteObservationRing.defaultObservationCapacity);
        let decorator = Qwen35MoePagedExpertDecorator(
            expertPageMaterializer: pagingFixture.expertPageMaterializer,
            routeObservationRing: routeObservationRing,
            attributionEnabled: false);

        _ = try decorator.decoratedLayerOutputs(
            layerIndex: 5,
            inputTokenEmbeddings: pagingFixture.tokenInputs,
            routingIndices: firstRoutingIndices,
            switchGlu: pagingFixture.pagedSwitchGlu,
            inputTokenId: 21);
        _ = try decorator.decoratedLayerOutputs(
            layerIndex: 5,
            inputTokenEmbeddings: pagingFixture.tokenInputs,
            routingIndices: secondRoutingIndices,
            switchGlu: pagingFixture.pagedSwitchGlu,
            inputTokenId: 22);

        #expect(routeObservationRing.observationCount == 2);
        let firstObservation: RouteObservationRecord = routeObservationRing.observations()[0];
        #expect(firstObservation.inputTokenId == 21);
        #expect(firstObservation.tokenRoute.count == 6);
        #expect(firstObservation.tokenRoute[0] == nil);
        #expect(firstObservation.tokenRoute[5] == [UInt16(0), UInt16(2)]);
        let secondObservation: RouteObservationRecord = routeObservationRing.observations()[1];
        #expect(secondObservation.inputTokenId == 22);
        #expect(secondObservation.tokenRoute[5] == [UInt16(1)]);
    }

    /// Maximum absolute elementwise difference between two equally shaped
    /// arrays, fully evaluated.
    private func maximumAbsoluteDifference(_ leftOutputs: MLXArray, _ rightOutputs: MLXArray) -> Float {
        let differenceArray: MLXArray = MLX.max(MLX.abs(leftOutputs - rightOutputs));
        return differenceArray.asArray(Float.self)[0];
    }
}

/// One quantized SwitchGLU pair: a fully resident reference and a zeroed
/// paged twin, plus a fake materializer that serves per-expert quantized
/// slices extracted from the resident parameters and counts every served
/// page. The twin starts from all-zero weights so any unselected expert
/// that leaked into execution would surface as a nonzero difference.
private final class DeterministicPagedSwitchGluFixture {

    let pagedSwitchGlu: SwitchGLU;
    let tokenInputs: MLXArray;
    let expertPageMaterializer: QuantizedExpertPageFakeMaterializer;
    private let residentSwitchGlu: SwitchGLU;

    init(
        expertCount: Int,
        inputDimensions: Int,
        hiddenDimensions: Int,
        tokenCount: Int = 8
    ) throws {
        let residentSwitchGlu: SwitchGLU = SwitchGLU(
            inputDims: inputDimensions, hiddenDims: hiddenDimensions,
            numExperts: expertCount, bias: false);
        let pagedSwitchGlu: SwitchGLU = SwitchGLU(
            inputDims: inputDimensions, hiddenDims: hiddenDimensions,
            numExperts: expertCount, bias: false);

        let gateWeights: MLXArray = DeterministicPagedSwitchGluFixture.constructedWeights(
            leadingDimensions: [expertCount, hiddenDimensions], trailingDimensions: inputDimensions,
            salt: 1);
        let upWeights: MLXArray = DeterministicPagedSwitchGluFixture.constructedWeights(
            leadingDimensions: [expertCount, hiddenDimensions], trailingDimensions: inputDimensions,
            salt: 2);
        let downWeights: MLXArray = DeterministicPagedSwitchGluFixture.constructedWeights(
            leadingDimensions: [expertCount, inputDimensions], trailingDimensions: hiddenDimensions,
            salt: 3);
        let boundWeights: Dictionary<String, MLXArray> = [
            "gate_proj.weight": gateWeights,
            "up_proj.weight": upWeights,
            "down_proj.weight": downWeights,
        ];
        try residentSwitchGlu.update(
            parameters: ModuleParameters.unflattened(boundWeights), verify: [.all]);
        try pagedSwitchGlu.update(
            parameters: ModuleParameters.unflattened(boundWeights), verify: [.all]);

        let affineQuantizationClosure = { (_: String, _: Module) -> (
            groupSize: Int, bits: Int, mode: QuantizationMode
        )? in
            return (groupSize: 32, bits: 4, mode: QuantizationMode.affine);
        };
        MLXNN.quantize(model: residentSwitchGlu, filter: affineQuantizationClosure);
        MLXNN.quantize(model: pagedSwitchGlu, filter: affineQuantizationClosure);

        let initializedInputs: Array<Float> = (0..<(tokenCount * inputDimensions)).map(
            { (flatIndex: Int) -> Float in
                return Float((flatIndex % 7) - 3) * 0.1;
            });
        self.tokenInputs = MLXArray(initializedInputs, [tokenCount, inputDimensions]);

        try pagedSwitchGlu.update(
            parameters: ModuleParameters.unflattened(
                DeterministicPagedSwitchGluFixture.zeroedParameters(model: pagedSwitchGlu)),
            verify: [.all]);
        MLX.eval(pagedSwitchGlu);

        self.residentSwitchGlu = residentSwitchGlu;
        self.pagedSwitchGlu = pagedSwitchGlu;
        self.expertPageMaterializer = QuantizedExpertPageFakeMaterializer(
            slicesByProjectionName: DeterministicPagedSwitchGluFixture.extractPerExpertSlices(
                model: residentSwitchGlu, expertCount: expertCount));
    }

    /// The resident all-experts reference outputs, fully evaluated.
    func residentOutputs(routingIndices: MLXArray) -> MLXArray {
        let residentOutputs: MLXArray = self.residentSwitchGlu(self.tokenInputs, routingIndices);
        MLX.eval(residentOutputs);
        return residentOutputs;
    }

    /// Replaces every parameter with a zero array of the same shape and
    /// dtype, leaving quantized packed shapes intact.
    private static func zeroedParameters(model: Module) -> Dictionary<String, MLXArray> {
        var zeroedParameters: Dictionary<String, MLXArray> = [:];
        for (parameterName, parameterArray) in model.parameters().flattened() {
            let parameterShape: Array<Int> = parameterArray.shape;
            zeroedParameters[parameterName] = MLX.zeros(parameterShape, dtype: parameterArray.dtype);
        }
        return zeroedParameters;
    }

    /// Splits the resident quantized parameters into per-expert slices,
    /// grouped by projection and parameter basename.
    private static func extractPerExpertSlices(
        model: Module,
        expertCount: Int
    ) -> Dictionary<String, Dictionary<String, Dictionary<Int, MLXArray>>> {
        let residentParameters: Dictionary<String, MLXArray> = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened());
        var slicesByProjectionName: Dictionary<String, Dictionary<String, Dictionary<Int, MLXArray>>> = [:];
        for projectionName: String in ["gate_proj", "up_proj", "down_proj"] {
            let projectionPrefix: String = "\(projectionName).";
            var basenameSlices: Dictionary<String, Dictionary<Int, MLXArray>> = [:];
            for (parameterName, parameterArray) in residentParameters {
                guard parameterName.hasPrefix(projectionPrefix) else {
                    continue;
                }
                let parameterBasename: String = String(parameterName.dropFirst(projectionPrefix.count));
                var expertSlices: Dictionary<Int, MLXArray> = [:];
                for expertId: Int in 0..<expertCount {
                    expertSlices[expertId] = parameterArray[expertId];
                }
                basenameSlices[parameterBasename] = expertSlices;
            }
            slicesByProjectionName[projectionName] = basenameSlices;
        }
        for (_, parameterArray) in residentParameters {
            MLX.eval(parameterArray);
        }
        return slicesByProjectionName;
    }

    /// Fills [e1, e2, trailing] with small deterministic values spanning
    /// negative and positive magnitudes.
    private static func constructedWeights(
        leadingDimensions: Array<Int>,
        trailingDimensions: Int,
        salt: Int
    ) -> MLXArray {
        let flatCount: Int = leadingDimensions.reduce(trailingDimensions, { (partial: Int, dimension: Int) -> Int in
            return partial * dimension;
        });
        let constructedValues: Array<Float> = (0..<flatCount).map(
            { (flatIndex: Int) -> Float in
                let columnIndex: Int = flatIndex % trailingDimensions;
                let leadingIndex: Int = flatIndex / trailingDimensions;
                return Float(((leadingIndex * 7 + columnIndex * 3 + salt) % 9) - 4) * 0.1;
            });
        return MLXArray(constructedValues, leadingDimensions + [trailingDimensions]);
    }
}

/// Fake materializer serving prebuilt per-expert quantized slices while
/// recording every consultation and served page for journey assertions.
private final class QuantizedExpertPageFakeMaterializer: Qwen35MoeExpertPageMaterializing {

    private(set) var requestedExpertIdsByCall: Array<Array<Int>> = [];
    private(set) var servedExpertPageTotal: Int = 0;
    private let slicesByProjectionName: Dictionary<String, Dictionary<String, Dictionary<Int, MLXArray>>>;

    init(slicesByProjectionName: Dictionary<String, Dictionary<String, Dictionary<Int, MLXArray>>>) {
        self.slicesByProjectionName = slicesByProjectionName;
    }

    public func materializeExpertWeights(
        layerIndex: Int,
        expertIds: Array<Int>
    ) throws -> Qwen35MoeMaterializedExpertWeights {
        self.requestedExpertIdsByCall.append(expertIds);
        self.servedExpertPageTotal = self.servedExpertPageTotal + expertIds.count;
        return Qwen35MoeMaterializedExpertWeights(
            gateProjection: self.projectionSlices(projectionName: "gate_proj", expertIds: expertIds),
            upProjection: self.projectionSlices(projectionName: "up_proj", expertIds: expertIds),
            downProjection: self.projectionSlices(projectionName: "down_proj", expertIds: expertIds),
            expertPageReadCount: expertIds.count);
    }

    private func projectionSlices(
        projectionName: String,
        expertIds: Array<Int>
    ) -> Qwen35MoeMaterializedProjectionSlices {
        var requestedBasenameSlices: Dictionary<String, Dictionary<Int, MLXArray>> = [:];
        let availableBasenameSlices: Dictionary<String, Dictionary<Int, MLXArray>> =
            self.slicesByProjectionName[projectionName] ?? [:];
        for (parameterBasename, availableExpertSlices) in availableBasenameSlices {
            var requestedExpertSlices: Dictionary<Int, MLXArray> = [:];
            for expertId: Int in expertIds {
                if let expertSlice: MLXArray = availableExpertSlices[expertId] {
                    requestedExpertSlices[expertId] = expertSlice;
                }
            }
            requestedBasenameSlices[parameterBasename] = requestedExpertSlices;
        }
        return Qwen35MoeMaterializedProjectionSlices(parametersByParameterBasename: requestedBasenameSlices);
    }
}
