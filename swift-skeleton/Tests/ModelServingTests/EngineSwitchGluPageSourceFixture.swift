import Foundation;

import MLX;

import ModelServing;

@testable import ModelServing;

/**
 * The page source seeded from a fully resident engine: every expert slice
 * the journeys serve was extracted from the resident twin's loaded expert
 * primitives, so a paged engine reading through this source executes the
 * exact quantities its resident twin does. Every consultation and served
 * page is recorded for the issue #629 read-once assertions.
 */
final class EngineSwitchGluPageSourceFixture: Qwen35MoeExpertPageMaterializing {

    /// The expert id sets this source was consulted with, in call order.
    private(set) var requestedExpertIdsByCall: Array<Array<Int>> = [];

    /// The pages this source has served in total across every layer.
    private(set) var servedExpertPageTotal: Int = 0;

    private let slicesByLayerIndexByProjectionName: Dictionary<
        Int, Dictionary<String, Dictionary<String, Dictionary<Int, MLXArray>>>>;

    init(
        residentEngine: Qwen35MoeEngine,
        layerCount: Int,
        expertCount: Int
    ) throws {
        var slicesByLayerIndexByProjectionName: Dictionary<
            Int, Dictionary<String, Dictionary<String, Dictionary<Int, MLXArray>>>> = [:];
        for layerIndex: Int in 0..<layerCount {
            let parameterArrays: Dictionary<String, MLXArray> = try residentEngine
                .switchGluParameterArrays(layerIndex: layerIndex);
            var basenameSlicesByProjectionName: Dictionary<
                String, Dictionary<String, Dictionary<Int, MLXArray>>> = [:];
            for projectionName: String in ["gate_proj", "up_proj", "down_proj"] {
                let projectionPrefix: String = "\(projectionName).";
                var basenameSlices: Dictionary<String, Dictionary<Int, MLXArray>> = [:];
                for (parameterName, parameterArray) in parameterArrays {
                    guard parameterName.hasPrefix(projectionPrefix) else {
                        continue;
                    }
                    let parameterBasename: String = String(parameterName.dropFirst(projectionPrefix.count));
                    var expertSlices: Dictionary<Int, MLXArray> = [:];
                    for expertId: Int in 0..<expertCount {
                        let expertSlice: MLXArray = parameterArray[expertId];
                        MLX.eval(expertSlice);
                        expertSlices[expertId] = expertSlice;
                    }
                    basenameSlices[parameterBasename] = expertSlices;
                }
                basenameSlicesByProjectionName[projectionName] = basenameSlices;
            }
            slicesByLayerIndexByProjectionName[layerIndex] = basenameSlicesByProjectionName;
        }
        self.slicesByLayerIndexByProjectionName = slicesByLayerIndexByProjectionName;
    }

    public func materializeExpertWeights(
        layerIndex: Int,
        expertIds: Array<Int>
    ) throws -> Qwen35MoeMaterializedExpertWeights {
        self.requestedExpertIdsByCall.append(expertIds);
        self.servedExpertPageTotal = self.servedExpertPageTotal + expertIds.count;
        let basenameSlicesByProjectionName: Dictionary<String, Dictionary<String, Dictionary<Int, MLXArray>>>
            = self.slicesByLayerIndexByProjectionName[layerIndex]
                ?? Dictionary();
        return Qwen35MoeMaterializedExpertWeights(
            gateProjection: self.projectionSlices(
                basenameSlices: basenameSlicesByProjectionName["gate_proj"] ?? Dictionary(),
                expertIds: expertIds),
            upProjection: self.projectionSlices(
                basenameSlices: basenameSlicesByProjectionName["up_proj"] ?? Dictionary(),
                expertIds: expertIds),
            downProjection: self.projectionSlices(
                basenameSlices: basenameSlicesByProjectionName["down_proj"] ?? Dictionary(),
                expertIds: expertIds),
            expertPageReadCount: expertIds.count);
    }

    private func projectionSlices(
        basenameSlices: Dictionary<String, Dictionary<Int, MLXArray>>,
        expertIds: Array<Int>
    ) -> Qwen35MoeMaterializedProjectionSlices {
        var requestedBasenameSlices: Dictionary<String, Dictionary<Int, MLXArray>> = [:];
        for (parameterBasename, availableExpertSlices) in basenameSlices {
            var requestedExpertSlices: Dictionary<Int, MLXArray> = [:];
            for expertId: Int in expertIds {
                if let expertSlice: MLXArray = availableExpertSlices[expertId] {
                    requestedExpertSlices[expertId] = expertSlice;
                }
            }
            requestedBasenameSlices[parameterBasename] = requestedExpertSlices;
        }
        return Qwen35MoeMaterializedProjectionSlices(
            parametersByParameterBasename: requestedBasenameSlices);
    }
}
