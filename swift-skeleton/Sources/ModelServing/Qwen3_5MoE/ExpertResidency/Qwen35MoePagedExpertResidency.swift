import Foundation;

import IpcProtocol;

/**
 * The paged-expert ownership facts one MoE engine load derives from its
 * validated configuration plus the paging plan: a retained subset of each
 * layer's routed experts sits in wired memory while misses page through
 * the materializer seam. The classification is the Memory subpackage's
 * single answer — retained bytes make the mode hybrid, an empty retained
 * set makes it paged; the engine never invents a second one.
 */
struct Qwen35MoePagedExpertResidency: Equatable {

    let totalLayerCount: UInt32;
    let retainedExpertCount: UInt32;
    let retainedExpertPayloadBytes: UInt64;

    init(
        repositoryConfiguration: Qwen3_5Config,
        retainedExpertIdsPerLayer: Array<Array<Int>>
    ) throws {
        let totalLayerCount: UInt32 = repositoryConfiguration.layerCount();
        guard retainedExpertIdsPerLayer.count == Int(totalLayerCount) else {
            throw InferenceEngineError.modelLoad(
                reason: "the paging plan must name retained experts for every decoder layer");
        }
        let expertCount: UInt32 = repositoryConfiguration.expertCount();
        for (layerIndex, retainedExpertIds) in retainedExpertIdsPerLayer.enumerated() {
            for retainedExpertId in retainedExpertIds {
                guard retainedExpertId >= 0, UInt32(retainedExpertId) < expertCount else {
                    throw InferenceEngineError.modelLoad(
                        reason: "the paging plan retains expert \(retainedExpertId) outside layer \(layerIndex)'s expert range");
                }
            }
        }
        let retainedExpertCount: UInt32 = UInt32(
            retainedExpertIdsPerLayer.reduce(0, { (retainedTotal: Int, retainedExpertIds: Array<Int>) -> Int in
                return retainedTotal + retainedExpertIds.count;
            }));
        self.totalLayerCount = totalLayerCount;
        self.retainedExpertCount = retainedExpertCount;
        self.retainedExpertPayloadBytes = try Qwen35MoeExpertPayloadArithmetic.payloadBytes(
            repositoryConfiguration: repositoryConfiguration, expertCount: retainedExpertCount);
    }

    /// The single expert-ownership answer for a paged install: paging is
    /// configured and the retained set is the only resident payload.
    var expertMemoryMode: ExpertMemoryMode {
        return ExpertMemoryModeClassification.classify(
            completeSparseOwnerIsInstalled: false,
            sparseExpertPagingIsConfigured: true,
            retainedPagedExpertPayloadBytes: self.retainedExpertPayloadBytes);
    }

    /// The wire-shaped final residency state for the worker's events; the
    /// retained set is the resident expert payload under paging.
    var telemetry: ExpertResidencyTelemetry {
        return ExpertResidencyTelemetry(
            totalLayerCount: self.totalLayerCount,
            residentExpertCount: self.retainedExpertCount,
            residentExpertPayloadBytes: self.retainedExpertPayloadBytes);
    }
}
