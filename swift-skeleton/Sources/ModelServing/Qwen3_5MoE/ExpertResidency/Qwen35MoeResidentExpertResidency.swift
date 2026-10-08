import Foundation;

import IpcProtocol;

/// The resident-expert ownership facts one MoE engine load derives from its
/// validated configuration: resident execution keeps every routed expert of
/// every decoder layer in wired memory, so the telemetry is the full config
/// product rather than a measured or packaged constant. The payload size is
/// the bf16 gate, up, and down matrices of one SwitchGLU expert times the
/// resident set; router gates and shared experts are not routed payload.
struct Qwen35MoeResidentExpertResidency: Equatable {

    let totalLayerCount: UInt32;
    let residentExpertCount: UInt32;
    let residentExpertPayloadBytes: UInt64;

    init(repositoryConfiguration: Qwen3_5Config) throws {
        let totalLayerCount: UInt32 = repositoryConfiguration.layerCount();
        let expertCount: UInt32 = repositoryConfiguration.expertCount();
        let (residentExpertCount, countOverflow) = totalLayerCount
            .multipliedReportingOverflow(by: expertCount);
        if countOverflow {
            throw InferenceEngineError.modelLoad(
                reason: "the resident expert residency overflows its expert count arithmetic");
        }
        let residentExpertPayloadBytes: UInt64 = try Qwen35MoeExpertPayloadArithmetic.payloadBytes(
            repositoryConfiguration: repositoryConfiguration, expertCount: residentExpertCount);
        self.totalLayerCount = totalLayerCount;
        self.residentExpertCount = residentExpertCount;
        self.residentExpertPayloadBytes = residentExpertPayloadBytes;
    }

    /// The single expert-ownership answer the Memory subpackage classifies
    /// from structural facts; the resident engine never invents a second one.
    var expertMemoryMode: ExpertMemoryMode {
        return ExpertMemoryModeClassification.classify(
            completeSparseOwnerIsInstalled: true,
            sparseExpertPagingIsConfigured: false,
            retainedPagedExpertPayloadBytes: 0);
    }

    /// The wire-shaped final residency state for the worker's events.
    var telemetry: ExpertResidencyTelemetry {
        return ExpertResidencyTelemetry(
            totalLayerCount: self.totalLayerCount,
            residentExpertCount: self.residentExpertCount,
            residentExpertPayloadBytes: self.residentExpertPayloadBytes);
    }
}
