import Foundation;

import IpcProtocol;

/**
 * The expert payload byte arithmetic shared by every residency family: the
 * bf16 gate, up, and down matrices of one SwitchGLU expert times the owned
 * expert set. Router gates and shared experts are not routed payload.
 */
enum Qwen35MoeExpertPayloadArithmetic {

    /// The routed payload bytes of the given expert count, failing closed
    /// when the product overflows.
    static func payloadBytes(
        repositoryConfiguration: Qwen3_5Config,
        expertCount: UInt32
    ) throws -> UInt64 {
        let projectionMatrixCountPerExpert: UInt64 = 3;
        let bfloat16BytesPerElement: UInt64 = 2;
        let perExpertPayloadBytes: UInt64 = projectionMatrixCountPerExpert
            * UInt64(repositoryConfiguration.hiddenSize())
            * UInt64(repositoryConfiguration.expertIntermediateSize())
            * bfloat16BytesPerElement;
        let (payloadBytes, payloadOverflow) = UInt64(expertCount)
            .multipliedReportingOverflow(by: perExpertPayloadBytes);
        if payloadOverflow {
            throw InferenceEngineError.modelLoad(
                reason: "the expert payload byte arithmetic overflows its expert count");
        }
        return payloadBytes;
    }
}
