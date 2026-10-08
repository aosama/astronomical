import Foundation;

import IpcProtocol;

/**
 * The expert-ownership surface one MoE engine load exposes to the worker
 * wire: a single classified mode plus the wire-shaped telemetry. The
 * resident and paged families both answer through the Memory
 * subpackage's classification, so the engine never branches on which
 * family is installed.
 */
protocol Qwen35MoeExpertResidency {

    /// The single expert-ownership answer for the installed family.
    var expertMemoryMode: ExpertMemoryMode { get }

    /// The wire-shaped final residency state for the worker's events.
    var telemetry: ExpertResidencyTelemetry { get }
}

extension Qwen35MoeResidentExpertResidency: Qwen35MoeExpertResidency {
}

extension Qwen35MoePagedExpertResidency: Qwen35MoeExpertResidency {
}
