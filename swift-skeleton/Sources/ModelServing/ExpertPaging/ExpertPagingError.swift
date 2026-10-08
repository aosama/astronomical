import Foundation

import RuntimeIntegration

/// Domain failures for the quantized expert paging family: manifest
/// accounting, source validation, page assembly, and the two runtime
/// failure channels (native MLX execution and memory-budget admission)
/// that execution owners classify as recoverable capacity pressure.
public enum ExpertPagingError: Error, Equatable, Sendable {
    /// A per-expert payload byte sum overflowed 64-bit accounting.
    case expertPayloadAccountingOverflow(layerPrefix: String)
    /// A complete-layer payload product overflowed 64-bit accounting.
    case completeLayerPayloadAccountingOverflow(layerPrefix: String)
    /// A validated source or manifest invariant failed at startup time.
    case manifestValidationFailure(description: String)
    /// A page request named a decoder layer the startup plans do not cover.
    case layerIndexOutOfRange(layerIndex: Int, layerCount: Int)
    /// The loaded page lacks one tensor the layer plan requires, so no
    /// partial page can reach execution.
    case pageTensorMissing(tensorName: String, layerPrefix: String)
    /// A loaded page tensor holds a different expert-row count than the
    /// page manifest seated, so slices cannot be indexed by compact slot.
    case pageTensorExpertCountMismatch(
        tensorName: String, expectedSlotCount: Int, actualSlotCount: Int)
    /// A direct native runtime failure observed while streaming expert
    /// layers.
    case nativeRuntime(MlxRuntimeError)
    /// The allocation admission policy rejected a staged expert load.
    case memoryBudget(MlxAllocationAdmissionError)
}
