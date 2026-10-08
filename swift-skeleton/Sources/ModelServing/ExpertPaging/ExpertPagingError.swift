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
    /// A direct native runtime failure observed while streaming expert
    /// layers.
    case nativeRuntime(MlxRuntimeError)
    /// The allocation admission policy rejected a staged expert load.
    case memoryBudget(MlxAllocationAdmissionError)
}
