import Foundation

/// Domain failures for the quantized expert paging family: manifest
/// accounting, source validation, and page assembly. Port of the Rust
/// `ExpertManifestError` surface as the single checked-error enum this
/// layer throws.
public enum ExpertPagingError: Error, Equatable, Sendable {
    /// A per-expert payload byte sum overflowed 64-bit accounting.
    case expertPayloadAccountingOverflow(layerPrefix: String)
    /// A complete-layer payload product overflowed 64-bit accounting.
    case completeLayerPayloadAccountingOverflow(layerPrefix: String)
    /// A validated source or manifest invariant failed at startup time.
    case manifestValidationFailure(description: String)
}
