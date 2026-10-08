import Foundation

/// One exhaustive decoder-cache state family in a model layer, port of
/// the Rust `DecoderCacheLayerLayout`. Case construction replaces the
/// Rust factory functions because Swift case labels already carry the
/// role vocabulary.
public enum DecoderCacheLayerLayout: Equatable, Sendable {

    /// Append-only attention state, stored in token-sliceable blocks.
    case appendOnlyAttention(
        keys: DecoderCacheTensorLayout,
        values: DecoderCacheTensorLayout,
        capacityGrowthTokens: Int)

    /// Bounded rotating attention state persisted at complete boundaries.
    case rotatingWindowAttention(
        keys: DecoderCacheTensorLayout,
        values: DecoderCacheTensorLayout,
        windowSize: Int)

    /// Fixed state restored only from the newest complete prompt boundary.
    case recurrentTensor(tensor: DecoderCacheTensorLayout)

    /// Ordered state components for hybrid decoder layers.
    case composite(components: [DecoderCacheLayerLayout])

    /// Boundary tensors that persist rotating counters beside key/value
    /// slabs, port of the Rust `rotating_layout` counter vocabulary.
    static func rotatingWindowCounterLayouts() -> [DecoderCacheTensorLayout] {
        [
            .fixed(
                tensorRoleName: "attention.absolute_position",
                dtype: .float32,
                dimensions: [1]),
            .fixed(
                tensorRoleName: "attention.ring_write_index",
                dtype: .float32,
                dimensions: [1]),
        ]
    }
}
