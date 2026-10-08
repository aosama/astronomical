import Foundation

import MLX

/**
 * Quantized parameter slices one projection received for the experts a
 * layer's routes selected. Keys are the parameter basenames inside the
 * projection (`weight`, `scales`, `biases`) and the expert identifiers
 * whose slices were materialized, so the decorator can assemble full
 * parameter arrays without knowing the projection's parameter layout.
 */
public struct Qwen35MoeMaterializedProjectionSlices {

    /// Per-expert slices keyed by parameter basename, then expert id.
    public var parametersByParameterBasename: [String: [Int: MLXArray]]

    public init(parametersByParameterBasename: [String: [Int: MLXArray]]) {
        self.parametersByParameterBasename = parametersByParameterBasename
    }
}
