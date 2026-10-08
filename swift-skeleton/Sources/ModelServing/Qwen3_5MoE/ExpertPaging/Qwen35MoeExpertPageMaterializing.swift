import Foundation

/**
 * The residency seam the paged expert decorator consults for the experts a
 * layer's routes select. Implementations own the residency and I/O policy
 * — retained-page lookup, disk reads, byte accounting against the composed
 * memory budget — while the decorator stays a pure orchestration of
 * materialize, install, execute, and record.
 */
public protocol Qwen35MoeExpertPageMaterializing {

    /**
     * Materializes the quantized weight slices for exactly these experts
     * of one decoder layer, reading only what residency is missing.
     *
     * - Parameters:
     *   - layerIndex: The decoder layer whose experts are requested.
     *   - expertIds: The distinct routed expert identifiers, ascending.
     * - Returns: The materialized slices and the page-read count.
     * - Throws: When the backing store cannot serve the requested pages.
     */
    func materializeExpertWeights(
        layerIndex: Int,
        expertIds: [Int]
    ) throws -> Qwen35MoeMaterializedExpertWeights
}
