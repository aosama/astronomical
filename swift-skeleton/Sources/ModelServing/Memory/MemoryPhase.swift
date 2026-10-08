import Foundation

/**
 * Request lifecycle position a memory decision is made for.
 *
 * The one lifecycle vocabulary for every memory question in this package:
 * budget composition, residency planning, and release enactment all read the
 * same phase instead of carrying private mismatched enums. Budget mapping
 * sites treat `generationPreparation` as decode-equivalent, but residency
 * keeps it distinct so plans can preserve more expert payload before the
 * first generated token.
 */
public enum MemoryPhase: Equatable, Hashable, Sendable {

    /// Prompt chunks are being processed; activations are large and chunk-shaped.
    case prefill

    /// The prompt finished; resident expert ownership is prepared before the
    /// first generated token.
    case generationPreparation

    /// Tokens are being generated one at a time.
    case decode

    /// No request work is in flight.
    case idle
}
