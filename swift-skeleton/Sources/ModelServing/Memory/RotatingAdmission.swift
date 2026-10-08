import Foundation

/// Architecture-neutral rotating-cache admission geometry, port of the
/// Rust `memory::admission::rotating` module.
///
/// A sliding layer commits at most `windowSize` tokens. Multi-token
/// prefill may temporarily own `windowSize + promptChunk - 1` tokens so
/// every new token still sees a full prior window. Memory admission must
/// charge that transient, not only the steady-state ring.
public enum RotatingAdmission {

    /// Returns the peak token count owned during one rotating prefill chunk.
    public static func rotatingPrefillTransientTokenCount(
        windowSize: UInt32,
        promptChunkTokenCount: UInt32
    ) throws -> UInt32 {
        if windowSize == 0 {
            throw RotatingAdmissionError.zeroWindowSize
        }
        if promptChunkTokenCount == 0 {
            return windowSize
        }
        let (windowPlusChunk, additionOverflowed) =
            windowSize.addingReportingOverflow(promptChunkTokenCount)
        if additionOverflowed {
            throw RotatingAdmissionError.transientTokenCountOverflow(
                windowSize: windowSize,
                promptChunkTokenCount: promptChunkTokenCount)
        }
        let (peakTokenCount, subtractionOverflowed) = windowPlusChunk.subtractingReportingOverflow(1)
        if subtractionOverflowed {
            throw RotatingAdmissionError.transientTokenCountOverflow(
                windowSize: windowSize,
                promptChunkTokenCount: promptChunkTokenCount)
        }
        return peakTokenCount
    }

    /// Returns how many tokens remain after a rotating update commits.
    public static func rotatingCommittedTokenCount(
        windowSize: UInt32,
        absolutePosition: UInt32
    ) -> UInt32 {
        return min(absolutePosition, windowSize)
    }
}
