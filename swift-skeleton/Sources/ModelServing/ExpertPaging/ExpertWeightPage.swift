import Foundation

/// Opaque per-page payload the retained cache stores but never inspects.
///
/// The model family performs SafeTensors input/output and MLX evaluation
/// before offering a page to the cache; the cache uses only the byte count.
public protocol ExpertWeightPage {

    /// Bytes this page occupies in wired memory; zero-byte pages are
    /// rejected by ownership accounting.
    func residentPayloadByteCount() -> UInt64
}
