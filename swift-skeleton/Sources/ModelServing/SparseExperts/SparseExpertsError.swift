import Foundation

/// Domain failures for the sparse expert selection shell. Port of the
/// Rust `SparseExpertError` surface.
public enum SparseExpertsError: Error, Equatable, Sendable {
    /// An assignment permutation was not a valid complete mapping.
    case invalidAssignmentGeometry(description: String)
}
