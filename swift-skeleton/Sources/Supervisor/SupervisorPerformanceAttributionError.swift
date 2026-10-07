import Foundation

/**
 * Typed failures of the supervisor performance attribution path, mirroring
 * the io::Error values the Rust log propagates.
 */
public enum SupervisorPerformanceAttributionError: Error, CustomStringConvertible {

    /// The injectable wall clock failed; the carried text preserves the
    /// underlying failure for diagnosis.
    case clockFailure(problem: String)

    /// The attribution sink rejected the write; the carried text preserves
    /// the underlying failure for diagnosis.
    case writerFailure(problem: String)

    /// The measured operation/detail pairing contract was violated.
    case mismatchedOperationDetail

    /// A download measurement carried a non-canonical artifact identity.
    case invalidDownloadIdentity(problem: String)

    /// A file-transfer measurement carried a non-canonical relative path.
    case invalidRelativeFilePath(problem: String)

    public var description: String {
        switch (self) {
        case let .clockFailure(problem): return problem
        case let .writerFailure(problem): return problem
        case .mismatchedOperationDetail:
            return "supervisor attribution operation and measurement detail do not match"
        case let .invalidDownloadIdentity(problem): return problem
        case let .invalidRelativeFilePath(problem): return problem
        }
    }
}
