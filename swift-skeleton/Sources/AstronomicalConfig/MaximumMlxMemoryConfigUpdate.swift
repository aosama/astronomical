import Foundation;

/**
 * Exact before/after bytes for one atomic memory configuration mutation, the
 * Swift port of the Rust `MaximumMlxMemoryConfigUpdate`. The prior bytes are
 * the document the candidate was built from; commit re-reads the file and
 * refuses to overwrite when they no longer match.
 */
public struct MaximumMlxMemoryConfigUpdate {
    public let priorConfigBytes: Data?;
    public let candidateConfigBytes: Data;

    public init(priorConfigBytes: Data?, candidateConfigBytes: Data) {
        self.priorConfigBytes = priorConfigBytes;
        self.candidateConfigBytes = candidateConfigBytes;
    }
}
