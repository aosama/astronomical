import Foundation;

/**
 * Exact before/after bytes for one atomic memory configuration mutation, the
 * Swift port of the Rust `MaximumMlxMemoryConfigUpdate`. The prior bytes are
 * the document the candidate was built from; commit re-reads the file and
 * refuses to overwrite when they no longer match.
 */
internal struct MaximumMlxMemoryConfigUpdate {
    internal let priorConfigBytes: Data?;
    internal let candidateConfigBytes: Data;

    internal init(priorConfigBytes: Data?, candidateConfigBytes: Data) {
        self.priorConfigBytes = priorConfigBytes;
        self.candidateConfigBytes = candidateConfigBytes;
    }
}
