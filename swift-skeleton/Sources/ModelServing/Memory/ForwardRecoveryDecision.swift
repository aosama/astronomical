import Foundation

/// Retry authorization and its complete calculation evidence.
public enum ForwardRecoveryDecision: Equatable, Hashable, Sendable {

    /// The unchanged forward may retry once the named reclamation completes.
    case retry(fixedForwardWorkspaceBytes: Int, requiredReclamationBytes: Int)

    /// The request must fail at the named boundary; no retry is authorized.
    case reject(
        boundary: MemoryBoundary,
        fixedForwardWorkspaceBytes: Int,
        requiredReclamationBytes: Int)
}
