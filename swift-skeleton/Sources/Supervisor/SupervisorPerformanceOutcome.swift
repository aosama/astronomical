import Foundation

/**
 * Explicit outcome for one measured supervisor boundary, mirroring the Rust
 * enum from apps/supervisor/src/supervisor_performance_record.rs.
 */
public enum SupervisorPerformanceOutcome: Equatable, Sendable {

    case success
    case failure
    case paused
    case cancelled

    public var wireName: String {
        switch (self) {
        case .success: return "success"
        case .failure: return "failure"
        case .paused: return "paused"
        case .cancelled: return "cancelled"
        }
    }
}
