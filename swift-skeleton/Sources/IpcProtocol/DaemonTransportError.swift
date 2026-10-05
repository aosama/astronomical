import Foundation;

/// Failures of the local unix-socket transport between the daemon and a CLI
/// process. Descriptions mirror the Rust thiserror Display strings byte for
/// byte, including the "(os error N)" source rendering.
public enum DaemonTransportError: Error, CustomStringConvertible {
    case socketParentDirectoryMissing(socketPath: String);
    case bindFailed(socketPath: String, source: IpcIoError);
    case socketPermissionDenied(socketPath: String, source: IpcIoError);
    case daemonNotRunning(socketPath: String);
    case connectFailed(socketPath: String, source: IpcIoError);
    case acceptFailed(source: IpcIoError);
    case socketCleanupFailed(socketPath: String, source: IpcIoError);
    /// Carries the rendered failure because the Rust variant wraps a tokio
    /// JoinError, which has no Swift counterpart in this port yet.
    case serviceTaskFailed(description: String);
    /// Named protocolFailure because `protocol` is a reserved Swift keyword.
    case protocolFailure(ProtocolError);

    public var description: String {
        switch (self) {
        case .socketParentDirectoryMissing(let socketPath):
            return "daemon IPC socket parent directory is missing: \(socketPath)";
        case .bindFailed(let socketPath, let source):
            return "daemon IPC socket could not be bound at \(socketPath): \(source)";
        case .socketPermissionDenied(let socketPath, let source):
            return "daemon IPC socket permissions could not be restricted at \(socketPath): \(source)";
        case .daemonNotRunning(let socketPath):
            return "no daemon is running at the IPC socket \(socketPath)";
        case .connectFailed(let socketPath, let source):
            return "daemon IPC connection failed at \(socketPath): \(source)";
        case .acceptFailed(let source):
            return "daemon IPC listener failed to accept a connection: \(source)";
        case .socketCleanupFailed(let socketPath, let source):
            return "daemon IPC socket file could not be cleaned up at \(socketPath): \(source)";
        case .serviceTaskFailed(let description):
            return "daemon IPC service task ended unexpectedly: \(description)";
        case .protocolFailure(let protocolError):
            return "daemon IPC protocol failure: \(protocolError)";
        }
    }
}
