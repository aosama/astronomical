import Foundation;

/// Transport frame budget shared by the message codec and by validators that
/// must bound how far one inbound frame can expand in memory.
internal enum IpcFrameLimits {
    /// Matches the Rust `MAX_IPC_FRAME_BYTES` wire contract.
    internal static let maximumIpcFrameBytes: Int = 32 * 1024 * 1024;
}
