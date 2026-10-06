import Foundation;

/// Swift stand-in for `tokio::net::UnixStream`: one POSIX AF_UNIX stream
/// socket over which the length-delimited IPC frames travel. The Rust
/// transport splits every stream into owned read and write halves; this class
/// keeps one file descriptor that ProtocolReader and ProtocolWriter share,
/// because the Swift frame codec is synchronous and runs on the caller's
/// thread instead of inside an async runtime.
public final class UnixSocketStream: FrameTransport {
    private let fileDescriptor: Int32;
    private let lifecycleLock: NSLock = NSLock();
    private var isFileDescriptorClosed: Bool = false;

    internal init(ownedFileDescriptor: Int32) {
        self.fileDescriptor = ownedFileDescriptor;
    }

    deinit {
        self.close();
    }

    /// Suppresses SIGPIPE so writes to a vanished peer return EPIPE like they
    /// do in Rust, whose std sets SO_NOSIGPIPE on every socket it creates;
    /// without this a mid-stream client death kills the whole process.
    internal static func suppressSigpipe(fileDescriptor: Int32) throws -> Void {
        var sigpipeDisabled: Int32 = 1;
        let suppressResult: Int32 = setsockopt(
            fileDescriptor, SOL_SOCKET, SO_NOSIGPIPE, &sigpipeDisabled,
            socklen_t(MemoryLayout<Int32>.size));
        if suppressResult < 0 {
            throw IpcPosixError.fromErrno();
        }
    }

    /// Connects one client stream to the unix socket at `socketPath`.
    public static func connect(path socketPath: String) throws -> UnixSocketStream {
        let clientFileDescriptor: Int32 = socket(AF_UNIX, SOCK_STREAM, 0);
        if clientFileDescriptor < 0 {
            throw IpcPosixError.fromErrno();
        }
        do {
            try UnixSocketStream.suppressSigpipe(fileDescriptor: clientFileDescriptor);
            try UnixSocketStream.connectFileDescriptor(clientFileDescriptor, toPath: socketPath);
        } catch {
            Darwin.close(clientFileDescriptor);
            throw error;
        }
        return UnixSocketStream(ownedFileDescriptor: clientFileDescriptor);
    }

    /// Builds the sockaddr_un for `socketPath`, rejecting paths that cannot fit
    /// a sun_path the same way tokio rejects them before calling the OS.
    internal static func unixSocketAddress(path socketPath: String) throws -> sockaddr_un {
        var socketAddress: sockaddr_un = sockaddr_un();
        let pathByteCount: Int = socketPath.utf8CString.count;
        if pathByteCount > MemoryLayout.size(ofValue: socketAddress.sun_path) {
            throw IpcIoError(underlyingErrorDescription: "path must be shorter than SUN_LEN");
        }
        socketAddress.sun_family = sa_family_t(AF_UNIX);
        withUnsafeMutableBytes(of: &socketAddress.sun_path) { (destinationBuffer: UnsafeMutableRawBufferPointer) -> Void in
            socketPath.withCString { (pathPointer: UnsafePointer<CChar>) -> Void in
                memcpy(destinationBuffer.baseAddress, pathPointer, pathByteCount);
            };
        };
        return socketAddress;
    }

    /// Receives at most destinationBuffer.count bytes into the caller's buffer
    /// and returns how many arrived; 0 means the peer closed cleanly (EOF).
    public func readSome(into destinationBuffer: inout Array<UInt8>) throws -> Int {
        while true {
            let receivedByteCount: Int = destinationBuffer.withUnsafeMutableBytes { (rawBuffer: UnsafeMutableRawBufferPointer) -> Int in
                return recv(self.fileDescriptor, rawBuffer.baseAddress, rawBuffer.count, 0);
            };
            if receivedByteCount >= 0 {
                return receivedByteCount;
            }
            if errno == EINTR {
                continue;
            }
            throw IpcPosixError.fromErrno();
        }
    }

    /// Sends every byte of `outgoingBytes`, handling partial sends and EINTR.
    public func writeAll(_ outgoingBytes: Data) throws -> Void {
        var remainingByteCount: Int = outgoingBytes.count;
        while remainingByteCount > 0 {
            let sentByteCount: Int = outgoingBytes.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) -> Int in
                let unsentBaseAddress: UnsafeRawPointer? = rawBuffer.baseAddress?.advanced(by: outgoingBytes.count - remainingByteCount);
                return send(self.fileDescriptor, unsentBaseAddress, remainingByteCount, 0);
            };
            if sentByteCount > 0 {
                remainingByteCount -= sentByteCount;
                continue;
            }
            if sentByteCount == 0 || errno == EINTR {
                continue;
            }
            throw IpcPosixError.fromErrno();
        }
    }

    /// Half-closes the stream so the peer observes EOF while this side keeps
    /// the descriptor open for reads, mirroring dropping the Rust write half.
    /// A failed half-close leaves the caller to close the whole transport, so
    /// the errno is swallowed here exactly like the Rust drop path.
    public func shutdownWrite() {
        let shutdownResult: Int32 = shutdown(self.fileDescriptor, SHUT_WR);
        _ = shutdownResult;
    }

    /// Closes the file descriptor; safe to call repeatedly.
    public func close() -> Void {
        self.lifecycleLock.lock();
        defer { self.lifecycleLock.unlock(); }
        if self.isFileDescriptorClosed {
            return;
        }
        self.isFileDescriptorClosed = true;
        Darwin.close(self.fileDescriptor);
    }

    private static func connectFileDescriptor(_ clientFileDescriptor: Int32, toPath socketPath: String) throws -> Void {
        var socketAddress: sockaddr_un = try UnixSocketStream.unixSocketAddress(path: socketPath);
        let connectResult: Int32 = withUnsafePointer(to: &socketAddress) { (addressPointer: UnsafePointer<sockaddr_un>) -> Int32 in
            return addressPointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { (sockaddrPointer: UnsafePointer<sockaddr>) -> Int32 in
                return Darwin.connect(clientFileDescriptor, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_un>.size));
            };
        };
        if connectResult < 0 {
            throw IpcPosixError.fromErrno();
        }
    }
}
