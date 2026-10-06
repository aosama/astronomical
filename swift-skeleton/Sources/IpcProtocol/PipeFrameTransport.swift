import Foundation;

/// Byte transport the framed protocol reader and writer run over.
///
/// UnixSocketStream is the local daemon and worker socket transport; the
/// pipe transport carries the same framing over a worker process's stdio
/// pipes. The protocol bytes are identical in both worlds.
public protocol FrameTransport: AnyObject {

    /// Reads up to the destination buffer's capacity, returning the received
    /// byte count; zero means the transport reached its end.
    func readSome(into destinationBuffer: inout Array<UInt8>) throws -> Int;

    /// Writes every outgoing byte or fails.
    func writeAll(_ outgoingBytes: Data) throws;

    /// Half-closes the write side to deliver EOF to the peer.
    func shutdownWrite();

    /// Closes the underlying descriptor outright; the worker-process owner
    /// calls this for the read end after reaping a replaced child.
    func closeTransportFileDescriptor();

    /// Reports whether bytes are ready within `timeoutMilliseconds` without
    /// consuming them; the worker's decode loop uses this to interleave
    /// supervisor commands between engine steps.
    func pollReadReadiness(timeoutMilliseconds: Int32) -> Bool;
}

#if canImport(Glibc)
import Glibc;
#else
import Darwin;
#endif

/// Carries framed protocol bytes over one pipe file descriptor pair owned by
/// a child process (the supervisor writes commands to the worker's stdin and
/// reads events from its stdout).
public final class PipeFrameTransport: FrameTransport {

    private let fileDescriptor: Int32;
    private let isWriteEnd: Bool;

    /// Takes ownership of one end of a child-process pipe. The caller must
    /// not close the descriptor afterwards.
    public init(fileDescriptor: Int32, isWriteEnd: Bool) {
        self.fileDescriptor = fileDescriptor;
        self.isWriteEnd = isWriteEnd;
    }

    public func readSome(into destinationBuffer: inout Array<UInt8>) throws -> Int {
        let receivedByteCount: Int = read(
            self.fileDescriptor,
            &destinationBuffer,
            destinationBuffer.count);
        if receivedByteCount > 0 {
            return receivedByteCount;
        }
        if receivedByteCount == 0 {
            return 0;
        }
        throw IpcPosixError.fromErrno();
    }

    public func writeAll(_ outgoingBytes: Data) throws {
        var writtenByteCount: Int = 0;
        try outgoingBytes.withUnsafeBytes { (rawBytes: UnsafeRawBufferPointer) -> Void in
            while writtenByteCount < outgoingBytes.count {
                let writeResult: Int = write(
                    self.fileDescriptor,
                    rawBytes.baseAddress!.advanced(by: writtenByteCount),
                    outgoingBytes.count - writtenByteCount);
                if writeResult > 0 {
                    writtenByteCount += writeResult;
                    continue;
                }
                throw IpcPosixError.fromErrno();
            }
        };
    }

    /// Half-closes the supervisor's command side: the worker reads EOF on its
    /// stdin, which is the graceful-shutdown signal in the Rust supervisor.
    /// Only the write end supports half-close; a read end is simply closed,
    /// which the worker-process owner does after reaping the child.
    public func shutdownWrite() {
        if self.isWriteEnd {
            close(self.fileDescriptor);
        }
    }

    public func closeTransportFileDescriptor() {
        close(self.fileDescriptor);
    }

    public func pollReadReadiness(timeoutMilliseconds: Int32) -> Bool {
        if self.isWriteEnd {
            return false;
        }
        return FrameTransportPolling.isReadReady(
            fileDescriptor: self.fileDescriptor,
            timeoutMilliseconds: timeoutMilliseconds);
    }
}
