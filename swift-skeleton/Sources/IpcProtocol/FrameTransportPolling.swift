import Foundation;

#if canImport(Glibc)
import Glibc;
#else
import Darwin;
#endif

/// Read-readiness polling for framed transports, mirroring the tokio
/// `select!` arm the Rust engine-backed worker uses to interleave supervisor
/// commands between decode steps. Both concrete transports own one POSIX
/// descriptor, so a shared `poll(2)` helper serves them; in-memory test
/// transports that cannot poll report permanent readiness instead of
/// pretending a timeout.
public enum FrameTransportPolling {

    /// Returns whether the descriptor has bytes ready within
    /// `timeoutMilliseconds`, or immediately when it reached end of stream.
    public static func isReadReady(
        fileDescriptor: Int32,
        timeoutMilliseconds: Int32
    ) -> Bool {
        var pollDescriptor: pollfd = pollfd(
            fd: fileDescriptor,
            events: Int16(POLLIN),
            revents: 0);
        let pollOutcome: Int32 = poll(&pollDescriptor, 1, timeoutMilliseconds);
        if pollOutcome < 0 {
            // A signal interrupt or descriptor trouble leaves the caller on
            // the blocking read path, which owns real error reporting.
            return false;
        }
        if pollOutcome == 0 {
            return false;
        }
        // POLLHUP surfaces end of stream as readable: the follow-up read
        // returns zero and the protocol reader reports the clean close.
        return true;
    }
}
