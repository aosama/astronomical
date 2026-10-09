import Foundation;


/// Writes one fixture file atomically, retrying once on the transient
/// descriptor failures parallel test targets surface under load. The
/// atomic option writes a same-directory temporary file and renames it, so
/// readers never observe a partial fixture; the single retry covers the
/// intermittent `EBADF` the direct-write path hits while pipe-heavy suites
/// run concurrently.
public func writeFixtureData(_ fileData: Data, to fileUrl: URL) throws {
    var lastWriteError: Error? = nil;
    for _ in 0..<2 {
        do {
            try fileData.write(to: fileUrl, options: .atomic);
            return;
        } catch {
            lastWriteError = error;
        }
    }
    guard let recordedWriteError: Error = lastWriteError else {
        return;
    }
    throw recordedWriteError;
}
