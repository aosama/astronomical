import Foundation;

/**
 * One accepted loopback TCP connection: buffered reads for the HTTP grammar,
 * capped writes, and the socket timeouts that keep a silent or stalled client
 * from pinning a serving thread forever. The connection is single-use — the
 * endpoint answers exactly one request and closes.
 */
public final class RestHttpConnection {

    private static let RECEIVE_TIMEOUT_SECONDS: Int = 30;
    private static let SEND_TIMEOUT_SECONDS: Int = 30;
    private static let READ_CHUNK_BYTE_COUNT: Int = 4096;
    private static let HEADER_BLOCK_TERMINATOR: Data = Data("\r\n\r\n".utf8);

    private let fileDescriptor: Int32;
    /// Bytes read from the socket but not yet consumed by the HTTP grammar.
    private var pendingBytes: Data;

    public init(fileDescriptor: Int32) {
        self.fileDescriptor = fileDescriptor;
        self.pendingBytes = Data();

        var noSigPipeFlag: Int32 = 1;
        _ = setsockopt(
            fileDescriptor, SOL_SOCKET, SO_NOSIGPIPE,
            &noSigPipeFlag, socklen_t(MemoryLayout<Int32>.size));
        var noDelayFlag: Int32 = 1;
        _ = setsockopt(
            fileDescriptor, Int32(IPPROTO_TCP), TCP_NODELAY,
            &noDelayFlag, socklen_t(MemoryLayout<Int32>.size));
        var receiveTimeout: timeval = timeval(
            tv_sec: RestHttpConnection.RECEIVE_TIMEOUT_SECONDS, tv_usec: 0);
        _ = setsockopt(
            fileDescriptor, SOL_SOCKET, SO_RCVTIMEO,
            &receiveTimeout, socklen_t(MemoryLayout<timeval>.size));
        var sendTimeout: timeval = timeval(
            tv_sec: RestHttpConnection.SEND_TIMEOUT_SECONDS, tv_usec: 0);
        _ = setsockopt(
            fileDescriptor, SOL_SOCKET, SO_SNDTIMEO,
            &sendTimeout, socklen_t(MemoryLayout<timeval>.size));
    }

    /// Reads bytes up to and including the blank line that ends the header
    /// block. Bytes past the terminator stay buffered for the body read.
    public func readHeaderBlock(maximumHeaderBlockBytes: UInt) throws -> Data {
        var searchOffset: Int = 0;
        while true {
            if let terminatorRange: Range<Data.Index> = self.pendingBytes.range(of: RestHttpConnection.HEADER_BLOCK_TERMINATOR, in: searchOffset..<self.pendingBytes.count) {
                let headerBlockEndIndex: Int = terminatorRange.upperBound;
                let headerBlockBytes: Data = Data(self.pendingBytes.prefix(headerBlockEndIndex));
                self.pendingBytes = self.pendingBytes.subdata(in: headerBlockEndIndex..<self.pendingBytes.count);
                return headerBlockBytes;
            }
            // Restart the terminator search only past what was already
            // scanned, so a terminator split across reads is still found.
            searchOffset = max(0, self.pendingBytes.count - (RestHttpConnection.HEADER_BLOCK_TERMINATOR.count - 1));
            if UInt(self.pendingBytes.count) > maximumHeaderBlockBytes {
                throw RestEndpointFailure(statusCode: 400, message: "the request header block exceeds the allowed size");
            }
            let nextChunkBytes: Data = try self.readChunkFromSocket();
            if nextChunkBytes.isEmpty {
                throw RestConnectionError.clientDisconnected;
            }
            self.pendingBytes.append(nextChunkBytes);
        }
    }

    /// Consumes exactly the given body byte count, reading from the socket
    /// only for what the buffered remainder does not already cover.
    public func readBodyBytes(byteCount: UInt) throws -> Data {
        if byteCount == 0 {
            return Data();
        }
        while UInt(self.pendingBytes.count) < byteCount {
            let nextChunkBytes: Data = try self.readChunkFromSocket();
            if nextChunkBytes.isEmpty {
                throw RestConnectionError.clientDisconnected;
            }
            self.pendingBytes.append(nextChunkBytes);
        }
        let bodyBytes: Data = Data(self.pendingBytes.prefix(Int(byteCount)));
        self.pendingBytes = self.pendingBytes.subdata(in: Int(byteCount)..<self.pendingBytes.count);
        return bodyBytes;
    }

    /// Writes the interim 100 Continue answer for clients that asked for it.
    public func writeInterimContinue() throws -> Void {
        try self.writeBytes(Data("HTTP/1.1 100 Continue\r\n\r\n".utf8));
    }

    public func writeBytes(_ bytes: Data) throws -> Void {
        var writeOffset: Int = 0;
        while writeOffset < bytes.count {
            let writeOutcome: Int = bytes.withUnsafeBytes({ (rawBuffer: UnsafeRawBufferPointer) -> Int in
                guard let basePointer: UnsafeRawPointer = rawBuffer.baseAddress else {
                    return -1;
                }
                return send(self.fileDescriptor, basePointer + writeOffset, bytes.count - writeOffset, 0);
            });
            if writeOutcome <= 0 {
                throw RestConnectionError.clientDisconnected;
            }
            writeOffset += writeOutcome;
        }
    }

    public func discard() -> Void {
        close(self.fileDescriptor);
    }

    private func readChunkFromSocket() throws -> Data {
        var readBuffer: Array<UInt8> = Array(repeating: 0, count: RestHttpConnection.READ_CHUNK_BYTE_COUNT);
        let bytesRead: Int = read(self.fileDescriptor, &readBuffer, readBuffer.count);
        if bytesRead > 0 {
            return Data(readBuffer[0..<bytesRead]);
        }
        if bytesRead == 0 {
            // The client closed its side without finishing the request.
            return Data();
        }
        if errno == EAGAIN || errno == EWOULDBLOCK {
            throw RestConnectionError.receiveTimedOut;
        }
        throw RestConnectionError.clientDisconnected;
    }
}

/// Socket-level conditions under which no HTTP answer is possible; the
/// transport drops the connection silently instead of fabricating a reply.
public enum RestConnectionError: Error {
    case clientDisconnected;
    case receiveTimedOut;
}
