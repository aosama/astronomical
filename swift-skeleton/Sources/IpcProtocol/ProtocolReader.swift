import Foundation;

/// Receives bounded, length-delimited JSON frames from the peer and decodes
/// them into the four IPC message enums. This mirrors the Rust ProtocolReader
/// over tokio's LengthDelimitedCodec: a four-byte big-endian length prefix,
/// the 32 MiB frame cap enforced before any payload is buffered, clean EOF at
/// a frame boundary reported as nil, and a truncated stream reported as an
/// error. Reads block on the caller's thread the way the tokio reader awaits
/// its transport.
public final class ProtocolReader {
    private static let lengthPrefixByteCount: Int = 4;
    /// recv() scratch size per transport read.
    private static let receiveBufferByteCount: Int = 65_536;

    /// One `poll(2)` wait per partial-frame slice; bounds cancel latency
    /// without spinning when a frame trickles in across transport reads.
    private static let pollSliceMilliseconds: Int32 = 20;

    private let socket: any FrameTransport;
    private var pendingBytes: Array<UInt8> = Array<UInt8>();
    private let performanceAttributionEnabled: Bool;

    public convenience init(socket: UnixSocketStream, performanceAttributionEnabled: Bool = false) {
        self.init(transport: socket, performanceAttributionEnabled: performanceAttributionEnabled);
    }

    public init(transport: any FrameTransport, performanceAttributionEnabled: Bool = false) {
        self.socket = transport;
        self.performanceAttributionEnabled = performanceAttributionEnabled;
    }

    /// Closes the underlying transport descriptor outright; the
    /// worker-process owner calls this for a replaced child's read end.
    public func closeTransportFileDescriptor() -> Void {
        self.socket.closeTransportFileDescriptor();
    }

    /// Reads the next supervisor command, or `nil` when the transport closes cleanly.
    public func nextCommand() throws -> WorkerCommand? {
        guard let serializedCommand: Data = try self.nextFrame() else {
            return nil;
        }
        return try MessageCodec.decodeCommand(serializedCommand);
    }

    /// The outcome of one bounded poll for the next supervisor command.
    public enum PolledCommand {
        /// A complete command frame arrived and decoded.
        case command(WorkerCommand);
        /// No complete frame arrived within the timeout; the stream stays open.
        case none;
        /// The transport closed cleanly; the caller ends the loop.
        case endOfStream;
    }

    /// Waits up to `timeoutMilliseconds` for the next complete command frame.
    ///
    /// The engine-backed worker's decode loop calls this between engine steps
    /// so cancel and memory commands interleave at token boundaries, exactly
    /// as the Rust loop's biased `tokio::select!` interleaves them between
    /// decode yields. A partially received frame keeps polling within the
    /// remaining budget instead of resurfacing as a blocking read.
    public func pollNextCommand(timeoutMilliseconds: Int32) throws -> PolledCommand {
        if self.hasCompleteFrameBuffered() {
            return try self.decodeBufferedCommand();
        }
        let clock: ContinuousClock = ContinuousClock();
        let pollDeadline: ContinuousClock.Instant = clock.now.advanced(
            by: .milliseconds(Int64(timeoutMilliseconds)));
        while clock.now < pollDeadline {
            guard self.socket.pollReadReadiness(
                timeoutMilliseconds: ProtocolReader.pollSliceMilliseconds) else {
                return .none;
            }
            if try self.readMoreBytes() == false {
                return .endOfStream;
            }
            if self.hasCompleteFrameBuffered() {
                return try self.decodeBufferedCommand();
            }
        }
        return .none;
    }

    /// Pops and decodes one already-buffered complete frame as a command.
    private func decodeBufferedCommand() throws -> PolledCommand {
        guard let serializedCommand: Data = try self.nextFrame() else {
            return .endOfStream;
        }
        return .command(try MessageCodec.decodeCommand(serializedCommand));
    }

    /// Returns whether one complete frame already sits in the pending buffer.
    private func hasCompleteFrameBuffered() -> Bool {
        guard self.pendingBytes.count >= ProtocolReader.lengthPrefixByteCount else {
            return false;
        }
        let bufferedFrameLength: Int = self.pendingFrameLength();
        if bufferedFrameLength > IpcFrameLimits.maximumIpcFrameBytes {
            return false;
        }
        return self.pendingBytes.count
            >= ProtocolReader.lengthPrefixByteCount + bufferedFrameLength;
    }

    /// Reads the next worker event, or `nil` when the transport closes cleanly.
    public func nextEvent() throws -> WorkerEvent? {
        guard let serializedEvent: Data = try self.nextFrame() else {
            return nil;
        }
        return try MessageCodec.decodeEvent(serializedEvent);
    }

    /// Reads the next daemon request, or `nil` when the transport closes cleanly.
    public func nextDaemonRequest() throws -> DaemonRequest? {
        guard let serializedRequest: Data = try self.nextFrame() else {
            return nil;
        }
        return try MessageCodec.decodeDaemonRequest(serializedRequest);
    }

    /// Reads the next daemon response, or `nil` when the transport closes cleanly.
    public func nextDaemonResponse() throws -> DaemonResponse? {
        guard let serializedResponse: Data = try self.nextFrame() else {
            return nil;
        }
        return try MessageCodec.decodeDaemonResponse(serializedResponse);
    }

    private func nextFrame() throws -> Data? {
        let frameReadStart: ContinuousClock.Instant? = IpcProtocolPerformanceAttribution.startedOperation(
            operationName: "ipc_read_frame", performanceAttributionEnabled: self.performanceAttributionEnabled);
        do {
            let framedMessage: Data? = try self.readFramedMessage();
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_read_frame", operationStart: frameReadStart,
                operationOutcome: framedMessage == nil ? "closed" : "success",
                performanceAttributionEnabled: self.performanceAttributionEnabled);
            return framedMessage;
        } catch {
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_read_frame", operationStart: frameReadStart,
                operationOutcome: "failure", performanceAttributionEnabled: self.performanceAttributionEnabled);
            throw error;
        }
    }

    private func readFramedMessage() throws -> Data? {
        while self.pendingBytes.count < ProtocolReader.lengthPrefixByteCount {
            if try self.readMoreBytes() == false {
                if self.pendingBytes.isEmpty {
                    return nil;
                }
                throw ProtocolError.readFrame(IpcIoError(underlyingErrorDescription: "bytes remaining on stream"));
            }
        }
        let frameLength: Int = self.pendingFrameLength();
        if frameLength > IpcFrameLimits.maximumIpcFrameBytes {
            throw ProtocolError.readFrame(IpcIoError(underlyingErrorDescription: "frame size too big"));
        }
        let frameEndOffset: Int = ProtocolReader.lengthPrefixByteCount + frameLength;
        while self.pendingBytes.count < frameEndOffset {
            if try self.readMoreBytes() == false {
                throw ProtocolError.readFrame(IpcIoError(underlyingErrorDescription: "bytes remaining on stream"));
            }
        }
        let frameBytes: Array<UInt8> = Array(self.pendingBytes[ProtocolReader.lengthPrefixByteCount ..< frameEndOffset]);
        self.pendingBytes.removeFirst(frameEndOffset);
        return Data(frameBytes);
    }

    private func pendingFrameLength() -> Int {
        var frameLength: UInt32 = 0;
        for prefixByte in self.pendingBytes.prefix(ProtocolReader.lengthPrefixByteCount) {
            frameLength = (frameLength << 8) | UInt32(prefixByte);
        }
        return Int(frameLength);
    }

    /// Reads one transport chunk into the pending buffer; returns false on a
    /// clean peer EOF.
    private func readMoreBytes() throws -> Bool {
        var receiveBuffer: Array<UInt8> = Array<UInt8>(repeating: 0, count: ProtocolReader.receiveBufferByteCount);
        let receivedByteCount: Int = try self.socket.readSome(into: &receiveBuffer);
        if receivedByteCount == 0 {
            return false;
        }
        self.pendingBytes.append(contentsOf: receiveBuffer[0 ..< receivedByteCount]);
        return true;
    }
}
