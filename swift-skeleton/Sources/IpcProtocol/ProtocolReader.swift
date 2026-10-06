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

    /// Reads the next supervisor command, or `nil` when the transport closes cleanly.
    public func nextCommand() throws -> WorkerCommand? {
        guard let serializedCommand: Data = try self.nextFrame() else {
            return nil;
        }
        return try MessageCodec.decodeCommand(serializedCommand);
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
