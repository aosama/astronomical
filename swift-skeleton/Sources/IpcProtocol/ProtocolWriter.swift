import Foundation;

/// Sends bounded, length-delimited JSON frames to the peer, mirroring the Rust
/// ProtocolWriter over tokio's LengthDelimitedCodec: every frame is prefixed
/// with its four-byte big-endian length, the 32 MiB cap is enforced before any
/// bytes hit the socket, and close() half-closes the stream so the peer
/// observes EOF while reads on the shared descriptor keep working.
public final class ProtocolWriter {
    private let socket: any FrameTransport;
    private let performanceAttributionEnabled: Bool;

    public convenience init(socket: UnixSocketStream, performanceAttributionEnabled: Bool = false) {
        self.init(transport: socket, performanceAttributionEnabled: performanceAttributionEnabled);
    }

    public init(transport: any FrameTransport, performanceAttributionEnabled: Bool = false) {
        self.socket = transport;
        self.performanceAttributionEnabled = performanceAttributionEnabled;
    }

    /// Serializes and transmits one supervisor command frame.
    public func sendCommand(_ workerCommand: WorkerCommand) throws -> Void {
        let serializedCommand: Data = try MessageCodec.encodeCommand(workerCommand);
        return try self.sendSerializedMessage(serializedCommand);
    }

    /// Serializes and transmits one worker event frame.
    public func sendEvent(_ workerEvent: WorkerEvent) throws -> Void {
        let serializedEvent: Data = try MessageCodec.encodeEvent(workerEvent);
        return try self.sendSerializedMessage(serializedEvent);
    }

    /// Serializes and transmits one daemon request frame.
    public func sendDaemonRequest(_ daemonRequest: DaemonRequest) throws -> Void {
        let serializedRequest: Data = try MessageCodec.encodeDaemonRequest(daemonRequest);
        return try self.sendSerializedMessage(serializedRequest);
    }

    /// Serializes and transmits one daemon response frame.
    public func sendDaemonResponse(_ daemonResponse: DaemonResponse) throws -> Void {
        let serializedResponse: Data = try MessageCodec.encodeDaemonResponse(daemonResponse);
        return try self.sendSerializedMessage(serializedResponse);
    }

    /// Half-closes the stream to deliver EOF to the peer.
    public func close() throws -> Void {
        self.socket.shutdownWrite();
    }

    private func sendSerializedMessage(_ serializedMessage: Data) throws -> Void {
        let frameWriteStart: ContinuousClock.Instant? = IpcProtocolPerformanceAttribution.startedOperation(
            operationName: "ipc_write_frame", performanceAttributionEnabled: self.performanceAttributionEnabled);
        do {
            try self.transmitFramedMessage(serializedMessage);
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_write_frame", operationStart: frameWriteStart,
                operationOutcome: "success", performanceAttributionEnabled: self.performanceAttributionEnabled);
        } catch {
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_write_frame", operationStart: frameWriteStart,
                operationOutcome: "failure", performanceAttributionEnabled: self.performanceAttributionEnabled);
            throw error;
        }
    }

    private func transmitFramedMessage(_ serializedMessage: Data) throws -> Void {
        if serializedMessage.count > IpcFrameLimits.maximumIpcFrameBytes {
            throw ProtocolError.outgoingMessageTooLarge(
                actualMessageBytes: serializedMessage.count,
                maximumMessageBytes: IpcFrameLimits.maximumIpcFrameBytes);
        }
        do {
            try self.socket.writeAll(ProtocolWriter.framedBytes(serializedMessage));
        } catch let posixError as IpcPosixError {
            throw ProtocolError.writeFrame(posixError.ioError);
        }
    }

    private static func framedBytes(_ serializedMessage: Data) -> Data {
        var framedBytes: Array<UInt8> = Array<UInt8>();
        let frameLength: UInt32 = UInt32(serializedMessage.count);
        framedBytes.append(UInt8((frameLength >> 24) & 0xFF));
        framedBytes.append(UInt8((frameLength >> 16) & 0xFF));
        framedBytes.append(UInt8((frameLength >> 8) & 0xFF));
        framedBytes.append(UInt8(frameLength & 0xFF));
        framedBytes.append(contentsOf: Array<UInt8>(serializedMessage));
        return Data(framedBytes);
    }
}
