import Foundation;

/// Client one ephemeral CLI process uses to talk to the local daemon.
public final class DaemonIpcClient {
    private let connectionSocket: UnixSocketStream;
    private let protocolReader: ProtocolReader;
    private let protocolWriter: ProtocolWriter;

    /// Connects to the daemon socket, reporting a missing or dead daemon as
    /// DaemonTransportError.daemonNotRunning instead of a raw IO failure, the
    /// way the Rust client classifies NotFound and ConnectionRefused.
    public static func connect(socketPath: String, performanceAttributionEnabled: Bool = false) throws -> DaemonIpcClient {
        let connectStart: ContinuousClock.Instant? = IpcProtocolPerformanceAttribution.startedOperation(
            operationName: "ipc_client_connect", performanceAttributionEnabled: performanceAttributionEnabled);
        do {
            let connectedStream: UnixSocketStream;
            do {
                connectedStream = try UnixSocketStream.connect(path: socketPath);
            } catch let posixError as IpcPosixError {
                if posixError.errnoValue == ENOENT || posixError.errnoValue == ECONNREFUSED {
                    throw DaemonTransportError.daemonNotRunning(socketPath: socketPath);
                }
                throw DaemonTransportError.connectFailed(socketPath: socketPath, source: posixError.ioError);
            }
            let connectedClient: DaemonIpcClient = DaemonIpcClient(
                connectionSocket: connectedStream, performanceAttributionEnabled: performanceAttributionEnabled);
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_client_connect", operationStart: connectStart,
                operationOutcome: "success", performanceAttributionEnabled: performanceAttributionEnabled);
            return connectedClient;
        } catch {
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_client_connect", operationStart: connectStart,
                operationOutcome: "failure", performanceAttributionEnabled: performanceAttributionEnabled);
            throw error;
        }
    }

    private init(connectionSocket: UnixSocketStream, performanceAttributionEnabled: Bool = false) {
        self.connectionSocket = connectionSocket;
        self.protocolReader = ProtocolReader(socket: connectionSocket, performanceAttributionEnabled: performanceAttributionEnabled);
        self.protocolWriter = ProtocolWriter(socket: connectionSocket, performanceAttributionEnabled: performanceAttributionEnabled);
    }

    /// Transmits one request to the daemon.
    public func sendRequest(_ daemonRequest: DaemonRequest) throws -> Void {
        do {
            try self.protocolWriter.sendDaemonRequest(daemonRequest);
        } catch let protocolError as ProtocolError {
            throw DaemonTransportError.protocolFailure(protocolError);
        }
    }

    /// Reads the next response, or `nil` when the daemon closes the connection.
    public func nextResponse() throws -> DaemonResponse? {
        do {
            return try self.protocolReader.nextDaemonResponse();
        } catch let protocolError as ProtocolError {
            throw DaemonTransportError.protocolFailure(protocolError);
        }
    }
}
