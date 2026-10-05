import Foundation;

/// Unix-socket listener serving one ephemeral local request per connection.
///
/// The socket lives beside the instance lock file inside the instance state
/// directory so every runtime instance owns an isolated endpoint without any
/// extra configuration.
public final class DaemonIpcListener {
    private let listenerFileDescriptor: Int32;
    private let boundSocketPath: String;
    private let performanceAttributionEnabled: Bool;

    /// Binds the owner-only daemon socket, replacing a stale socket file left
    /// behind by a crashed daemon while refusing to take over a socket a live
    /// daemon still answers on.
    public static func bind(socketPath: String, performanceAttributionEnabled: Bool = false) throws -> DaemonIpcListener {
        let bindStart: ContinuousClock.Instant? = IpcProtocolPerformanceAttribution.startedOperation(
            operationName: "ipc_listener_bind", performanceAttributionEnabled: performanceAttributionEnabled);
        do {
            let boundListener: DaemonIpcListener = try DaemonIpcListener.bindAndConfigure(
                socketPath: socketPath, performanceAttributionEnabled: performanceAttributionEnabled);
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_listener_bind", operationStart: bindStart,
                operationOutcome: "success", performanceAttributionEnabled: performanceAttributionEnabled);
            return boundListener;
        } catch {
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_listener_bind", operationStart: bindStart,
                operationOutcome: "failure", performanceAttributionEnabled: performanceAttributionEnabled);
            throw error;
        }
    }

    private static func bindAndConfigure(socketPath: String, performanceAttributionEnabled: Bool) throws -> DaemonIpcListener {
        let socketParentDirectory: String = (socketPath as NSString).deletingLastPathComponent;
        if socketParentDirectory.isEmpty || FileManager.default.fileExists(atPath: socketParentDirectory) == false {
            throw DaemonTransportError.socketParentDirectoryMissing(socketPath: socketPath);
        }
        if FileManager.default.fileExists(atPath: socketPath) {
            try DaemonIpcListener.refuseLiveSocketOrRemoveStaleFile(socketPath: socketPath);
        }
        let boundFileDescriptor: Int32;
        do {
            boundFileDescriptor = try DaemonIpcListener.bindUnixListenerSocket(socketPath: socketPath);
        } catch let posixError as IpcPosixError {
            throw DaemonTransportError.bindFailed(socketPath: socketPath, source: posixError.ioError);
        }
        let chmodResult: Int32 = chmod(socketPath, 0o600);
        if chmodResult < 0 {
            let permissionFailure: IpcPosixError = IpcPosixError.fromErrno();
            close(boundFileDescriptor);
            throw DaemonTransportError.socketPermissionDenied(socketPath: socketPath, source: permissionFailure.ioError);
        }
        return DaemonIpcListener(
            listenerFileDescriptor: boundFileDescriptor, boundSocketPath: socketPath,
            performanceAttributionEnabled: performanceAttributionEnabled);
    }

    private init(listenerFileDescriptor: Int32, boundSocketPath: String, performanceAttributionEnabled: Bool = false) {
        self.listenerFileDescriptor = listenerFileDescriptor;
        self.boundSocketPath = boundSocketPath;
        self.performanceAttributionEnabled = performanceAttributionEnabled;
    }

    deinit {
        close(self.listenerFileDescriptor);
    }

    /// Socket file path this listener serves.
    public var socketPath: String {
        return self.boundSocketPath;
    }

    /// Accepts one connection, answers one request, then closes the
    /// connection. A peer that disconnects before sending a request is not an
    /// error so the listener stays available for the next local client.
    public func serveNextRequest(handleRequest: (DaemonRequest) -> DaemonResponse) throws -> Void {
        let connectionSocket: UnixSocketStream = try self.acceptConnection();
        defer { connectionSocket.close(); }
        let protocolReader: ProtocolReader = ProtocolReader(socket: connectionSocket, performanceAttributionEnabled: self.performanceAttributionEnabled);
        let protocolWriter: ProtocolWriter = ProtocolWriter(socket: connectionSocket, performanceAttributionEnabled: self.performanceAttributionEnabled);
        let daemonRequest: DaemonRequest?;
        do {
            daemonRequest = try protocolReader.nextDaemonRequest();
        } catch let protocolError as ProtocolError {
            throw DaemonTransportError.protocolFailure(protocolError);
        }
        guard let unwrappedRequest: DaemonRequest = daemonRequest else {
            return;
        }
        let daemonResponse: DaemonResponse = handleRequest(unwrappedRequest);
        do {
            try protocolWriter.sendDaemonResponse(daemonResponse);
            try protocolWriter.close();
        } catch let protocolError as ProtocolError {
            throw DaemonTransportError.protocolFailure(protocolError);
        }
    }

    /// Accepts one connection and hands the request plus the streaming writer
    /// to the handler, which sends every response frame before the connection
    /// closes. Like serveNextRequest, a peer that disconnects before sending a
    /// request is not an error.
    public func serveStreamingRequest(handleRequest: (DaemonRequest, StreamingResponseWriter) throws -> Void) throws -> Void {
        let connectionSocket: UnixSocketStream = try self.acceptConnection();
        defer { connectionSocket.close(); }
        let protocolReader: ProtocolReader = ProtocolReader(socket: connectionSocket, performanceAttributionEnabled: self.performanceAttributionEnabled);
        let protocolWriter: ProtocolWriter = ProtocolWriter(socket: connectionSocket, performanceAttributionEnabled: self.performanceAttributionEnabled);
        let daemonRequest: DaemonRequest?;
        do {
            daemonRequest = try protocolReader.nextDaemonRequest();
        } catch let protocolError as ProtocolError {
            throw DaemonTransportError.protocolFailure(protocolError);
        }
        guard let unwrappedRequest: DaemonRequest = daemonRequest else {
            return;
        }
        let streamingResponseWriter: StreamingResponseWriter = StreamingResponseWriter(protocolWriter: protocolWriter);
        try handleRequest(unwrappedRequest, streamingResponseWriter);
    }

    private func acceptConnection() throws -> UnixSocketStream {
        let acceptStart: ContinuousClock.Instant? = IpcProtocolPerformanceAttribution.startedOperation(
            operationName: "ipc_listener_accept", performanceAttributionEnabled: self.performanceAttributionEnabled);
        do {
            let connectionSocket: UnixSocketStream = try self.acceptAndWrapConnection();
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_listener_accept", operationStart: acceptStart,
                operationOutcome: "success", performanceAttributionEnabled: self.performanceAttributionEnabled);
            return connectionSocket;
        } catch {
            IpcProtocolPerformanceAttribution.finishedOperation(
                operationName: "ipc_listener_accept", operationStart: acceptStart,
                operationOutcome: "failure", performanceAttributionEnabled: self.performanceAttributionEnabled);
            throw error;
        }
    }

    private func acceptAndWrapConnection() throws -> UnixSocketStream {
        let connectionFileDescriptor: Int32 = accept(self.listenerFileDescriptor, nil, nil);
        if connectionFileDescriptor < 0 {
            throw DaemonTransportError.acceptFailed(source: IpcPosixError.fromErrno().ioError);
        }
        // Best effort: macOS hands back a dead descriptor for a queued
        // connection whose peer already vanished, and setsockopt on it fails
        // with EINVAL; reads still surface EOF through the normal path, and a
        // write is only ever attempted after a full frame was read from a
        // live peer, so suppression is always armed where writes can happen.
        try? UnixSocketStream.suppressSigpipe(fileDescriptor: connectionFileDescriptor);
        return UnixSocketStream(ownedFileDescriptor: connectionFileDescriptor);
    }

    private static func bindUnixListenerSocket(socketPath: String) throws -> Int32 {
        let listenerFileDescriptor: Int32 = socket(AF_UNIX, SOCK_STREAM, 0);
        if listenerFileDescriptor < 0 {
            throw IpcPosixError.fromErrno();
        }
        do {
            try UnixSocketStream.suppressSigpipe(fileDescriptor: listenerFileDescriptor);
        } catch {
            close(listenerFileDescriptor);
            throw error;
        }
        var socketAddress: sockaddr_un;
        do {
            socketAddress = try UnixSocketStream.unixSocketAddress(path: socketPath);
        } catch {
            close(listenerFileDescriptor);
            throw error;
        }
        let bindResult: Int32 = withUnsafePointer(to: &socketAddress) { (addressPointer: UnsafePointer<sockaddr_un>) -> Int32 in
            return addressPointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { (sockaddrPointer: UnsafePointer<sockaddr>) -> Int32 in
                return Darwin.bind(listenerFileDescriptor, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_un>.size));
            };
        };
        if bindResult < 0 {
            let bindFailure: IpcPosixError = IpcPosixError.fromErrno();
            close(listenerFileDescriptor);
            throw bindFailure;
        }
        // Rust's UnixListener::bind completes socket, bind, and listen together;
        // without listen the kernel refuses every connection and accept fails.
        let listenResult: Int32 = listen(listenerFileDescriptor, SOMAXCONN);
        if listenResult < 0 {
            let listenFailure: IpcPosixError = IpcPosixError.fromErrno();
            close(listenerFileDescriptor);
            throw listenFailure;
        }
        return listenerFileDescriptor;
    }

    private static func refuseLiveSocketOrRemoveStaleFile(socketPath: String) throws -> Void {
        do {
            let liveConnection: UnixSocketStream = try UnixSocketStream.connect(path: socketPath);
            liveConnection.close();
            // A successful connect proves a live daemon still owns the socket; a
            // rename-based takeover would silently steal its endpoint otherwise.
            throw DaemonTransportError.bindFailed(
                socketPath: socketPath,
                source: IpcIoError(underlyingErrorDescription: "address in use"));
        } catch let refusalError as DaemonTransportError {
            throw refusalError;
        } catch {
            // No daemon answers the probe, so the socket file is stale residue.
            do {
                try FileManager.default.removeItem(atPath: socketPath);
            } catch let removalError as NSError {
                throw DaemonTransportError.socketCleanupFailed(
                    socketPath: socketPath,
                    source: IpcIoError(underlyingErrorDescription: removalError.localizedDescription));
            }
        }
    }
}

/// Owns the write side of one accepted daemon IPC connection so a handler can
/// stream multiple response frames before the connection closes.
public final class StreamingResponseWriter {
    private let protocolWriter: ProtocolWriter;

    internal init(protocolWriter: ProtocolWriter) {
        self.protocolWriter = protocolWriter;
    }

    /// Transmits one response frame, keeping the connection open.
    public func sendResponse(_ daemonResponse: DaemonResponse) throws -> Void {
        do {
            try self.protocolWriter.sendDaemonResponse(daemonResponse);
        } catch let protocolError as ProtocolError {
            throw DaemonTransportError.protocolFailure(protocolError);
        }
    }

    /// Half-closes the stream to deliver EOF to the CLI process.
    public func close() throws -> Void {
        do {
            try self.protocolWriter.close();
        } catch let protocolError as ProtocolError {
            throw DaemonTransportError.protocolFailure(protocolError);
        }
    }
}
