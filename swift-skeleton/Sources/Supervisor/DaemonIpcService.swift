import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// One running daemon IPC service bound to the instance socket.
///
/// Mirrors apps/supervisor/src/daemon_ipc.rs: ephemeral CLI verbs talk over
/// the owner-only unix socket, one request is served per connection, and the
/// service shares no transport with the REST surface. Every verb is served
/// through the streaming writer — simple verbs send one frame and close,
/// exactly as the Rust service does — and the chat verb streams ordered
/// frames until a terminal one. Library and embeddings handlers land with
/// their supervisor slices; those verbs reject with the same wire frame the
/// Rust service uses for rejected requests.
public final class DaemonIpcService: @unchecked Sendable {

    private let listener: DaemonIpcListener;
    private let socketFilePath: String;
    private let instancePaths: AstronomicalInstancePaths;
    private let stateLock: NSLock;
    private let healthProvider: () -> DaemonStatusReport;
    private let chatContext: DaemonIpcChatContext?;
    private let modelsContext: DaemonIpcModelsContext?;
    private let requestIdAllocator: ChatRequestIdAllocator;
    private var isShutdownRequested: Bool;
    private var serviceThread: Thread?;

    private init(
        listener: DaemonIpcListener,
        socketFilePath: String,
        instancePaths: AstronomicalInstancePaths,
        healthProvider: @escaping () -> DaemonStatusReport,
        chatContext: DaemonIpcChatContext?,
        modelsContext: DaemonIpcModelsContext?
    ) {
        self.listener = listener;
        self.socketFilePath = socketFilePath;
        self.instancePaths = instancePaths;
        self.healthProvider = healthProvider;
        self.chatContext = chatContext;
        self.modelsContext = modelsContext;
        self.requestIdAllocator = ChatRequestIdAllocator();
        self.stateLock = NSLock();
        self.isShutdownRequested = false;
    }

    /// The socket file path the service is serving on.
    public var socketPath: String {
        return self.socketFilePath;
    }

    /// Starts serving ephemeral local daemon requests on the instance socket.
    ///
    /// The accept loop runs on its own thread because the sync listener blocks
    /// per connection. Shutdown wakes the blocked accept with a throwaway
    /// local connection, so stopping the service never depends on a client.
    public static func start(
        instancePaths: AstronomicalInstancePaths,
        healthProvider: @escaping () -> DaemonStatusReport,
        chatContext: DaemonIpcChatContext? = nil,
        modelsContext: DaemonIpcModelsContext? = nil
    ) throws -> DaemonIpcService {
        let socketFilePath: String = instancePaths.ipcSocketFilePath.string;
        let listener: DaemonIpcListener = try DaemonIpcListener.bind(
            socketPath: socketFilePath);
        let service: DaemonIpcService = DaemonIpcService(
            listener: listener,
            socketFilePath: socketFilePath,
            instancePaths: instancePaths,
            healthProvider: healthProvider,
            chatContext: chatContext,
            modelsContext: modelsContext
        );
        let serviceThread: Thread = Thread {
            service.serveUntilShutdown();
        };
        serviceThread.name = "astronomicald-daemon-ipc";
        serviceThread.start();
        service.stateLock.lock();
        service.serviceThread = serviceThread;
        service.stateLock.unlock();
        return service;
    }

    /// Stops serving, waits for the serving thread to end, and removes the
    /// socket file so clients cannot discover a dead endpoint.
    public func shutdown() {
        self.stateLock.lock();
        self.isShutdownRequested = true;
        let serviceThread: Thread? = self.serviceThread;
        self.stateLock.unlock();
        // A throwaway connection unblocks the accept that is waiting for a
        // client; the loop then observes the flag and exits.
        let wakeupClient: DaemonIpcClient? = try? DaemonIpcClient.connect(
            socketPath: self.socketFilePath);
        _ = try? wakeupClient?.sendRequest(DaemonRequest.handshake);
        _ = wakeupClient;
        if let serviceThread: Thread = serviceThread {
            // A timeout keeps a wedged transport from hanging the caller;
            // the daemon exit path removes the socket file regardless.
            let joinDeadline: Date = Date().addingTimeInterval(5);
            while serviceThread.isExecuting && Date() < joinDeadline {
                Thread.sleep(forTimeInterval: 0.01);
            }
        }
        removeSocketFileIgnoringMissing(socketFilePath: self.socketFilePath);
    }

    private func serveUntilShutdown() {
        while true {
            self.stateLock.lock();
            let shouldShutdown: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if shouldShutdown {
                return;
            }
            // One misbehaving local client must not take the endpoint down;
            // only a failed accept ends the loop, exactly as the Rust service
            // treats AcceptFailed as terminal and every request error as
            // survivable.
            do {
                try self.listener.serveStreamingRequest { (daemonRequest: DaemonRequest, streamingResponseWriter: StreamingResponseWriter) throws -> Void in
                    try self.handleDaemonRequest(daemonRequest, streamingResponseWriter: streamingResponseWriter);
                }
            } catch let transportError as DaemonTransportError {
                if case .acceptFailed = transportError {
                    return;
                }
                self.stateLock.lock();
                let shouldShutdownAfterFailure: Bool = self.isShutdownRequested;
                self.stateLock.unlock();
                if shouldShutdownAfterFailure {
                    return;
                }
                continue;
            } catch {
                self.stateLock.lock();
                let shouldShutdownAfterFailure: Bool = self.isShutdownRequested;
                self.stateLock.unlock();
                if shouldShutdownAfterFailure {
                    return;
                }
                continue;
            }
            self.stateLock.lock();
            let shouldShutdownAfterServe: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if shouldShutdownAfterServe {
                return;
            }
        }
    }

    private func handleDaemonRequest(
        _ daemonRequest: DaemonRequest,
        streamingResponseWriter: StreamingResponseWriter
    ) throws -> Void {
        switch (daemonRequest) {
        case .handshake:
            try streamingResponseWriter.sendResponse(DaemonResponse.handshakeAccepted(
                protocolVersion: DaemonProtocol.protocolVersion,
                applicationName: DaemonProtocol.applicationName));
            try streamingResponseWriter.close();
        case .status:
            let statusReport: DaemonStatusReport = self.healthProvider();
            try streamingResponseWriter.sendResponse(DaemonResponse.status(
                workerStatus: statusReport.workerStatus,
                readyModelId: statusReport.readyModelId,
                defaultModelId: DaemonIpcService.effectiveDefaultModelId(
                    instancePaths: self.instancePaths)));
            try streamingResponseWriter.close();
        case .chatGenerate:
            guard let chatContext: DaemonIpcChatContext = self.chatContext else {
                try streamingResponseWriter.sendResponse(DaemonResponse.requestRejected(
                    reason: "this daemon verb is not wired into the Swift supervisor yet"));
                try streamingResponseWriter.close();
                return;
            }
            return try DaemonIpcChat.serve(
                daemonRequest,
                chatContext: chatContext,
                requestIdAllocator: self.requestIdAllocator,
                streamingResponseWriter: streamingResponseWriter);
        case let .embedGenerate(model, inputs, dimensions):
            guard let modelsContext: DaemonIpcModelsContext = self.modelsContext else {
                try streamingResponseWriter.sendResponse(DaemonResponse.requestRejected(
                    reason: "this daemon verb is not wired into the Swift supervisor yet"));
                try streamingResponseWriter.close();
                return;
            }
            return try DaemonIpcEmbeddings.serve(
                model: model,
                inputs: inputs,
                dimensions: dimensions,
                modelsContext: modelsContext,
                requestIdAllocator: self.requestIdAllocator,
                streamingResponseWriter: streamingResponseWriter);
        case .modelsList:
            guard let modelsContext: DaemonIpcModelsContext = self.modelsContext else {
                try streamingResponseWriter.sendResponse(DaemonResponse.requestRejected(
                    reason: "this daemon verb is not wired into the Swift supervisor yet"));
                try streamingResponseWriter.close();
                return;
            }
            return try DaemonIpcModels.handleModelsList(
                modelsContext: modelsContext,
                streamingResponseWriter: streamingResponseWriter);
        case .catalog:
            guard let modelsContext: DaemonIpcModelsContext = self.modelsContext else {
                try streamingResponseWriter.sendResponse(DaemonResponse.requestRejected(
                    reason: "this daemon verb is not wired into the Swift supervisor yet"));
                try streamingResponseWriter.close();
                return;
            }
            return try DaemonIpcModels.handleCatalog(
                modelsContext: modelsContext,
                streamingResponseWriter: streamingResponseWriter);
        case let .downloadStart(modelId):
            guard let modelsContext: DaemonIpcModelsContext = self.modelsContext else {
                try streamingResponseWriter.sendResponse(DaemonResponse.requestRejected(
                    reason: "the daemon has no Library download coordinator wired"));
                try streamingResponseWriter.close();
                return;
            }
            return try DaemonIpcModels.handleDownloadStart(
                modelId: modelId,
                modelsContext: modelsContext,
                streamingResponseWriter: streamingResponseWriter);
        case .downloadStatus:
            guard let modelsContext: DaemonIpcModelsContext = self.modelsContext else {
                // No Library coordinator is wired: no download can be active.
                try streamingResponseWriter.sendResponse(DaemonResponse.downloadStatus(job: nil));
                try streamingResponseWriter.close();
                return;
            }
            return try DaemonIpcModels.handleDownloadStatus(
                modelsContext: modelsContext,
                streamingResponseWriter: streamingResponseWriter);
        case let .defaultModelSet(modelId):
            guard let modelsContext: DaemonIpcModelsContext = self.modelsContext else {
                try streamingResponseWriter.sendResponse(DaemonResponse.requestRejected(
                    reason: "this daemon verb is not wired into the Swift supervisor yet"));
                try streamingResponseWriter.close();
                return;
            }
            return try DaemonIpcModels.handleDefaultModelSet(
                modelId: modelId,
                modelsContext: modelsContext,
                streamingResponseWriter: streamingResponseWriter);
        }
    }

    /// The model ID CLI verbs fall back to: the user-configured default
    /// model, or the built-in fallback when none is configured. The config
    /// file is read fresh so a `models default` made on another CLI process
    /// is visible here.
    private static func effectiveDefaultModelId(
        instancePaths: AstronomicalInstancePaths
    ) -> String {
        let configuredDefaultModelId: String? = (try? AstronomicalConfig.loadFromInstancePaths(
            instancePaths
        ))?.defaultModel;
        return configuredDefaultModelId ?? DefaultModel.builtinDefaultModelId;
    }
}

/// Removes the socket file after shutdown; a missing file is already the
/// desired end state.
private func removeSocketFileIgnoringMissing(socketFilePath: String) {
    try? FileManager.default.removeItem(atPath: socketFilePath);
}
