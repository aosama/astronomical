import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// One running daemon IPC service bound to the instance socket.
///
/// Mirrors apps/supervisor/src/daemon_ipc.rs: ephemeral CLI verbs talk over
/// the owner-only unix socket, one request is served per connection, and the
/// service shares no transport with the REST surface. Generation, library,
/// and embeddings handlers land with their supervisor slices; this shell
/// serves the handshake and status verbs and rejects the rest with the same
/// wire frame the Rust service uses for rejected requests.
public final class DaemonIpcService {

    private let listener: DaemonIpcListener;
    private let socketFilePath: String;
    private let stateLock: NSLock;
    private let healthProvider: () -> DaemonStatusReport;
    private var isShutdownRequested: Bool;
    private var serviceThread: Thread?;

    private init(
        listener: DaemonIpcListener,
        socketFilePath: String,
        healthProvider: @escaping () -> DaemonStatusReport
    ) {
        self.listener = listener;
        self.socketFilePath = socketFilePath;
        self.healthProvider = healthProvider;
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
        healthProvider: @escaping () -> DaemonStatusReport
    ) throws -> DaemonIpcService {
        let socketFilePath: String = instancePaths.ipcSocketFilePath.string;
        let listener: DaemonIpcListener = try DaemonIpcListener.bind(
            socketPath: socketFilePath);
        let service: DaemonIpcService = DaemonIpcService(
            listener: listener,
            socketFilePath: socketFilePath,
            healthProvider: healthProvider);
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
            // One misbehaving local client must not take the endpoint down.
            try? self.listener.serveNextRequest { (daemonRequest: DaemonRequest) -> DaemonResponse in
                return self.handleDaemonRequest(daemonRequest);
            }
            self.stateLock.lock();
            let shouldShutdownAfterServe: Bool = self.isShutdownRequested;
            self.stateLock.unlock();
            if shouldShutdownAfterServe {
                return;
            }
        }
    }

    private func handleDaemonRequest(_ daemonRequest: DaemonRequest) -> DaemonResponse {
        switch (daemonRequest) {
        case .handshake:
            return DaemonResponse.handshakeAccepted(
                protocolVersion: DaemonProtocol.protocolVersion,
                applicationName: DaemonProtocol.applicationName);
        case .status:
            let statusReport: DaemonStatusReport = self.healthProvider();
            return DaemonResponse.status(
                workerStatus: statusReport.workerStatus,
                readyModelId: statusReport.readyModelId,
                defaultModelId: nil);
        case .chatGenerate, .embedGenerate, .modelsList, .catalog, .downloadStart,
             .downloadStatus, .defaultModelSet:
            return DaemonResponse.requestRejected(
                reason: "this daemon verb is not wired into the Swift supervisor yet");
        }
    }
}

/// Removes the socket file after shutdown; a missing file is already the
/// desired end state.
private func removeSocketFileIgnoringMissing(socketFilePath: String) {
    try? FileManager.default.removeItem(atPath: socketFilePath);
}
