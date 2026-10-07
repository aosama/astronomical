import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// One daemon status probe: worker state, resident model, and effective
/// default model from a single Status frame.
///
/// Mirrors apps/astronomical/src/daemon_probe.rs. Each stage runs on its own
/// fresh connection so a wedged daemon cannot poison later stages.
public struct DaemonStatusSnapshot: Equatable {

    /// Whether the inference engine can serve generations right now.
    public let workerStatus: DaemonWorkerStatus;
    /// Model currently resident in the worker, if any.
    public let readyModelId: String?;
    /// Effective default model: the persisted value, else the built-in.
    public let defaultModelId: String?;

    internal init(
        workerStatus: DaemonWorkerStatus,
        readyModelId: String?,
        defaultModelId: String?
    ) {
        self.workerStatus = workerStatus;
        self.readyModelId = readyModelId;
        self.defaultModelId = defaultModelId;
    }
}

public enum DaemonProbeError: Error, Equatable {
    /// No candidate socket accepted a connection.
    case daemonNotRunning
    /// The daemon stopped responding mid-journey.
    case daemonStoppedResponding
    /// The daemon declined the request.
    case daemonRejected(reason: String)
}

public enum DaemonProbe {

    /// Worker state, resident model, and effective default model in one probe.
    public static func statusSnapshot(
        candidateSocketPaths: Array<String>
    ) throws -> DaemonStatusSnapshot {
        let statusResponse: DaemonResponse = try requestOnFreshConnection(
            candidateSocketPaths: candidateSocketPaths,
            daemonRequest: .status);
        guard case let .status(workerStatus, readyModelId, defaultModelId) = statusResponse else {
            throw DaemonProbeError.daemonStoppedResponding;
        }
        return DaemonStatusSnapshot(
            workerStatus: workerStatus,
            readyModelId: readyModelId,
            defaultModelId: defaultModelId);
    }

    /// The models discovered on this Mac, with their resident markers.
    public static func modelsList(
        candidateSocketPaths: Array<String>
    ) throws -> Array<DaemonListedModel> {
        let listResponse: DaemonResponse = try requestOnFreshConnection(
            candidateSocketPaths: candidateSocketPaths,
            daemonRequest: .modelsList);
        switch (listResponse) {
        case let .modelsList(models):
            return models;
        case let .requestRejected(reason):
            throw DaemonProbeError.daemonRejected(reason: reason);
        default:
            throw DaemonProbeError.daemonStoppedResponding;
        }
    }

    /// The release download catalog with local readiness per entry.
    public static func catalog(
        candidateSocketPaths: Array<String>
    ) throws -> Array<DaemonCatalogEntry> {
        let catalogResponse: DaemonResponse = try requestOnFreshConnection(
            candidateSocketPaths: candidateSocketPaths,
            daemonRequest: .catalog);
        switch (catalogResponse) {
        case let .catalog(entries):
            return entries;
        case let .requestRejected(reason):
            throw DaemonProbeError.daemonRejected(reason: reason);
        default:
            throw DaemonProbeError.daemonStoppedResponding;
        }
    }

    /// Starts (or resumes) a download; the daemon's refusal arrives as an error.
    public static func downloadStart(
        modelId: String,
        candidateSocketPaths: Array<String>
    ) throws -> Void {
        let startResponse: DaemonResponse = try requestOnFreshConnection(
            candidateSocketPaths: candidateSocketPaths,
            daemonRequest: .downloadStart(modelId: modelId));
        switch (startResponse) {
        case .downloadStarted:
            return;
        case let .requestRejected(reason):
            throw DaemonProbeError.daemonRejected(reason: reason);
        default:
            throw DaemonProbeError.daemonStoppedResponding;
        }
    }

    /// Persists a new default model id; the daemon's refusal arrives as an error.
    public static func defaultModelSet(
        modelId: String,
        candidateSocketPaths: Array<String>
    ) throws -> String {
        let setResponse: DaemonResponse = try requestOnFreshConnection(
            candidateSocketPaths: candidateSocketPaths,
            daemonRequest: .defaultModelSet(modelId: modelId));
        switch (setResponse) {
        case let .defaultModelSet(defaultModelId):
            return defaultModelId;
        case let .requestRejected(reason):
            throw DaemonProbeError.daemonRejected(reason: reason);
        default:
            throw DaemonProbeError.daemonStoppedResponding;
        }
    }

    /// Opens a connection the caller owns for its own request.
    public static func connect(
        candidateSocketPaths: Array<String>
    ) throws -> DaemonIpcClient {
        for candidateSocketPath: String in candidateSocketPaths {
            if let daemonClient: DaemonIpcClient = try? DaemonIpcClient.connect(
                socketPath: candidateSocketPath) {
                return daemonClient;
            }
        }
        throw DaemonProbeError.daemonNotRunning;
    }

    /// The active download job, if one runs.
    public static func downloadStatus(
        candidateSocketPaths: Array<String>
    ) throws -> DaemonDownloadJob? {
        let downloadResponse: DaemonResponse = try requestOnFreshConnection(
            candidateSocketPaths: candidateSocketPaths,
            daemonRequest: .downloadStatus);
        switch (downloadResponse) {
        case let .downloadStatus(job):
            return job;
        case let .requestRejected(reason):
            throw DaemonProbeError.daemonRejected(reason: reason);
        default:
            throw DaemonProbeError.daemonStoppedResponding;
        }
    }

    private static func requestOnFreshConnection(
        candidateSocketPaths: Array<String>,
        daemonRequest: DaemonRequest
    ) throws -> DaemonResponse {
        for candidateSocketPath: String in candidateSocketPaths {
            guard let client: DaemonIpcClient = try? DaemonIpcClient.connect(
                socketPath: candidateSocketPath) else {
                continue;
            }
            do {
                try client.sendRequest(daemonRequest);
                guard let daemonResponse: DaemonResponse = try client.nextResponse() else {
                    throw DaemonProbeError.daemonStoppedResponding;
                }
                if case let .requestRejected(reason) = daemonResponse {
                    throw DaemonProbeError.daemonRejected(reason: reason);
                }
                return daemonResponse;
            } catch let probeError as DaemonProbeError {
                throw probeError;
            } catch {
                // One unresponsive daemon socket falls through to the next
                // candidate, exactly like the Rust probe.
                continue;
            }
        }
        throw DaemonProbeError.daemonNotRunning;
    }
}
