import Foundation;

import IpcProtocol;

/// The worker-derived slice of the daemon status response.
public struct DaemonStatusReport: Equatable {

    /// Coarse availability of the resident local worker.
    public let workerStatus: DaemonWorkerStatus;
    /// Exact model identity the worker reported as resident, when ready.
    public let readyModelId: String?;

    public init(workerStatus: DaemonWorkerStatus, readyModelId: String?) {
        self.workerStatus = workerStatus;
        self.readyModelId = readyModelId;
    }

    /// The report of a supervisor that has no worker at all.
    public static func unavailable() -> DaemonStatusReport {
        return DaemonStatusReport(workerStatus: .unavailable, readyModelId: nil);
    }
}
