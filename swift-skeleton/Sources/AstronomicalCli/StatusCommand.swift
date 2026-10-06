import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import Supervisor;

/// The `astronomical status` verb: worker state, resident model, and the
/// effective default model, reported against the resident daemon.
///
/// Mirrors apps/astronomical/src/status_command.rs. The Library download
/// section reports the active job when the daemon serves the verb, and says
/// so explicitly when the daemon has not wired that verb yet.
public enum StatusCommand {

    /// Runs the status journey against the resident daemon and returns the
    /// report for the caller to print.
    public static func run(
        instancePaths: AstronomicalInstancePaths
    ) throws -> String {
        let candidateSocketPaths: Array<String> = [
            instancePaths.ipcSocketFilePath.string,
        ];
        let statusSnapshot: DaemonStatusSnapshot;
        do {
            statusSnapshot = try DaemonProbe.statusSnapshot(
                candidateSocketPaths: candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            switch (probeError) {
            case .daemonNotRunning:
                return "astronomical: the daemon is not running for this instance";
            case .daemonStoppedResponding:
                return "astronomical: the daemon stopped responding";
            case let .daemonRejected(reason):
                return "astronomical: the daemon declined the request: \(reason)";
            }
        }
        var report: String = String();
        report += "worker:   \(renderWorkerStatus(statusSnapshot))\n";
        if let readyModelId: String = statusSnapshot.readyModelId {
            report += "model:    \(readyModelId)\n";
        }
        if let defaultModelId: String = statusSnapshot.defaultModelId {
            report += "default:  \(defaultModelId)\n";
        }
        report += renderDownloadLine(candidateSocketPaths: candidateSocketPaths);
        return report;
    }

    private static func renderWorkerStatus(_ statusSnapshot: DaemonStatusSnapshot) -> String {
        let workerState: String;
        switch (statusSnapshot.workerStatus) {
        case .ready: workerState = "ready";
        case .loading: workerState = "loading";
        case .unavailable: workerState = "unavailable";
        }
        if let readyModelId: String = statusSnapshot.readyModelId {
            return "\(workerState) (resident: \(readyModelId))";
        }
        return workerState;
    }

    private static func renderDownloadLine(candidateSocketPaths: Array<String>) -> String {
        let activeDownloadJob: DaemonDownloadJob?;
        do {
            activeDownloadJob = try DaemonProbe.activeDownloadJob(
                candidateSocketPaths: candidateSocketPaths);
        } catch DaemonProbeError.daemonRejected {
            // The daemon has not wired the Library verbs yet; say so instead
            // of pretending no download is active.
            return "download:  unavailable (daemon has not wired Library verbs yet)\n";
        } catch {
            return "download:  none\n";
        }
        if activeDownloadJob == nil {
            return "download:  none\n";
        }
        return "download:  active\n";
    }
}
