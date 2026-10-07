import Foundation

import AstronomicalConfig;
import IpcProtocol;

/**
 * The `astronomical status` verb: worker state, resident model, effective
 * default model, and the active download job, if any. Porting
 * status_command.rs; the report shape is the carried contract.
 */
public enum StatusCommand {

    /// Runs the status journey against the resident daemon and returns the
    /// report for the caller to print.
    public static func run(
        instancePaths: AstronomicalInstancePaths
    ) throws -> String {
        let candidateSocketPaths: Array<String> = [
            instancePaths.ipcSocketFilePath.string,
        ];
        return try StatusCommand.run(candidateSocketPaths: candidateSocketPaths);
    }

    /// Runs the status journey over the given instance sockets.
    public static func run(
        candidateSocketPaths: Array<String>
    ) throws -> String {
        let statusSnapshot: DaemonStatusSnapshot;
        do {
            statusSnapshot = try DaemonProbe.statusSnapshot(
                candidateSocketPaths: candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            switch (probeError) {
            case .daemonNotRunning:
                throw StatusError.daemonNotRunning;
            default:
                throw StatusError.daemonStoppedResponding;
            }
        }
        let activeDownloadJob: DaemonDownloadJob?;
        do {
            activeDownloadJob = try DaemonProbe.downloadStatus(
                candidateSocketPaths: candidateSocketPaths);
        } catch let probeError as DaemonProbeError {
            switch (probeError) {
            case .daemonNotRunning:
                throw StatusError.daemonNotRunning;
            default:
                throw StatusError.daemonStoppedResponding;
            }
        }
        var report: String = "";
        report += "worker:   \(StatusCommand.renderWorkerStatus(statusSnapshot))\n";
        report += "default:  \(statusSnapshot.defaultModelId ?? "none")\n";
        report += "download: \(StatusCommand.renderDownloadLine(activeDownloadJob))\n";
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

    private static func renderDownloadLine(_ activeDownloadJob: DaemonDownloadJob?) -> String {
        guard let activeDownloadJob: DaemonDownloadJob = activeDownloadJob else {
            return "none";
        }
        if let downloadError: String = activeDownloadJob.error {
            return "\(activeDownloadJob.huggingfaceId) (failed: \(downloadError))";
        }
        if (activeDownloadJob.bytesTotal == 0) {
            return "\(activeDownloadJob.huggingfaceId) (\(activeDownloadJob.state))";
        }
        return "\(activeDownloadJob.huggingfaceId) — \(activeDownloadJob.state) "
            + "\(CliFormatting.formatGigabytes(activeDownloadJob.bytesCompleted)) GB / "
            + "\(CliFormatting.formatGigabytes(activeDownloadJob.bytesTotal)) GB";
    }
}
