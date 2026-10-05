// InferenceWorkerMain.swift — InferenceWorker
//
// MIGRATION MARKER — placeholder entry point; the worker startup sequence
// is not ported yet (wave 3), so main only reports that and exits.
//
// Migrates from (wave 3): apps/inference-worker main and worker_startup*
// modules — process entry, IPC (Inter-Process Communication) connection
// handshake, and startup sequencing.
//
// Carried contracts:
// - Startup speaks the unchanged IpcProtocol surface; supervisor and worker
//   swap implementations in coordinated waves, never both ad hoc.
// - One real-model worker journey at a time on the GPU (Graphics Processing
//   Unit); hermetic CPU work may parallelize.

import Foundation;

@main
final class InferenceWorkerMain {
    static func main() {
        let notPortedNotice: String = "inference-worker: the Swift worker is not ported yet\n";
        guard let noticeBytes: Data = notPortedNotice.data(using: .utf8) else {
            exit(1);
        }
        FileHandle.standardError.write(noticeBytes);
        exit(1);
    }
}
