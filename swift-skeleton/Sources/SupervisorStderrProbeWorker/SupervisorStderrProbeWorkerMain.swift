import Foundation

import IpcProtocol

/**
 * The stderr-diagnostic probe worker, migrating
 * apps/supervisor/tests/fixtures/stderr_probe_worker.rs: write a bounded
 * oversized diagnostic and one visible line to stderr, emit exactly one
 * framed Idle event on stdout, then exit cleanly — so the supervisor's
 * stream-closure diagnostics must surface the exit status and the bounded
 * stderr tail. Environments that point stderr at /dev/null are skipped the
 * same way the Rust fixture skips them.
 */
@main
final class SupervisorStderrProbeWorkerMain {

    static func main() {
        signal(SIGPIPE, SIG_IGN)
        if (SupervisorStderrProbeWorkerMain.stderrPointsToDevNull()) {
            exit(0)
        }
        let stderrPipe: FileHandle = FileHandle.standardError
        stderrPipe.write(Data((String(repeating: "x", count: 16 * 1_024) + "\n").utf8))
        stderrPipe.write(Data("stderr-probe worker observed visible stderr\n".utf8))
        do {
            try ProtocolWriter(transport: PipeFrameTransport(
                fileDescriptor: FileHandle.standardOutput.fileDescriptor,
                isWriteEnd: true)).sendEvent(.idle(
                machineMlxMemoryCeilingBytes: 40_000_000_000,
                effectiveMlxMemoryCeilingBytes: 40_000_000_000,
                minimumMlxMemoryCeilingBytes: 1))
            exit(0)
        } catch {
            let failureNotice: String = "stderr-probe worker failed: \(error)\n"
            FileHandle.standardError.write(Data(failureNotice.utf8))
            exit(1)
        }
    }

    /// Compares the stderr descriptor against /dev/null by device and inode,
    /// the same identity check the Rust fixture performs.
    private static func stderrPointsToDevNull() -> Bool {
        var stderrStatus: stat = stat()
        var devNullStatus: stat = stat()
        guard fstat(FileHandle.standardError.fileDescriptor, &stderrStatus) == 0,
            stat("/dev/null", &devNullStatus) == 0 else {
            return false
        }
        return stderrStatus.st_dev == devNullStatus.st_dev
            && stderrStatus.st_ino == devNullStatus.st_ino
    }
}
