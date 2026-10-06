import Foundation;

import IpcProtocol;

/// The inference-worker process entry point.
///
/// Mirrors apps/inference-worker/src/main.rs: serve the framed command and
/// event protocol over stdin and stdout until the supervisor closes the
/// command side. A startup or serving failure reaches the operator through
/// stderr and a nonzero exit; a clean end of stream exits successfully.
@main
final class InferenceWorkerMain {

    static func main() {
        // The framed event pipe is the worker's only output channel: a
        // supervisor that closes its read side mid-frame must surface as a
        // failed write, not as a process-killing SIGPIPE, exactly as the Rust
        // worker's ignored SIGPIPE default behaves.
        signal(SIGPIPE, SIG_IGN);
        do {
            try WorkerCommandLoop.runBootstrappedWorker(
                readTransport: PipeFrameTransport(
                    fileDescriptor: FileHandle.standardInput.fileDescriptor,
                    isWriteEnd: false),
                writeTransport: PipeFrameTransport(
                    fileDescriptor: FileHandle.standardOutput.fileDescriptor,
                    isWriteEnd: true));
            exit(0);
        } catch {
            let failureNotice: String = "inference-worker: \(error)\n";
            FileHandle.standardError.write(Data(failureNotice.utf8));
            exit(1);
        }
    }
}
