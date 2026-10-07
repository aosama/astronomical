import Foundation

import IpcProtocol

/**
 * The supervisor test worker entry point, migrating the process shell of
 * apps/supervisor/tests/fixtures/idle_worker.rs: speak the framed command
 * and event protocol over stdin and stdout until the supervisor closes the
 * command side. Failures reach the operator through stderr and a nonzero
 * exit; a clean end of stream exits successfully.
 *
 * The first positional argument, when present, names a control directory
 * where scripted-behavior markers land (the disconnect tripwire probe).
 */
@main
final class SupervisorIdleWorkerMain {

    static func main() {
        // The framed event pipe is the worker's only output channel: a
        // supervisor closing its read side mid-frame must surface as a
        // failed write, not as a process-killing SIGPIPE.
        signal(SIGPIPE, SIG_IGN)
        let controlDirectoryPath: String? = SupervisorIdleWorkerMain.controlDirectoryFromArguments(
            CommandLine.arguments)
        do {
            try IdleWorkerScenario.runFixture(
                commandReader: ProtocolReader(transport: PipeFrameTransport(
                    fileDescriptor: FileHandle.standardInput.fileDescriptor,
                    isWriteEnd: false)),
                eventWriter: ProtocolWriter(transport: PipeFrameTransport(
                    fileDescriptor: FileHandle.standardOutput.fileDescriptor,
                    isWriteEnd: true)),
                controlDirectoryPath: controlDirectoryPath)
            exit(0)
        } catch let fixtureError {
            let failureNotice: String = "supervisor idle worker fixture failed: \(fixtureError)\n"
            FileHandle.standardError.write(Data(failureNotice.utf8))
            exit(1)
        }
    }

    private static func controlDirectoryFromArguments(
        _ arguments: Array<String>
    ) -> String? {
        guard let firstArgument: String = arguments.count > 1 ? arguments[1] : nil else {
            return nil
        }
        return firstArgument
    }
}
