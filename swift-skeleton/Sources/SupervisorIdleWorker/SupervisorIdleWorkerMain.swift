import Foundation;

import IpcProtocol;

/**
 * The supervisor test worker entry point, migrating the process shells of
 * apps/supervisor/tests/fixtures/{idle_worker, loading_forever_worker,
 * mismatched_ready_worker, replacement_ready_worker}.rs: speak the framed
 * command and event protocol over stdin and stdout until the supervisor
 * closes the command side. Failures reach the operator through stderr and a
 * nonzero exit; a clean end of stream exits successfully.
 *
 * The first positional argument names a control directory where
 * scripted-behavior markers land; a leading `--` argument instead selects
 * one of the dedicated replacement-candidate process shapes.
 */
@main
final class SupervisorIdleWorkerMain {

    enum FixtureProcessMode: String {

        case loadingForever = "--loading-forever";
        case mismatchedReady = "--mismatched-ready";
        case replacementCandidate = "--replacement-candidate";
    }

    enum FixtureFailure: Error, CustomStringConvertible {

        case unrecognizedModeArgument(String);

        var description: String {
            switch (self) {
            case let .unrecognizedModeArgument(modeArgument):
                return "supervisor idle worker received unknown mode \(modeArgument)";
            }
        }
    }

    static func main() {
        // The framed event pipe is the worker's only output channel: a
        // supervisor closing its read side mid-frame must surface as a
        // failed write, not as a process-killing SIGPIPE.
        signal(SIGPIPE, SIG_IGN);
        do {
            try SupervisorIdleWorkerMain.runFixtureProcess(
                modeArgument: SupervisorIdleWorkerMain.modeArgument(CommandLine.arguments),
                controlDirectoryPath: SupervisorIdleWorkerMain.controlDirectoryFromArguments(
                    CommandLine.arguments));
            exit(0);
        } catch let fixtureError {
            let failureNotice: String = "supervisor idle worker fixture failed: \(fixtureError)\n"
            FileHandle.standardError.write(Data(failureNotice.utf8))
            exit(1)
        }
    }

    private static func runFixtureProcess(
        modeArgument: String?,
        controlDirectoryPath: String?
    ) throws -> Void {
        let commandReader: ProtocolReader = ProtocolReader(transport: PipeFrameTransport(
            fileDescriptor: FileHandle.standardInput.fileDescriptor,
            isWriteEnd: false));
        let eventWriter: ProtocolWriter = ProtocolWriter(transport: PipeFrameTransport(
            fileDescriptor: FileHandle.standardOutput.fileDescriptor,
            isWriteEnd: true));
        guard let modeArgument = modeArgument else {
            try IdleWorkerScenario.runFixture(
                commandReader: commandReader,
                eventWriter: eventWriter,
                controlDirectoryPath: controlDirectoryPath);
            return;
        }
        guard let processMode: FixtureProcessMode = FixtureProcessMode(rawValue: modeArgument) else {
            throw FixtureFailure.unrecognizedModeArgument(modeArgument);
        }
        switch (processMode) {
        case .loadingForever:
            // A candidate that never acknowledges anything: the supervisor
            // must time the handshake out and reap the process.
            while true {
                Thread.sleep(forTimeInterval: 3_600);
            }
        case .mismatchedReady:
            // Acknowledge readiness with a model the supervisor never asked
            // about, then end the process before any runtime-policy
            // acknowledgement can arrive.
            _ = try commandReader.nextCommand();
            try eventWriter.sendEvent(.ready(
                modelId: "astronomical/wrong-model",
                capabilities: .from(chatCapabilities: ChatModelCapabilities(
                    supportsReasoning: false,
                    supportsToolCalls: false,
                    hasVision: false,
                    maxInputTokens: 241_664,
                    maxOutputTokens: 20_480,
                    contextWindow: 262_144))));
        case .replacementCandidate:
            try IdleWorkerReplacementCandidateScenario.runFixture(
                commandReader: commandReader,
                eventWriter: eventWriter);
        }
    }

    private static func modeArgument(_ arguments: Array<String>) -> String? {
        guard let firstArgument: String = arguments.count > 1 ? arguments[1] : nil else {
            return nil
        }
        return firstArgument.hasPrefix("--") ? firstArgument : nil
    }

    private static func controlDirectoryFromArguments(
        _ arguments: Array<String>
    ) -> String? {
        guard let firstArgument: String = arguments.count > 1 ? arguments[1] : nil else {
            return nil
        }
        return firstArgument.hasPrefix("--") ? nil : firstArgument
    }
}
