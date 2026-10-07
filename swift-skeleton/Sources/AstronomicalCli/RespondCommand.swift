import Foundation

import AstronomicalConfig;
import IpcProtocol;

/**
 * Collaborators the respond journey needs, injected so tests can stub them.
 * The output sinks collect text; production wiring passes the process
 * standard streams.
 */
public struct RespondDependencies {

    /// Instance sockets to try, most preferred first.
    public let candidateSocketPaths: Array<String>;
    /// Where the answer payload goes.
    public let stdout: TextOutputWriter;
    /// Where progress, reasoning, and errors go.
    public let stderr: TextOutputWriter;
    /// Bound for the whole download-wait stage, if the model must download.
    public let downloadStageBoundSeconds: Double;
    /// Wait between download status polls.
    public let downloadPollIntervalSeconds: Double;

    public init(
        candidateSocketPaths: Array<String>,
        stdout: TextOutputWriter,
        stderr: TextOutputWriter,
        downloadStageBoundSeconds: Double = ModelLifecycle.downloadWaitStageBoundSeconds,
        downloadPollIntervalSeconds: Double = ModelLifecycle.downloadPollIntervalSeconds
    ) {
        self.candidateSocketPaths = candidateSocketPaths;
        self.stdout = stdout;
        self.stderr = stderr;
        self.downloadStageBoundSeconds = downloadStageBoundSeconds;
        self.downloadPollIntervalSeconds = downloadPollIntervalSeconds;
    }
}

/// One write-and-forget text sink; a closed sink reports failures instead of
/// crashing the verb.
public protocol TextOutputWriter: AnyObject {

    /// Writes text and flushes; returns false when the sink is unwritable.
    @discardableResult
    func write(_ text: String) -> Bool
}

/// A writer over a plain in-memory buffer, for journeys and capture.
public final class BufferedTextOutputWriter: TextOutputWriter {

    private let stateLock: NSLock = NSLock();
    private var collectedText: String = "";

    public init() {
    }

    public func write(_ text: String) -> Bool {
        self.stateLock.lock();
        self.collectedText += text;
        self.stateLock.unlock();
        return true;
    }

    /// Drops everything written so far, like clearing a capture buffer.
    public func clear() {
        self.stateLock.lock();
        self.collectedText = "";
        self.stateLock.unlock();
    }

    /// Everything written so far.
    public var text: String {
        self.stateLock.lock();
        let currentText: String = self.collectedText;
        self.stateLock.unlock();
        return currentText;
    }
}

/// A writer over the process standard output or error stream.
public final class StandardStreamTextOutputWriter: TextOutputWriter {

    private let fileHandle: FileHandle;

    public init(fileHandle: FileHandle) {
        self.fileHandle = fileHandle;
    }

    public func write(_ text: String) -> Bool {
        self.fileHandle.write(Data(text.utf8));
        return true;
    }
}

/**
 * The one-shot `astronomical respond` journey, porting respond.rs: prompt
 * in, streamed answer out. The CLI never touches the REST surface; it speaks
 * the framed daemon IPC protocol over the instance's unix socket.
 */
public enum RespondCommand {

    /// Runs the whole respond journey against the resident daemon: resolve
    /// the model (flag, daemon default, or built-in), let the lifecycle
    /// download it when the Mac does not have it yet, then stream the answer.
    public static func run(
        respondArguments: RespondArguments,
        respondDependencies: RespondDependencies
    ) throws -> Void {
        let modelLifecycle: ModelLifecycle = ModelLifecycle(
            candidateSocketPaths: respondDependencies.candidateSocketPaths,
            downloadStageBoundSeconds: respondDependencies.downloadStageBoundSeconds,
            downloadPollIntervalSeconds: respondDependencies.downloadPollIntervalSeconds
        );
        // Local inputs fail before any daemon work: a missing schema file or
        // an unreadable image must never start a model download.
        let schemaJson: String? = try respondArguments.schemaPath.map { (schemaPath: String) -> String in
            return try RespondInputs.readSchemaInput(schemaPath: schemaPath);
        };
        let images: Array<ChatImageInput> = try RespondInputs.readImageInputs(
            respondArguments.imagePaths
        );
        var progressStarted: Bool = false;
        let chatModelId: String;
        do {
            chatModelId = try modelLifecycle.prepareModelId(
                requestedModelId: respondArguments.modelId,
                requiredCapability: .chat,
                progress: { (progressLine: String) -> Void in
                    progressStarted = true;
                    _ = respondDependencies.stderr.write("\r\(progressLine)");
                }
            );
        } catch let lifecycleError as ModelLifecycleError {
            throw RespondError.from(lifecycleError);
        }
        if (progressStarted) {
            // Finalize the live progress line before the answer owns stderr.
            _ = respondDependencies.stderr.write("\n");
        }
        try RespondCommand.streamAnswer(
            respondArguments: respondArguments,
            modelLifecycle: modelLifecycle,
            respondDependencies: respondDependencies,
            chatModelId: chatModelId,
            schemaJson: schemaJson,
            images: images
        );
    }

    /// Streams one chat generation, rendering frames until a terminal frame.
    private static func streamAnswer(
        respondArguments: RespondArguments,
        modelLifecycle: ModelLifecycle,
        respondDependencies: RespondDependencies,
        chatModelId: String,
        schemaJson: String?,
        images: Array<ChatImageInput>
    ) throws -> Void {
        let daemonClient: DaemonIpcClient;
        do {
            daemonClient = try DaemonProbe.connect(
                candidateSocketPaths: modelLifecycle.candidateSocketPaths);
        } catch DaemonProbeError.daemonNotRunning {
            throw RespondError.daemonNotRunning;
        }
        // The instructions, when given, become the initial system message the
        // worker renders before the user's prompt.
        var messages: Array<ChatMessage> = [];
        if let instructions: String = respondArguments.instructions {
            messages.append(.system(content: instructions));
        }
        messages.append(.user(content: respondArguments.prompt, images: images));
        let chatGenerateRequest: DaemonRequest = .chatGenerate(
            model: chatModelId,
            messages: messages,
            // Zero is the sentinel for "no CLI opinion": the daemon fills the
            // policy default, then the worker-advertised capability limit.
            settings: ChatGenerationSettings(
                maxOutputTokens: 0,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: respondArguments.thinkingBudget),
            schemaJson: schemaJson);
        do {
            try daemonClient.sendRequest(chatGenerateRequest);
        } catch {
            throw RespondError.daemonStoppedResponding;
        }
        var bufferedAnswer: String = "";
        try RespondCommand.relayGenerationFrames(
            respondArguments: respondArguments,
            respondDependencies: respondDependencies,
            daemonClient: daemonClient,
            bufferedAnswer: &bufferedAnswer
        );
    }

    /// Relays frames until a terminal frame; writes the answer per stream mode.
    private static func relayGenerationFrames(
        respondArguments: RespondArguments,
        respondDependencies: RespondDependencies,
        daemonClient: DaemonIpcClient,
        bufferedAnswer: inout String
    ) throws -> Void {
        while (true) {
            let nextDaemonResponse: DaemonResponse?;
            do {
                nextDaemonResponse = try daemonClient.nextResponse();
            } catch {
                throw RespondError.daemonStoppedResponding;
            }
            guard let daemonResponse: DaemonResponse = nextDaemonResponse else {
                // EOF before a terminal frame: the daemon went away mid-answer.
                throw RespondError.daemonStoppedResponding;
            }
            switch (daemonResponse) {
            case let .chatGenerationText(text):
                if (respondArguments.noStream) {
                    bufferedAnswer += text;
                } else if !respondDependencies.stdout.write(text) {
                    throw RespondError.stdoutUnwritable(cause: "standard output is closed");
                }
            case let .chatGenerationReasoning(text):
                // Reasoning is progress on stderr, never the payload; a
                // closed stderr must not abort the answer stream.
                _ = respondDependencies.stderr.write(text);
            case let .chatGenerationToolCall(_, functionName, argumentsJson):
                // The daemon never leases tools on this surface, so a tool
                // call is progress, never payload; surface it on stderr.
                _ = respondDependencies.stderr.write("[tool call \(functionName) \(argumentsJson)]\n");
            case .chatGenerationCompleted:
                if (respondArguments.noStream) {
                    if !respondDependencies.stdout.write(bufferedAnswer) {
                        throw RespondError.stdoutUnwritable(cause: "standard output is closed");
                    }
                }
                return;
            case let .chatGenerationFailed(reason):
                throw RespondError.generationFailed(
                    reason: RespondCommand.chatGenerationFailureReasonText(reason));
            case let .generationRejected(reason):
                throw RespondError.generationRejected(reason: reason);
            default:
                // Handshake, status, and embeddings frames are protocol
                // violations in the middle of a generation stream.
                throw RespondError.daemonStoppedResponding;
            }
        }
    }

    /// Renders the worker's bounded failure reason for the CLI user.
    static func chatGenerationFailureReasonText(_ reason: ChatGenerationFailureReason) -> String {
        switch (reason) {
        case let .invalidRequest(reason):
            return reason
        case let .fatalExecution(reason):
            return reason
        case let .contextLengthExceeded(actualTotalContextTokens, maximumContextTokens):
            return "the prompt and requested output exceed the model context "
                + "(\(actualTotalContextTokens) of \(maximumContextTokens) tokens)";
        case .engineBusy:
            return "the engine is busy with another generation";
        case .malformedModelOutput:
            return "the model output could not be decoded";
        }
    }
}
