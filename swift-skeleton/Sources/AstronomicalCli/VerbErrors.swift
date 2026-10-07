import Foundation

import IpcProtocol;

/**
 * One-line user-facing failures for the one-shot verbs, porting errors.rs.
 * Recovery actions stay in the message so the CLI never dumps config recipes
 * as the happy path. Exit-code mapping lives in the main dispatch:
 * usage failures and `modelUnavailable` exit 2, the rest exit 1.
 */
public enum LaunchError: Error, Equatable, CustomStringConvertible {

    case astronomicalUnavailable
    case modelListUnavailable
    case noChatModels
    case openCodeMissing
    case unknownTool(requestedTool: String)
    case modelPickerRequired
    case requestedModelMissing(requestedModelId: String)
    case invalidModelSelection
    case openCodeConfigFailed
    case toolStartFailed(program: String, cause: String)

    public var description: String {
        switch (self) {
        case .astronomicalUnavailable:
            return "Start Astronomical first."
        case .modelListUnavailable:
            return "Astronomical is running but did not return a model list."
        case .noChatModels:
            return "No models in the Library yet. Open Astronomical and download one."
        case .openCodeMissing:
            return "Install OpenCode: curl -fsSL https://opencode.ai/install | bash"
        case let .unknownTool(requestedTool):
            return "Unknown tool \(requestedTool). Try: astronomical launch opencode"
        case .modelPickerRequired:
            return "Choose a model with --model; several chat models are in the Library."
        case let .requestedModelMissing(requestedModelId):
            return "Model \(requestedModelId) is not a chat model in the Library."
        case .invalidModelSelection:
            return "Select a model by number or id."
        case .openCodeConfigFailed:
            return "could not prepare OpenCode config."
        case let .toolStartFailed(program, cause):
            return "failed to start \"\(program)\": \(cause)"
        }
    }
}

/// Failures of the one-shot `respond` journey after arguments have parsed.
public enum RespondError: Error, CustomStringConvertible {

    case daemonNotRunning
    case workerNotReady
    case modelUnavailable(reason: String)
    case downloadFailed(reason: String)
    case generationRejected(reason: String)
    case generationFailed(reason: String)
    case daemonStoppedResponding
    case stdoutUnwritable(cause: String)
    case imageReadFailed(path: String, cause: String)
    case unsupportedImage(path: String, supported: String)
    case imageTooLarge(actualBytes: Int, maximumBytes: Int)
    case schemaReadFailed(path: String, cause: String)
    case schemaNotUtf8(path: String, cause: String)
    case schemaTooLarge(actualBytes: Int, maximumBytes: Int)

    public var description: String {
        switch (self) {
        case .daemonNotRunning:
            return "Astronomical isn't running — start it, then retry."
        case .workerNotReady:
            return "The Astronomical worker is not ready yet — retry in a moment."
        case let .modelUnavailable(reason):
            return reason
        case let .downloadFailed(reason):
            return "the model download failed: \(reason)"
        case let .generationRejected(reason):
            return "The daemon declined the request: \(reason)"
        case let .generationFailed(reason):
            return "The model failed to finish the response: \(reason)"
        case .daemonStoppedResponding:
            return "The daemon stopped responding."
        case let .stdoutUnwritable(cause):
            return "Could not write the answer to standard output: \(cause)"
        case let .imageReadFailed(path, cause):
            return "could not read image \(path): \(cause)"
        case let .unsupportedImage(path, supported):
            return "\(path) is not a supported image; supported formats: \(supported)"
        case let .imageTooLarge(actualBytes, maximumBytes):
            return "the combined image size is \(actualBytes) bytes, over the "
                + "\(maximumBytes)-byte limit; send smaller or fewer --image files"
        case let .schemaReadFailed(path, cause):
            return "could not read schema \(path): \(cause)"
        case let .schemaNotUtf8(path, cause):
            return "\(path) is not valid UTF-8 text: \(cause)"
        case let .schemaTooLarge(actualBytes, maximumBytes):
            return "the schema file is \(actualBytes) bytes, over the \(maximumBytes)-byte "
                + "limit; send a smaller --schema file"
        }
    }

    static func from(_ lifecycleError: ModelLifecycleError) -> RespondError {
        switch (lifecycleError) {
        case .daemonNotRunning:
            return .daemonNotRunning
        case .workerNotReady:
            return .workerNotReady
        case .daemonStoppedResponding:
            return .daemonStoppedResponding
        case let .modelUnavailable(reason):
            return .modelUnavailable(reason: reason)
        case let .downloadFailed(reason):
            return .downloadFailed(reason: reason)
        }
    }
}

/// Failures of the one-shot `embed` journey after arguments have parsed.
public enum EmbedError: Error, CustomStringConvertible {

    case daemonNotRunning
    case workerNotReady
    case modelUnavailable(reason: String)
    case downloadFailed(reason: String)
    case embeddingsRejected(reason: String)
    case embeddingsFailed(reason: EmbeddingsFailureReason)
    case embedInputRequired
    case inputFileUnreadable(filePath: String, cause: String)
    case stdinUnreadable(cause: String)
    case daemonStoppedResponding
    case stdoutUnwritable(cause: String)

    public var description: String {
        switch (self) {
        case .daemonNotRunning:
            return "Astronomical isn't running — start it, then retry."
        case .workerNotReady:
            return "The Astronomical worker is not ready yet — retry in a moment."
        case let .modelUnavailable(reason):
            return reason
        case let .downloadFailed(reason):
            return "the model download failed: \(reason)"
        case let .embeddingsRejected(reason):
            return "The daemon declined the request: \(reason)"
        case let .embeddingsFailed(reason):
            return EmbedError.embeddingsFailureReasonText(reason)
        case .embedInputRequired:
            return "embed needs input text. Try: astronomical embed 'Hello'"
        case let .inputFileUnreadable(filePath, cause):
            return "Could not read \(filePath): \(cause)"
        case let .stdinUnreadable(cause):
            return "Could not read standard input: \(cause)"
        case .daemonStoppedResponding:
            return "The daemon stopped responding."
        case let .stdoutUnwritable(cause):
            return "Could not write the vector document to standard output: \(cause)"
        }
    }

    /// Human phrasing for the typed embeddings failure.
    static func embeddingsFailureReasonText(_ reason: EmbeddingsFailureReason) -> String {
        switch (reason) {
        case let .invalidRequest(reason):
            return "the model rejected the input: \(reason)"
        case let .fatalExecution(reason):
            return "the model failed to finish the embedding: \(reason)"
        case let .contextLengthExceeded(actualTotalContextTokens, maximumContextTokens):
            return "the input needs \(actualTotalContextTokens) context tokens but the model's "
                + "context window is \(maximumContextTokens) tokens"
        case .engineBusy:
            return "the embedding engine is busy with another request"
        case .malformedModelOutput:
            return "the model produced vectors that could not be pooled or normalized"
        }
    }

    static func from(_ lifecycleError: ModelLifecycleError) -> EmbedError {
        switch (lifecycleError) {
        case .daemonNotRunning:
            return .daemonNotRunning
        case .workerNotReady:
            return .workerNotReady
        case .daemonStoppedResponding:
            return .daemonStoppedResponding
        case let .modelUnavailable(reason):
            return .modelUnavailable(reason: reason)
        case let .downloadFailed(reason):
            return .downloadFailed(reason: reason)
        }
    }
}

/// Failures of the `models` verb after arguments have parsed.
public enum ModelsVerbError: Error, CustomStringConvertible {

    case daemonNotRunning
    case daemonStoppedResponding
    case workerNotReady
    case modelUnavailable(reason: String)
    case downloadFailed(reason: String)
    case daemonRejected(reason: String)
    case stdoutUnwritable(cause: String)

    public var description: String {
        switch (self) {
        case .daemonNotRunning:
            return "Astronomical isn't running — start it, then retry."
        case .daemonStoppedResponding:
            return "The daemon stopped responding."
        case .workerNotReady:
            return "The Astronomical worker is not ready yet — retry in a moment."
        case let .modelUnavailable(reason):
            return reason
        case let .downloadFailed(reason):
            return "the model download failed: \(reason)"
        case let .daemonRejected(reason):
            return "The daemon declined the request: \(reason)"
        case let .stdoutUnwritable(cause):
            return "Could not write the model report to standard output: \(cause)"
        }
    }

    static func from(_ lifecycleError: ModelLifecycleError) -> ModelsVerbError {
        switch (lifecycleError) {
        case .daemonNotRunning:
            return .daemonNotRunning
        case .workerNotReady:
            return .workerNotReady
        case .daemonStoppedResponding:
            return .daemonStoppedResponding
        case let .modelUnavailable(reason):
            return .modelUnavailable(reason: reason)
        case let .downloadFailed(reason):
            return .downloadFailed(reason: reason)
        }
    }

    static func from(_ probeError: DaemonProbeError) -> ModelsVerbError {
        switch (probeError) {
        case .daemonNotRunning:
            return .daemonNotRunning
        case .daemonStoppedResponding:
            return .daemonStoppedResponding
        case let .daemonRejected(reason):
            return .daemonRejected(reason: reason)
        }
    }
}

/// Failures of the `status` verb after arguments have parsed.
public enum StatusError: Error, CustomStringConvertible {

    case daemonNotRunning
    case daemonStoppedResponding
    case stdoutUnwritable(cause: String)

    public var description: String {
        switch (self) {
        case .daemonNotRunning:
            return "Astronomical isn't running — start it, then retry."
        case .daemonStoppedResponding:
            return "The daemon stopped responding."
        case let .stdoutUnwritable(cause):
            return "Could not write the status report to standard output: \(cause)"
        }
    }
}
