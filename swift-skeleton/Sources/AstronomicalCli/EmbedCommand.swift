import Foundation

import AstronomicalConfig;
import IpcProtocol;

/**
 * Collaborators the embed journey needs, injected so tests can stub them.
 */
public struct EmbedDependencies {

    /// Instance sockets to try, most preferred first.
    public let candidateSocketPaths: Array<String>;
    /// Text supplied when neither TEXT nor --file was given (stdin).
    public let standardInputText: String;
    /// Where the JSON vector document goes.
    public let stdout: TextOutputWriter;
    /// Where download progress goes.
    public let stderr: TextOutputWriter;
    /// Bound for the whole download-wait stage, if the model must download.
    public let downloadStageBoundSeconds: Double;
    /// Wait between download status polls.
    public let downloadPollIntervalSeconds: Double;

    public init(
        candidateSocketPaths: Array<String>,
        standardInputText: String,
        stdout: TextOutputWriter,
        stderr: TextOutputWriter,
        downloadStageBoundSeconds: Double = ModelLifecycle.downloadWaitStageBoundSeconds,
        downloadPollIntervalSeconds: Double = ModelLifecycle.downloadPollIntervalSeconds
    ) {
        self.candidateSocketPaths = candidateSocketPaths;
        self.standardInputText = standardInputText;
        self.stdout = stdout;
        self.stderr = stderr;
        self.downloadStageBoundSeconds = downloadStageBoundSeconds;
        self.downloadPollIntervalSeconds = downloadPollIntervalSeconds;
    }
}

/**
 * The one-shot `astronomical embed` journey, porting embed.rs: text, file,
 * or stdin in, one JSON vector document on stdout, process exits.
 */
public enum EmbedCommand {

    /// Runs the whole embed journey against the resident daemon: resolve the
    /// model (flag, daemon default, or built-in), let the lifecycle download
    /// it when the Mac lacks it, then submit the embeddings batch.
    public static func run(
        embedArguments: EmbedArguments,
        embedDependencies: EmbedDependencies
    ) throws -> Void {
        let inputText: String = try EmbedCommand.resolveInputText(
            embedArguments: embedArguments,
            standardInputText: embedDependencies.standardInputText
        );
        let modelLifecycle: ModelLifecycle = ModelLifecycle(
            candidateSocketPaths: embedDependencies.candidateSocketPaths,
            downloadStageBoundSeconds: embedDependencies.downloadStageBoundSeconds,
            downloadPollIntervalSeconds: embedDependencies.downloadPollIntervalSeconds
        );
        var progressStarted: Bool = false;
        let embedModelId: String;
        do {
            embedModelId = try modelLifecycle.prepareModelId(
                requestedModelId: embedArguments.modelId,
                requiredCapability: .embeddings,
                progress: { (progressLine: String) -> Void in
                    progressStarted = true;
                    _ = embedDependencies.stderr.write("\r\(progressLine)");
                }
            );
        } catch let lifecycleError as ModelLifecycleError {
            throw EmbedError.from(lifecycleError);
        }
        if (progressStarted) {
            // Finalize the live progress line before the JSON document owns stdout.
            _ = embedDependencies.stderr.write("\n");
        }
        try EmbedCommand.embedInputText(
            inputText: inputText,
            embedModelId: embedModelId,
            modelLifecycle: modelLifecycle,
            embedDependencies: embedDependencies
        );
    }

    /// Text > file contents > stdin read to EOF. An empty resolved input is a
    /// usage failure, not an empty embedding.
    static func resolveInputText(
        embedArguments: EmbedArguments,
        standardInputText: String
    ) throws -> String {
        let resolvedInput: String;
        if let text: String = embedArguments.text {
            resolvedInput = text;
        } else if let filePath: String = embedArguments.filePath {
            guard let fileContents: String = try? String(
                contentsOfFile: filePath,
                encoding: .utf8
            ) else {
                throw EmbedError.inputFileUnreadable(
                    filePath: filePath,
                    cause: "the file could not be read as UTF-8 text"
                );
            }
            resolvedInput = fileContents;
        } else {
            resolvedInput = standardInputText;
        }
        if resolvedInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw EmbedError.embedInputRequired;
        }
        return resolvedInput;
    }

    /// Submits one embeddings batch and writes the single terminal frame as
    /// one JSON document on stdout.
    private static func embedInputText(
        inputText: String,
        embedModelId: String,
        modelLifecycle: ModelLifecycle,
        embedDependencies: EmbedDependencies
    ) throws -> Void {
        let daemonClient: DaemonIpcClient;
        do {
            daemonClient = try DaemonProbe.connect(
                candidateSocketPaths: modelLifecycle.candidateSocketPaths);
        } catch DaemonProbeError.daemonNotRunning {
            throw EmbedError.daemonNotRunning;
        }
        let embedGenerateRequest: DaemonRequest = .embedGenerate(
            model: embedModelId,
            inputs: [inputText],
            dimensions: nil);
        do {
            try daemonClient.sendRequest(embedGenerateRequest);
        } catch {
            throw EmbedError.daemonStoppedResponding;
        }
        let embeddingsResponse: DaemonResponse;
        do {
            guard let nextResponse: DaemonResponse = try daemonClient.nextResponse() else {
                throw EmbedError.daemonStoppedResponding;
            }
            embeddingsResponse = nextResponse;
        } catch let embedError as EmbedError {
            throw embedError;
        } catch {
            throw EmbedError.daemonStoppedResponding;
        }
        switch (embeddingsResponse) {
        case let .embeddingsCompleted(model, vectors, inputTokenCounts):
            let embeddingVector: Array<Float> = vectors.first ?? [];
            let inputTokens: UInt32 = inputTokenCounts.first ?? 0;
            try EmbedCommand.writeVectorDocument(
                embedDependencies.stdout,
                modelId: model,
                embedding: embeddingVector,
                inputTokens: inputTokens
            );
        case let .embeddingsFailed(reason):
            throw EmbedError.embeddingsFailed(reason: reason);
        case let .generationRejected(reason):
            throw EmbedError.embeddingsRejected(reason: reason);
        default:
            // Handshake, status, and chat frames cannot answer an embeddings
            // request; treat the daemon as gone rather than guessing.
            throw EmbedError.daemonStoppedResponding;
        }
    }

    /// Writes `{"embedding":[...],"input_tokens":N,"model":"..."}` — one
    /// compact JSON document with alphabetically sorted keys, mirroring the
    /// Rust serde_json (BTreeMap) serialization — followed by one newline.
    static func writeVectorDocument(
        _ stdout: TextOutputWriter,
        modelId: String,
        embedding: Array<Float>,
        inputTokens: UInt32
    ) throws -> Void {
        let vectorDocument: Dictionary<String, Any> = [
            "model": modelId,
            "embedding": embedding,
            "input_tokens": Int(inputTokens),
        ];
        let serializedDocument: Data;
        do {
            serializedDocument = try JSONSerialization.data(
                withJSONObject: vectorDocument,
                options: [.sortedKeys, .withoutEscapingSlashes]
            );
        } catch {
            throw EmbedError.stdoutUnwritable(
                cause: "the vector document could not be serialized");
        }
        var documentText: String = String(decoding: serializedDocument, as: UTF8.self);
        documentText += "\n";
        if !stdout.write(documentText) {
            throw EmbedError.stdoutUnwritable(cause: "standard output is closed");
        }
    }
}
