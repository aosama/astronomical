import Foundation

import AstronomicalCli;
import IpcProtocol;

@testable import AstronomicalCli;

/// Shared helpers for the CLI verb journeys, porting test_support.rs.
enum CliJourneySupport {

    /// Bound for every protocol stage in these tests: generous for a stub
    /// daemon on loopback, short enough that a hung exchange fails the test
    /// well inside the repo's 120-second ceiling.
    static let testDownloadPollIntervalSeconds: Double = 0.02;

    /// A unique temporary directory per test. Names stay short because unix
    /// socket paths must fit in sockaddr_un.
    static func freshTestDirectory(_ journeyName: String) -> String {
        let stateDirectory: String = NSTemporaryDirectory()
            + "acli-\(journeyName)-\(UUID().uuidString.prefix(6))";
        try? FileManager.default.createDirectory(
            atPath: stateDirectory,
            withIntermediateDirectories: true
        );
        return stateDirectory;
    }

    /// Parses arguments as the CLI would see them after the binary name.
    static func parse(
        _ arguments: Array<String>
    ) -> Result<CliCommand, UsageError> {
        return CliArgumentParser.parseCommand(arguments);
    }

    /// Stub with one resident chat model named as the effective default: the
    /// no-flag resolution path.
    static func residentDefaultStubConfig(chatFragments: Array<String>) -> StubDaemon.StubDaemonConfig {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.installedModels = [.chat("test/local-chatter", isResident: true)];
        stubConfig.defaultModelId = "test/local-chatter";
        stubConfig.chatFragments = chatFragments;
        return stubConfig;
    }

    /// Runs one stub daemon on a fresh short-lived socket for a journey.
    static func withStubDaemon(
        _ journeyName: String,
        _ stubConfig: StubDaemon.StubDaemonConfig,
        _ journey: (String, StubDaemon) throws -> Void
    ) rethrows -> Void {
        let stateDirectory: String = CliJourneySupport.freshTestDirectory(journeyName);
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let socketPath: String = stateDirectory + "/ipc.sock";
        let stubDaemon: StubDaemon = StubDaemon(socketPath: socketPath, config: stubConfig);
        do {
            try stubDaemon.start();
        } catch {
            return;
        }
        defer { stubDaemon.stop() }
        try journey(socketPath, stubDaemon);
    }

    /// The resident-default stub with captured writers, the most common shape.
    static func withResidentDefaultStubDaemon(
        chatFragments: Array<String>,
        _ journey: (String, StubDaemon, BufferedTextOutputWriter, BufferedTextOutputWriter) throws -> Void
    ) rethrows -> Void {
        let stubConfig: StubDaemon.StubDaemonConfig = CliJourneySupport.residentDefaultStubConfig(
            chatFragments: chatFragments
        );
        return try CliJourneySupport.withStubDaemon("respond-default", stubConfig) { (socketPath: String, stubDaemon: StubDaemon) throws -> Void in
            let stdout: BufferedTextOutputWriter = BufferedTextOutputWriter();
            let stderr: BufferedTextOutputWriter = BufferedTextOutputWriter();
            try journey(socketPath, stubDaemon, stdout, stderr);
        };
    }

    /// Stub with one resident embeddings model named as the effective
    /// default: the no-flag resolution path.
    static func embedderDefaultStubConfig() -> StubDaemon.StubDaemonConfig {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.installedModels = [.embeddings("test/local-embedder", isResident: true)];
        stubConfig.defaultModelId = "test/local-embedder";
        return stubConfig;
    }

    /// Stub with two installed models (one resident) plus a three-entry
    /// catalog covering the ready / downloading / absent states.
    static func catalogStubConfig() -> StubDaemon.StubDaemonConfig {
        var stubConfig: StubDaemon.StubDaemonConfig = StubDaemon.StubDaemonConfig();
        stubConfig.installedModels = [
            .chat("test/local-chatter", isResident: true),
            .embeddings("test/local-embedder", isResident: false),
        ];
        stubConfig.defaultModelId = "test/local-chatter";
        stubConfig.catalogEntries = [
            .chat("test/ready-model", "ready-model", true),
            StubDaemon.StubCatalogEntry(
                huggingfaceId: "test/half-model",
                requestableModelId: "half-model",
                readyOnThisMac: false,
                downloadState: "downloading",
                contextWindow: 32768,
                supportsEmbeddings: false
            ),
            .chat("test/absent-model", "absent-model", false),
        ];
        return stubConfig;
    }

    // MARK: - Verb runners with Result surface

    static func runRespond(
        prompt: String,
        imagePaths: Array<String> = [],
        modelId: String?,
        noStream: Bool,
        candidateSocketPaths: Array<String>,
        stdout: TextOutputWriter,
        stderr: TextOutputWriter
    ) -> Result<Void, Error> {
        let respondArguments: RespondArguments = RespondArguments(
            prompt: prompt,
            imagePaths: imagePaths,
            modelId: modelId,
            instructions: nil,
            thinkingBudget: nil,
            schemaPath: nil,
            noStream: noStream
        );
        let respondDependencies: RespondDependencies = RespondDependencies(
            candidateSocketPaths: candidateSocketPaths,
            stdout: stdout,
            stderr: stderr,
            downloadStageBoundSeconds: 10,
            downloadPollIntervalSeconds: CliJourneySupport.testDownloadPollIntervalSeconds
        );
        return Result { try RespondCommand.run(
            respondArguments: respondArguments,
            respondDependencies: respondDependencies
        ) };
    }

    static func runRespondWithControls(
        prompt: String,
        instructions: String?,
        thinkingBudget: UInt16?,
        candidateSocketPaths: Array<String>,
        stdout: TextOutputWriter,
        stderr: TextOutputWriter
    ) -> Result<Void, Error> {
        let respondArguments: RespondArguments = RespondArguments(
            prompt: prompt,
            imagePaths: [],
            modelId: nil,
            instructions: instructions,
            thinkingBudget: thinkingBudget,
            schemaPath: nil,
            noStream: false
        );
        let respondDependencies: RespondDependencies = RespondDependencies(
            candidateSocketPaths: candidateSocketPaths,
            stdout: stdout,
            stderr: stderr,
            downloadStageBoundSeconds: 10,
            downloadPollIntervalSeconds: CliJourneySupport.testDownloadPollIntervalSeconds
        );
        return Result { try RespondCommand.run(
            respondArguments: respondArguments,
            respondDependencies: respondDependencies
        ) };
    }

    static func runRespondWithSchema(
        prompt: String,
        schemaPath: String,
        candidateSocketPaths: Array<String>,
        stdout: TextOutputWriter,
        stderr: TextOutputWriter
    ) -> Result<Void, Error> {
        let respondArguments: RespondArguments = RespondArguments(
            prompt: prompt,
            imagePaths: [],
            modelId: nil,
            instructions: nil,
            thinkingBudget: nil,
            schemaPath: schemaPath,
            noStream: false
        );
        let respondDependencies: RespondDependencies = RespondDependencies(
            candidateSocketPaths: candidateSocketPaths,
            stdout: stdout,
            stderr: stderr,
            downloadStageBoundSeconds: 10,
            downloadPollIntervalSeconds: CliJourneySupport.testDownloadPollIntervalSeconds
        );
        return Result { try RespondCommand.run(
            respondArguments: respondArguments,
            respondDependencies: respondDependencies
        ) };
    }

    static func runEmbed(
        text: String?,
        filePath: String?,
        modelId: String?,
        standardInputText: String,
        candidateSocketPaths: Array<String>,
        stdout: TextOutputWriter,
        stderr: TextOutputWriter
    ) -> Result<Void, Error> {
        let embedArguments: EmbedArguments = EmbedArguments(
            text: text,
            filePath: filePath,
            modelId: modelId
        );
        let embedDependencies: EmbedDependencies = EmbedDependencies(
            candidateSocketPaths: candidateSocketPaths,
            standardInputText: standardInputText,
            stdout: stdout,
            stderr: stderr,
            downloadStageBoundSeconds: 10,
            downloadPollIntervalSeconds: CliJourneySupport.testDownloadPollIntervalSeconds
        );
        return Result { try EmbedCommand.run(
            embedArguments: embedArguments,
            embedDependencies: embedDependencies
        ) };
    }

    static func runModels(
        _ modelsCommand: ModelsSubcommand,
        candidateSocketPaths: Array<String>,
        stdout: TextOutputWriter,
        stderr: TextOutputWriter
    ) -> Result<Void, Error> {
        let modelsDependencies: ModelsDependencies = ModelsDependencies(
            candidateSocketPaths: candidateSocketPaths,
            stdout: stdout,
            stderr: stderr,
            downloadStageBoundSeconds: 10,
            downloadPollIntervalSeconds: CliJourneySupport.testDownloadPollIntervalSeconds
        );
        return Result { try ModelsVerb.run(
            modelsCommand: modelsCommand,
            modelsDependencies: modelsDependencies
        ) };
    }
}
