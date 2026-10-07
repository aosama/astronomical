import Foundation

/**
 * Builds process-scoped OpenCode config, porting opencode.rs. Uses OpenCode's
 * documented `OPENCODE_CONFIG_CONTENT` overlay so this session does not write
 * `~/.config/opencode`. Keys serialize alphabetically, mirroring serde_json.
 */
public enum OpencodeConfig {

    public static let opencodeConfigContentVariable: String = "OPENCODE_CONFIG_CONTENT";
    static let opencodeProviderId: String = "astronomical";

    static func opencodeConfigContent(
        host: String,
        port: UInt16,
        chatModel: LaunchModelSelection.LibraryChatModel
    ) -> String? {
        let openaiBaseUrl: String = "http://\(host):\(port)/v1";
        let providerModelId: String = "\(OpencodeConfig.opencodeProviderId)/\(chatModel.modelId)";
        var providerModels: Dictionary<String, Any> = [:];
        providerModels[chatModel.modelId] = ["name": chatModel.modelId];
        let configDocument: Dictionary<String, Any> = [
            "$schema": "https://opencode.ai/config.json",
            "provider": [
                OpencodeConfig.opencodeProviderId: [
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Astronomical",
                    "options": [
                        "baseURL": openaiBaseUrl,
                    ],
                    "models": providerModels,
                ],
            ],
            "model": providerModelId,
        ];
        guard let serializedConfig: Data = try? JSONSerialization.data(
            withJSONObject: configDocument,
            options: [.sortedKeys, .withoutEscapingSlashes]
        ) else {
            return nil;
        }
        return String(decoding: serializedConfig, as: UTF8.self);
    }
}

/**
 * Launch journey, porting launch.rs: resolve tool, find the running loopback
 * instance, pick a chat model only when needed, then prepare a
 * process-scoped harness exec.
 */
public enum LaunchCommand {

    /// Bound for each loopback probe of the launch journey.
    public static let defaultLoopbackTimeoutSeconds: Double = 3;
    static let statusPath: String = "/v1/status";
    static let modelsPath: String = "/v1/models";

    /// Collaborators tests replace so journeys do not exec a real harness.
    public struct LaunchDependencies {

        /// Ordered loopback candidates; the first one answering status wins.
        public let candidateBindEndpoints: Array<(host: String, port: UInt16)>;
        public let pathValue: String;
        public let isInteractive: Bool;
        /// The prompt input line when the picker must ask; `nil` means EOF.
        public let selectionInput: String?;
        public let stderr: TextOutputWriter;
        public let httpTimeoutSeconds: Double;

        public init(
            candidateBindEndpoints: Array<(host: String, port: UInt16)>,
            pathValue: String,
            isInteractive: Bool,
            selectionInput: String?,
            stderr: TextOutputWriter,
            httpTimeoutSeconds: Double = LaunchCommand.defaultLoopbackTimeoutSeconds
        ) {
            self.candidateBindEndpoints = candidateBindEndpoints;
            self.pathValue = pathValue;
            self.isInteractive = isInteractive;
            self.selectionInput = selectionInput;
            self.stderr = stderr;
            self.httpTimeoutSeconds = httpTimeoutSeconds;
        }
    }

    /// Process image the CLI will exec. Tests inspect this instead of
    /// replacing the test process.
    public struct PreparedLaunch {

        public let programPath: String;
        public let extraEnvironment: Array<(String, String)>;

        public init(programPath: String, extraEnvironment: Array<(String, String)>) {
            self.programPath = programPath;
            self.extraEnvironment = extraEnvironment;
        }
    }

    /// Resolves the harness, the instance, and the session config.
    public static func prepareLaunch(
        launchArguments: LaunchArguments,
        launchDependencies: LaunchDependencies
    ) -> Result<PreparedLaunch, LaunchError> {
        let resolvedLaunchTool: LaunchTools.ResolvedLaunchTool;
        switch (LaunchTools.resolveLaunchTool(
            requestedToolSlug: launchArguments.toolSlug,
            pathValue: launchDependencies.pathValue
        )) {
        case let .success(resolvedTool):
            resolvedLaunchTool = resolvedTool;
        case let .failure(launchError):
            return .failure(launchError);
        }
        let chosenBindEndpoint: (host: String, port: UInt16);
        switch (LaunchCommand.firstHealthyInstance(
            launchDependencies.candidateBindEndpoints,
            httpTimeoutSeconds: launchDependencies.httpTimeoutSeconds
        )) {
        case let .success(healthyEndpoint):
            chosenBindEndpoint = healthyEndpoint;
        case .failure:
            return .failure(.astronomicalUnavailable);
        }
        let modelsDocument: Any;
        switch (LaunchHttp.getLoopbackJson(
            host: chosenBindEndpoint.host,
            port: chosenBindEndpoint.port,
            requestPath: LaunchCommand.modelsPath,
            timeoutSeconds: launchDependencies.httpTimeoutSeconds
        )) {
        case let .success(fetchedDocument):
            modelsDocument = fetchedDocument;
        case .failure:
            return .failure(.modelListUnavailable);
        }
        let chatModels: Array<LaunchModelSelection.LibraryChatModel>;
        switch (LaunchModelSelection.chatModelsFromModelsDocument(modelsDocument)) {
        case let .success(parsedChatModels):
            chatModels = parsedChatModels;
        case let .failure(launchError):
            return .failure(launchError);
        }
        let selectedChatModel: LaunchModelSelection.LibraryChatModel;
        switch (LaunchModelSelection.selectChatModel(
            chatModels: chatModels,
            requestedModelId: launchArguments.modelId,
            isInteractive: launchDependencies.isInteractive,
            selectionInput: launchDependencies.selectionInput,
            stderr: launchDependencies.stderr
        )) {
        case let .success(selectedModel):
            selectedChatModel = selectedModel;
        case let .failure(launchError):
            return .failure(launchError);
        }
        LaunchModelSelection.warnIfContextWindowIsNarrow(
            selectedChatModel,
            stderr: launchDependencies.stderr
        );
        guard let configContent: String = OpencodeConfig.opencodeConfigContent(
            host: chosenBindEndpoint.host,
            port: chosenBindEndpoint.port,
            chatModel: selectedChatModel
        ) else {
            return .failure(.openCodeConfigFailed);
        }
        return .success(PreparedLaunch(
            programPath: resolvedLaunchTool.programPath,
            extraEnvironment: [(OpencodeConfig.opencodeConfigContentVariable, configContent)]
        ));
    }

    /// Returns the first candidate whose `/v1/status` looks like Astronomical.
    /// Status must include `application` so a random listener on the same
    /// port is not treated as the product.
    private static func firstHealthyInstance(
        _ candidateBindEndpoints: Array<(host: String, port: UInt16)>,
        httpTimeoutSeconds: Double
    ) -> Result<(host: String, port: UInt16), LaunchError> {
        for candidateBindEndpoint: (host: String, port: UInt16) in candidateBindEndpoints {
            switch (LaunchHttp.getLoopbackJson(
                host: candidateBindEndpoint.host,
                port: candidateBindEndpoint.port,
                requestPath: LaunchCommand.statusPath,
                timeoutSeconds: httpTimeoutSeconds
            )) {
            case let .success(statusDocument):
                guard let statusObject: Dictionary<String, Any> = statusDocument as? Dictionary<String, Any>,
                      statusObject["application"] != nil else {
                    continue;
                }
                return .success(candidateBindEndpoint);
            case .failure:
                continue;
            }
        }
        return .failure(.astronomicalUnavailable);
    }
}
