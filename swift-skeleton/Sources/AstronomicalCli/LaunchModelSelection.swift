import Foundation

/**
 * Supported launch tools, porting tools.rs: only OpenCode is launchable;
 * other names fail closed instead of showing a fake picker.
 */
public enum LaunchTools {

    public static let opencodeSlug: String = "opencode";
    static let opencodeBinaryName: String = "opencode";

    /// A supported harness that is present on PATH.
    public struct ResolvedLaunchTool: Equatable {

        public let slug: String;
        public let programPath: String;

        public init(slug: String, programPath: String) {
            self.slug = slug;
            self.programPath = programPath;
        }
    }

    static func resolveLaunchTool(
        requestedToolSlug: String?,
        pathValue: String
    ) -> Result<ResolvedLaunchTool, LaunchError> {
        if let requestedToolSlug: String = requestedToolSlug {
            if (requestedToolSlug != LaunchTools.opencodeSlug) {
                return .failure(.unknownTool(requestedTool: requestedToolSlug));
            }
        }
        guard let programPath: String = LaunchTools.findExecutable(
            LaunchTools.opencodeBinaryName,
            pathValue: pathValue
        ) else {
            return .failure(.openCodeMissing);
        }
        return .success(ResolvedLaunchTool(slug: LaunchTools.opencodeSlug, programPath: programPath));
    }

    private static func findExecutable(_ binaryName: String, pathValue: String) -> String? {
        for directory: Substring in pathValue.split(separator: ":") {
            if directory.isEmpty {
                continue;
            }
            let candidatePath: String = "\(directory)/\(binaryName)";
            if (LaunchTools.isExecutableFile(candidatePath)) {
                return candidatePath;
            }
        }
        return nil;
    }

    private static func isExecutableFile(_ candidatePath: String) -> Bool {
        // Follows PATH shims so a symlink to OpenCode still counts as installed.
        guard FileManager.default.isExecutableFile(atPath: candidatePath) else {
            return false;
        }
        var isDirectory: ObjCBool = false;
        let fileExists: Bool = FileManager.default.fileExists(atPath: candidatePath, isDirectory: &isDirectory);
        return fileExists && !isDirectory.boolValue;
    }
}

/**
 * Selects a Library chat model from `/v1/models`, porting models.rs and
 * prompt.rs. Image and embedding advertisements are not coding-harness
 * launch targets.
 */
public enum LaunchModelSelection {

    /// One chat model the launched session may use.
    public struct LibraryChatModel: Equatable {

        public let modelId: String;
        public let contextWindowTokens: UInt32?;

        public init(modelId: String, contextWindowTokens: UInt32?) {
            self.modelId = modelId;
            self.contextWindowTokens = contextWindowTokens;
        }
    }

    private static let chatCompletionsEndpoint: String = "/v1/chat/completions";
    private static let opencodePreferredContextWindowTokens: UInt32 = 65_536;

    /// Keeps advertised order so the picker matches Library listing.
    static func chatModelsFromModelsDocument(
        _ modelsDocument: Any
    ) -> Result<Array<LibraryChatModel>, LaunchError> {
        guard let documentObject: Dictionary<String, Any> = modelsDocument as? Dictionary<String, Any>,
              let advertisedModels: Array<Any> = documentObject["data"] as? Array<Any> else {
            return .failure(.modelListUnavailable);
        }
        var chatModels: Array<LibraryChatModel> = [];
        for advertisedModel: Any in advertisedModels {
            guard let advertisedModelObject: Dictionary<String, Any> = advertisedModel as? Dictionary<String, Any>,
                  let modelId: String = advertisedModelObject["id"] as? String else {
                continue;
            }
            if !LaunchModelSelection.advertisesChatCompletions(advertisedModelObject) {
                continue;
            }
            let contextWindowTokens: UInt32? = (advertisedModelObject["context_window"] as? Int)
                .flatMap { (contextWindow: Int) -> UInt32? in
                    return UInt32(exactly: contextWindow);
                };
            chatModels.append(LibraryChatModel(modelId: modelId, contextWindowTokens: contextWindowTokens));
        }
        return .success(chatModels);
    }

    private static func advertisesChatCompletions(
        _ advertisedModel: Dictionary<String, Any>
    ) -> Bool {
        guard let supportedEndpoints: Array<Any> = advertisedModel["supported_endpoints"] as? Array<Any> else {
            return false;
        }
        return supportedEndpoints.contains { (supportedEndpoint: Any) -> Bool in
            return (supportedEndpoint as? String) == LaunchModelSelection.chatCompletionsEndpoint;
        }
    }

    /// Asks for a Library chat model only when the session has several and
    /// the user did not already pass `--model`.
    static func selectChatModel(
        chatModels: Array<LibraryChatModel>,
        requestedModelId: String?,
        isInteractive: Bool,
        selectionInput: String?,
        stderr: TextOutputWriter
    ) -> Result<LibraryChatModel, LaunchError> {
        if chatModels.isEmpty {
            return .failure(.noChatModels);
        }
        if let requestedModelId: String = requestedModelId {
            guard let requestedModel: LibraryChatModel = chatModels.first(where: { (chatModel: LibraryChatModel) -> Bool in
                return chatModel.modelId == requestedModelId;
            }) else {
                return .failure(.requestedModelMissing(requestedModelId: requestedModelId));
            }
            return .success(requestedModel);
        }
        if (chatModels.count == 1) {
            return .success(chatModels[0]);
        }
        if !isInteractive {
            return .failure(.modelPickerRequired);
        }
        _ = stderr.write("Select a model:\n");
        for (modelIndex, chatModel) in chatModels.enumerated() {
            _ = stderr.write("\(modelIndex + 1). \(chatModel.modelId)\n");
        }
        guard let selectedLine: String = selectionInput else {
            return .failure(.invalidModelSelection);
        }
        let selectedText: String = selectedLine.trimmingCharacters(in: .whitespacesAndNewlines);
        if selectedText.isEmpty {
            return .failure(.invalidModelSelection);
        }
        if let selectedNumber: Int = Int(selectedText) {
            if (selectedNumber >= 1 && selectedNumber <= chatModels.count) {
                return .success(chatModels[selectedNumber - 1]);
            }
            return .failure(.invalidModelSelection);
        }
        guard let typedModel: LibraryChatModel = chatModels.first(where: { (chatModel: LibraryChatModel) -> Bool in
            return chatModel.modelId == selectedText;
        }) else {
            return .failure(.invalidModelSelection);
        }
        return .success(typedModel);
    }

    static func warnIfContextWindowIsNarrow(
        _ chatModel: LibraryChatModel,
        stderr: TextOutputWriter
    ) -> Void {
        guard let contextWindowTokens: UInt32 = chatModel.contextWindowTokens else {
            return;
        }
        if (contextWindowTokens >= LaunchModelSelection.opencodePreferredContextWindowTokens) {
            return;
        }
        // Advisory only: a write failure must not block launch.
        _ = stderr.write("OpenCode works better with a 64k or larger context window; "
            + "\(chatModel.modelId) advertises \(contextWindowTokens) tokens.\n");
    }
}
