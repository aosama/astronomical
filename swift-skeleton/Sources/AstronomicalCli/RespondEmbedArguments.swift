import Foundation

/**
 * Arguments for the ephemeral one-shot `astronomical respond` verb, porting
 * respond_arguments.rs plus the respond half of the shared parser.
 */
public struct RespondArguments: Equatable {

    /// The user's prompt, exactly as supplied on the command line.
    public let prompt: String;
    /// Raster images to attach to the prompt, in the order the user supplied them.
    public let imagePaths: Array<String>;
    /// Exact model identity to demand from the resident daemon, when given.
    public let modelId: String?;
    /// System-prompt-style guidance applied to the reply, when given.
    public let instructions: String?;
    /// Cap on the tokens a thinking model may spend reasoning, when given.
    public let thinkingBudget: UInt16?;
    /// Path to the JSON schema file the reply must satisfy, when given.
    public let schemaPath: String?;
    /// Buffer the answer and print it once instead of streaming fragments.
    public let noStream: Bool;

    public init(
        prompt: String,
        imagePaths: Array<String>,
        modelId: String?,
        instructions: String?,
        thinkingBudget: UInt16?,
        schemaPath: String?,
        noStream: Bool
    ) {
        self.prompt = prompt;
        self.imagePaths = imagePaths;
        self.modelId = modelId;
        self.instructions = instructions;
        self.thinkingBudget = thinkingBudget;
        self.schemaPath = schemaPath;
        self.noStream = noStream;
    }
}

/**
 * Arguments for the ephemeral one-shot `astronomical embed` verb, porting
 * embed_arguments.rs plus the embed half of the shared parser. Exactly one
 * input source survives parsing: text, file, or stdin.
 */
public struct EmbedArguments: Equatable {

    /// The text to embed, exactly as supplied on the command line.
    public let text: String?;
    /// File whose contents are embedded, when `--file` was given.
    public let filePath: String?;
    /// Exact model identity to demand from the resident daemon, when given.
    public let modelId: String?;

    public init(text: String?, filePath: String?, modelId: String?) {
        self.text = text;
        self.filePath = filePath;
        self.modelId = modelId;
    }
}

/// One `astronomical models` subcommand.
public enum ModelsSubcommand: Equatable {

    /// Models installed on this Mac.
    case list
    /// The release catalog: what can be downloaded and its local state.
    case supported
    /// Show the effective default model; with a MODEL_ID, persist it.
    case `default`(modelId: String?)
    /// Start (or resume) a download and wait, with live progress.
    case download(modelId: String)
}

/// The respond and embed halves of the shared argument parser.
enum RespondEmbedArgumentParser {

    static func parseRespondArguments(
        _ remainingArguments: Array<String>
    ) -> Result<RespondArguments, UsageError> {
        var prompt: String? = nil;
        var promptFromText: Bool = false;
        var modelId: String? = nil;
        var instructions: String? = nil;
        var thinkingBudget: UInt16? = nil;
        var noStream: Bool = false;
        var imagePaths: Array<String> = [];
        var schemaPath: String? = nil;
        var argumentIndex: Int = 0;
        while (argumentIndex < remainingArguments.count) {
            let argument: String = remainingArguments[argumentIndex];
            if (argument == "--schema") {
                if (schemaPath != nil) {
                    return .failure(.repeatedArgument("--schema"));
                }
                switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: argumentIndex + 1, flagName: "--schema")) {
                case let .success(schemaValue):
                    schemaPath = schemaValue;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
                continue;
            }
            if (argument == "--image") {
                switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: argumentIndex + 1, flagName: "--image")) {
                case let .success(imageValue):
                    imagePaths.append(imageValue);
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
                continue;
            }
            if (argument == "--model") {
                if (modelId != nil) {
                    return .failure(.repeatedArgument("--model"));
                }
                switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: argumentIndex + 1, flagName: "--model")) {
                case let .success(modelValue):
                    modelId = modelValue;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
                continue;
            }
            if (argument == "--text") {
                // An alternative to the positional prompt; the two are exclusive.
                if (promptFromText) {
                    return .failure(.repeatedArgument("--text"));
                }
                if (prompt != nil) {
                    return .failure(.respondPromptConflict);
                }
                switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: argumentIndex + 1, flagName: "--text")) {
                case let .success(textValue):
                    prompt = textValue;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                promptFromText = true;
                argumentIndex += 2;
                continue;
            }
            if (argument == "--instructions") {
                if (instructions != nil) {
                    return .failure(.repeatedArgument("--instructions"));
                }
                switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: argumentIndex + 1, flagName: "--instructions")) {
                case let .success(instructionsValue):
                    instructions = instructionsValue;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
                continue;
            }
            if (argument == "--thinking-budget") {
                if (thinkingBudget != nil) {
                    return .failure(.repeatedArgument("--thinking-budget"));
                }
                switch (RespondEmbedArgumentParser.parseThinkingBudget(remainingArguments, flagIndex: argumentIndex)) {
                case let .success(parsedBudget):
                    thinkingBudget = parsedBudget;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
                continue;
            }
            if (argument == "--no-stream") {
                if (noStream) {
                    return .failure(.repeatedArgument("--no-stream"));
                }
                noStream = true;
                argumentIndex += 1;
                continue;
            }
            if argument.hasPrefix("-") {
                return .failure(.unknownArgument(argument));
            }
            if (prompt != nil) {
                return .failure(.respondPromptConflict);
            }
            prompt = argument;
            argumentIndex += 1;
        }
        guard let resolvedPrompt: String = prompt else {
            return .failure(.respondPromptRequired);
        }
        return .success(RespondArguments(
            prompt: resolvedPrompt,
            imagePaths: imagePaths,
            modelId: modelId,
            instructions: instructions,
            thinkingBudget: thinkingBudget,
            schemaPath: schemaPath,
            noStream: noStream
        ));
    }

    /// Parses `--thinking-budget`'s value as a bounded token count. `0` is
    /// valid and means the model may spend no tokens reasoning.
    static func parseThinkingBudget(
        _ remainingArguments: Array<String>,
        flagIndex: Int
    ) -> Result<UInt16, UsageError> {
        let rawValue: String;
        switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: flagIndex + 1, flagName: "--thinking-budget")) {
        case let .success(budgetValue):
            rawValue = budgetValue;
        case let .failure(usageError):
            return .failure(usageError);
        }
        guard let parsedBudget: UInt16 = UInt16(rawValue) else {
            return .failure(.invalidThinkingBudget(rawValue));
        }
        return .success(parsedBudget);
    }

    static func parseEmbedArguments(
        _ remainingArguments: Array<String>
    ) -> Result<EmbedArguments, UsageError> {
        var text: String? = nil;
        var filePath: String? = nil;
        var modelId: String? = nil;
        var argumentIndex: Int = 0;
        while (argumentIndex < remainingArguments.count) {
            let argument: String = remainingArguments[argumentIndex];
            if (argument == "--model") {
                if (modelId != nil) {
                    return .failure(.repeatedArgument("--model"));
                }
                switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: argumentIndex + 1, flagName: "--model")) {
                case let .success(modelValue):
                    modelId = modelValue;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
                continue;
            }
            if (argument == "--file") {
                if (filePath != nil) {
                    return .failure(.repeatedArgument("--file"));
                }
                switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: argumentIndex + 1, flagName: "--file")) {
                case let .success(fileValue):
                    filePath = fileValue;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
                continue;
            }
            if argument.hasPrefix("-") {
                return .failure(.unknownArgument(argument));
            }
            if (text != nil) {
                return .failure(.unknownArgument(argument));
            }
            text = argument;
            argumentIndex += 1;
        }
        if (text != nil && filePath != nil) {
            return .failure(.embedInputConflict);
        }
        // No input at all parses: the journey resolves stdin and fails at
        // runtime when even stdin is empty.
        return .success(EmbedArguments(text: text, filePath: filePath, modelId: modelId));
    }
}
