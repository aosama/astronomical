import Foundation

import AstronomicalConfig;

/**
 * Parsed CLI invocation before any loopback work, porting the dispatch half
 * of apps/astronomical/src/arguments.rs. `--verbose`/`-v` may appear anywhere
 * and is silently ignored; `--help`/`-h` and `--version` win wherever they
 * appear.
 */
public enum CliCommand: Equatable {

    case help
    case version
    case launch(LaunchArguments)
    case respond(RespondArguments)
    case embed(EmbedArguments)
    case models(ModelsSubcommand)
    case status
    case schema(SchemaArguments)
    case validateConfig(ValidateConfigArguments)
}

/// Launch-specific arguments after `astronomical launch`.
public struct LaunchArguments: Equatable {

    public let toolSlug: String?;
    public let modelId: String?;

    public init(toolSlug: String?, modelId: String?) {
        self.toolSlug = toolSlug;
        self.modelId = modelId;
    }
}

/// The `astronomical status` argument surface: no arguments, so parsing only
/// rejects anything trailing.
public enum CliArgumentParser {

    /// The carried help contract, printed for `--help`/`-h` and usage errors.
    public static func helpText() -> String {
        return AstronomicalCliHelp.text;
    }

    /// Parses the arguments after the binary name.
    public static func parseCommand(
        _ processArguments: Array<String>
    ) -> Result<CliCommand, UsageError> {
        if processArguments.contains("--help") || processArguments.contains("-h") {
            return .success(.help);
        }
        if processArguments.contains("--version") {
            return .success(.version);
        }
        let remainingArguments: Array<String> = processArguments.filter { (argument: String) -> Bool in
            return argument != "--verbose" && argument != "-v";
        };
        if remainingArguments.isEmpty {
            return .failure(.missingCommand);
        }
        let commandName: String = remainingArguments[0];
        let trailingArguments: Array<String> = Array(remainingArguments.dropFirst());
        switch (commandName) {
        case "launch":
            return CliArgumentParser.parseLaunchArguments(trailingArguments).map { (launchArguments: LaunchArguments) -> CliCommand in
                return .launch(launchArguments);
            };
        case "respond":
            return RespondEmbedArgumentParser.parseRespondArguments(trailingArguments).map { (respondArguments: RespondArguments) -> CliCommand in
                return .respond(respondArguments);
            };
        case "embed":
            return RespondEmbedArgumentParser.parseEmbedArguments(trailingArguments).map { (embedArguments: EmbedArguments) -> CliCommand in
                return .embed(embedArguments);
            };
        case "models":
            return CliArgumentParser.parseModelsCommand(trailingArguments).map { (modelsCommand: ModelsSubcommand) -> CliCommand in
                return .models(modelsCommand);
            };
        case "status":
            if (remainingArguments.count > 1) {
                return .failure(.unknownArgument(remainingArguments[1]));
            }
            return .success(.status);
        case "schema":
            return CliArgumentParser.parseSchemaCommand(trailingArguments);
        case "validate":
            return CliArgumentParser.parseValidateCommand(trailingArguments);
        default:
            return .failure(.unknownCommand(commandName));
        }
    }

    private static func parseSchemaCommand(
        _ remainingArguments: Array<String>
    ) -> Result<CliCommand, UsageError> {
        guard let schemaTarget: String = remainingArguments.first else {
            return .failure(.missingCommand);
        }
        if (schemaTarget != "object") {
            return .failure(.unknownSchemaTarget(schemaTarget));
        }
        return SchemaArgumentParser.parseSchemaArguments(Array(remainingArguments.dropFirst()))
            .map { (schemaArguments: SchemaArguments) -> CliCommand in
                return .schema(schemaArguments);
            };
    }

    private static func parseValidateCommand(
        _ remainingArguments: Array<String>
    ) -> Result<CliCommand, UsageError> {
        guard let validateTarget: String = remainingArguments.first else {
            return .failure(.missingCommand);
        }
        if (validateTarget != "config") {
            return .failure(.unknownValidateTarget(validateTarget));
        }
        return ValidateConfigArgumentParser.parseValidateConfigArguments(Array(remainingArguments.dropFirst()))
            .map { (validateArguments: ValidateConfigArguments) -> CliCommand in
                return .validateConfig(validateArguments);
            };
    }

    private static func parseModelsCommand(
        _ remainingArguments: Array<String>
    ) -> Result<ModelsSubcommand, UsageError> {
        guard let subcommandName: String = remainingArguments.first else {
            return .failure(.modelsSubcommandRequired);
        }
        let trailingArguments: Array<String> = Array(remainingArguments.dropFirst());
        switch (subcommandName) {
        case "list":
            return CliArgumentParser.rejectTrailingArguments(trailingArguments)
                .map { (_: Void) -> ModelsSubcommand in return .list; };
        case "supported":
            return CliArgumentParser.rejectTrailingArguments(trailingArguments)
                .map { (_: Void) -> ModelsSubcommand in return .supported; };
        case "default":
            if let firstTrailingArgument: String = trailingArguments.first {
                if firstTrailingArgument.hasPrefix("-") {
                    return .failure(.unknownArgument(firstTrailingArgument));
                }
                if (trailingArguments.count > 1) {
                    return .failure(.unknownArgument(trailingArguments[1]));
                }
                switch (CliArgumentParser.parsePositionalModelId(firstTrailingArgument)) {
                case let .success(parsedModelId):
                    return .success(.default(modelId: parsedModelId));
                case let .failure(usageError):
                    return .failure(usageError);
                }
            }
            return .success(.default(modelId: nil));
        case "download":
            guard let rawModelId: String = trailingArguments.first else {
                return .failure(.modelsDownloadModelRequired);
            }
            if rawModelId.hasPrefix("-") {
                return .failure(.unknownArgument(rawModelId));
            }
            if (trailingArguments.count > 1) {
                return .failure(.unknownArgument(trailingArguments[1]));
            }
            switch (CliArgumentParser.parsePositionalModelId(rawModelId)) {
            case let .success(parsedModelId):
                return .success(.download(modelId: parsedModelId));
            case let .failure(usageError):
                return .failure(usageError);
            }
        default:
            return .failure(.unknownModelsSubcommand(subcommandName));
        }
    }

    private static func parseLaunchArguments(
        _ remainingArguments: Array<String>
    ) -> Result<LaunchArguments, UsageError> {
        var toolSlug: String? = nil;
        var modelId: String? = nil;
        var argumentIndex: Int = 0;
        while (argumentIndex < remainingArguments.count) {
            let argument: String = remainingArguments[argumentIndex];
            if (argument == "--model") {
                if (modelId != nil) {
                    return .failure(.repeatedArgument("--model"));
                }
                switch (CliArgumentParser.flagValue(remainingArguments, valueIndex: argumentIndex + 1, flagName: "--model")) {
                case let .success(flagModelId):
                    modelId = flagModelId;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
                continue;
            }
            if argument.hasPrefix("-") {
                return .failure(.unknownArgument(argument));
            }
            if (toolSlug != nil) {
                return .failure(.multipleTools);
            }
            toolSlug = argument;
            argumentIndex += 1;
        }
        return .success(LaunchArguments(toolSlug: toolSlug, modelId: modelId));
    }

    /// Reads the value that must follow a value-taking flag. A missing,
    /// empty, or flag-shaped value is the same usage failure: the flag's
    /// value is what is missing.
    static func flagValue(
        _ remainingArguments: Array<String>,
        valueIndex: Int,
        flagName: String
    ) -> Result<String, UsageError> {
        guard valueIndex < remainingArguments.count else {
            return .failure(.missingValue(flagName));
        }
        let rawValue: String = remainingArguments[valueIndex];
        if rawValue.isEmpty || rawValue.hasPrefix("-") {
            return .failure(.missingValue(flagName));
        }
        return .success(rawValue);
    }

    static func rejectTrailingArguments(
        _ trailingArguments: Array<String>
    ) -> Result<Void, UsageError> {
        if let extraArgument: String = trailingArguments.first {
            return .failure(.unknownArgument(extraArgument));
        }
        return .success(());
    }

    static func parsePositionalModelId(_ rawModelId: String) -> Result<String, UsageError> {
        if rawModelId.isEmpty {
            return .failure(.missingValue("MODEL_ID"));
        }
        return .success(rawModelId);
    }
}
