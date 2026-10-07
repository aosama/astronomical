import Foundation

/** Command-line usage failures. These drive exit code 2. Port of errors.rs UsageError. */
public enum UsageError: Error, Equatable, CustomStringConvertible {

    case missingCommand
    case unknownCommand(String)
    case unknownArgument(String)
    case missingValue(String)
    case repeatedArgument(String)
    case multipleTools
    case schemaNameRequired
    case schemaPropertyRequired
    case schemaModifierWithoutProperty(String)
    case schemaDuplicateProperty(String)
    case schemaInvalidPropertyPath(String)
    case unknownSchemaTarget(String)
    case unknownValidateTarget(String)
    case unknownInstance(String)
    case respondPromptRequired
    case respondPromptConflict
    case invalidThinkingBudget(String)
    case embedInputConflict
    case modelsSubcommandRequired
    case unknownModelsSubcommand(String)
    case modelsDownloadModelRequired

    public var description: String {
        switch (self) {
        case .missingCommand:
            return "missing command. Try: astronomical launch opencode"
        case let .unknownCommand(commandName):
            return "unknown command: \(commandName)"
        case let .unknownArgument(argumentName):
            return "unrecognized argument: \(argumentName)"
        case let .missingValue(flagName):
            return "missing value for \(flagName)"
        case let .repeatedArgument(flagName):
            return "argument may be supplied only once: \(flagName)"
        case .multipleTools:
            return "launch accepts at most one tool name"
        case .schemaNameRequired:
            return "schema object needs --name NAME"
        case .schemaPropertyRequired:
            return "schema object needs at least one property. Try: astronomical schema object "
                + "--name Thing --string label"
        case let .schemaModifierWithoutProperty(modifierName):
            return "\(modifierName) must follow a property"
        case let .schemaDuplicateProperty(propertyPath):
            return "property may be supplied only once: \(propertyPath)"
        case let .schemaInvalidPropertyPath(propertyPath):
            return "invalid property path: \(propertyPath)"
        case let .unknownSchemaTarget(schemaTarget):
            return "unknown schema target: \(schemaTarget). Supported targets: object"
        case let .unknownValidateTarget(validateTarget):
            return "unknown validate target: \(validateTarget). Supported targets: config"
        case let .unknownInstance(instanceName):
            return "unknown instance \(instanceName): expected stable or development"
        case .respondPromptRequired:
            return "respond needs a prompt. Try: astronomical respond 'Hello'"
        case .respondPromptConflict:
            return "respond takes one prompt. Use either the positional PROMPT or --text, "
                + "not both."
        case let .invalidThinkingBudget(rawValue):
            return "--thinking-budget must be a whole number of tokens from 0 to 65535: "
                + "\(rawValue)"
        case .embedInputConflict:
            return "embed takes one input. Pass TEXT, --file PATH, or pipe stdin — not several."
        case .modelsSubcommandRequired:
            return "models needs a subcommand: list, supported, default, download"
        case let .unknownModelsSubcommand(subcommandName):
            return "unknown models subcommand: \(subcommandName). Try: list, supported, "
                + "default, download"
        case .modelsDownloadModelRequired:
            return "models download needs a MODEL_ID. Try: astronomical models download "
                + "Qwen3.5-2B-4bit"
        }
    }
}
