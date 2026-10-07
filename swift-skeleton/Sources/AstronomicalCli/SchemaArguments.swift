import Foundation

/**
 * Arguments for `astronomical schema object`, porting schema_arguments.rs:
 * the fm-style modifier grammar where kind flags open a property and trailing
 * modifiers attach to the most recently opened one.
 */
public enum SchemaPropertyKind: Equatable {

    case string
    case integer
    case double
    case boolean

    public var kindFlag: String {
        switch (self) {
        case .string: return "--string"
        case .integer: return "--int"
        case .double: return "--double"
        case .boolean: return "--boolean"
        }
    }

    public var jsonTypeName: String {
        switch (self) {
        case .string: return "string"
        case .integer: return "integer"
        case .double: return "number"
        case .boolean: return "boolean"
        }
    }
}

/// One property as written on the command line, before nesting is resolved.
public struct SchemaPropertyInput: Equatable {

    public let dottedPath: String;
    public let kind: SchemaPropertyKind;
    public var isArray: Bool;
    public var isOptional: Bool;
    public var description: String?;

    public init(
        dottedPath: String,
        kind: SchemaPropertyKind,
        isArray: Bool,
        isOptional: Bool,
        description: String?
    ) {
        self.dottedPath = dottedPath;
        self.kind = kind;
        self.isArray = isArray;
        self.isOptional = isOptional;
        self.description = description;
    }
}

public struct SchemaArguments: Equatable {

    public let objectName: String;
    public let properties: Array<SchemaPropertyInput>;

    public init(objectName: String, properties: Array<SchemaPropertyInput>) {
        self.objectName = objectName;
        self.properties = properties;
    }
}

enum SchemaArgumentParser {

    static func parseSchemaArguments(
        _ remainingArguments: Array<String>
    ) -> Result<SchemaArguments, UsageError> {
        var objectName: String? = nil;
        var properties: Array<SchemaPropertyInput> = [];

        var argumentIndex: Int = 0;
        while (argumentIndex < remainingArguments.count) {
            let argumentText: String = remainingArguments[argumentIndex];
            switch (argumentText) {
            case "--name":
                if (objectName != nil) {
                    return .failure(.repeatedArgument("--name"));
                }
                switch (SchemaArgumentParser.requiredFlagValue(remainingArguments, flagIndex: argumentIndex, flag: "--name")) {
                case let .success(rawName):
                    if rawName.isEmpty || rawName.hasPrefix("-") {
                        return .failure(.missingValue("--name"));
                    }
                    objectName = rawName;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                argumentIndex += 2;
            case "--string", "--int", "--double", "--boolean":
                let kindFlag: String = argumentText;
                var rawDottedPath: String = "";
                switch (SchemaArgumentParser.requiredFlagValue(remainingArguments, flagIndex: argumentIndex, flag: kindFlag)) {
                case let .success(dottedPath):
                    rawDottedPath = dottedPath;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                if rawDottedPath.isEmpty || rawDottedPath.hasPrefix("-") {
                    return .failure(.missingValue(kindFlag));
                }
                switch (SchemaArgumentParser.validateDottedPropertyPath(rawDottedPath)) {
                case .success:
                    break;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                switch (SchemaArgumentParser.rejectConflictingPropertyPath(rawDottedPath, existingProperties: properties)) {
                case .success:
                    break;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                properties.append(SchemaPropertyInput(
                    dottedPath: rawDottedPath,
                    kind: SchemaArgumentParser.propertyKindForFlag(kindFlag),
                    isArray: false,
                    isOptional: false,
                    description: nil
                ));
                argumentIndex += 2;
            case "--array", "--optional":
                guard var modifiedProperty: SchemaPropertyInput = properties.last else {
                    return .failure(.schemaModifierWithoutProperty(argumentText));
                }
                if (argumentText == "--array") {
                    modifiedProperty.isArray = true;
                } else {
                    modifiedProperty.isOptional = true;
                }
                properties[properties.count - 1] = modifiedProperty;
                argumentIndex += 1;
            case "--description":
                var descriptionText: String = "";
                switch (SchemaArgumentParser.requiredFlagValue(remainingArguments, flagIndex: argumentIndex, flag: "--description")) {
                case let .success(description):
                    descriptionText = description;
                case let .failure(usageError):
                    return .failure(usageError);
                }
                if descriptionText.isEmpty || descriptionText.hasPrefix("-") {
                    return .failure(.missingValue("--description"));
                }
                guard var modifiedProperty: SchemaPropertyInput = properties.last else {
                    return .failure(.schemaModifierWithoutProperty("--description"));
                }
                modifiedProperty.description = descriptionText;
                properties[properties.count - 1] = modifiedProperty;
                argumentIndex += 2;
            default:
                return .failure(.unknownArgument(argumentText));
            }
        }

        guard let resolvedObjectName: String = objectName else {
            return .failure(.schemaNameRequired);
        }
        if properties.isEmpty {
            return .failure(.schemaPropertyRequired);
        }
        return .success(SchemaArguments(objectName: resolvedObjectName, properties: properties));
    }

    private static func propertyKindForFlag(_ kindFlag: String) -> SchemaPropertyKind {
        switch (kindFlag) {
        case "--string": return .string
        case "--int": return .integer
        case "--double": return .double
        default: return .boolean
        }
    }

    private static func requiredFlagValue(
        _ remainingArguments: Array<String>,
        flagIndex: Int,
        flag: String
    ) -> Result<String, UsageError> {
        guard flagIndex + 1 < remainingArguments.count else {
            return .failure(.missingValue(flag));
        }
        return .success(remainingArguments[flagIndex + 1]);
    }

    private static func validateDottedPropertyPath(_ dottedPath: String) -> Result<Void, UsageError> {
        let pathSegments: Array<String> = dottedPath.split(
            separator: ".",
            omittingEmptySubsequences: false
        ).map(String.init);
        for pathSegment: String in pathSegments {
            let isUsableSegment: Bool = !pathSegment.isEmpty
                && pathSegment.allSatisfy { (character: Character) -> Bool in
                    return character.isASCII
                        && (character.isLetter || character.isNumber || character == "_" || character == "-");
                };
            if !isUsableSegment {
                return .failure(.schemaInvalidPropertyPath(dottedPath));
            }
        }
        return .success(());
    }

    private static func rejectConflictingPropertyPath(
        _ candidatePath: String,
        existingProperties: Array<SchemaPropertyInput>
    ) -> Result<Void, UsageError> {
        for existingProperty: SchemaPropertyInput in existingProperties {
            let existingPath: String = existingProperty.dottedPath;
            let nestedPair: Bool = existingPath.hasPrefix("\(candidatePath).")
                || candidatePath.hasPrefix("\(existingPath).");
            if (nestedPair || existingPath == candidatePath) {
                return .failure(.schemaDuplicateProperty(candidatePath));
            }
        }
        return .success(());
    }
}
