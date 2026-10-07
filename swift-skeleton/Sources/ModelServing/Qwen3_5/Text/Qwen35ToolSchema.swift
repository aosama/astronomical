import Foundation;

import IpcProtocol;

/// One parsed tool-argument value with serde-compatible JSON serialization.
enum Qwen35ParsedToolValue {
    case null;
    case boolean(Bool);
    case integer(Int64);
    case number(Double);
    case string(String);
    case array(Array<Qwen35ParsedToolValue>);
    case object(Array<(key: String, value: Qwen35ParsedToolValue)>);

    static func == (lhsValue: Qwen35ParsedToolValue, rhsValue: Qwen35ParsedToolValue) -> Bool {
        switch (lhsValue, rhsValue) {
        case (.null, .null):
            return true;
        case let (.boolean(leftBoolean), .boolean(rightBoolean)):
            return leftBoolean == rightBoolean;
        case let (.integer(leftInteger), .integer(rightInteger)):
            return leftInteger == rightInteger;
        case let (.number(leftNumber), .number(rightNumber)):
            return leftNumber == rightNumber;
        case let (.string(leftText), .string(rightText)):
            return leftText == rightText;
        case let (.array(leftEntries), .array(rightEntries)):
            guard leftEntries.count == rightEntries.count else {
                return false;
            }
            return zip(leftEntries, rightEntries).allSatisfy { $0 == $1 };
        case let (.object(leftEntries), .object(rightEntries)):
            guard leftEntries.count == rightEntries.count else {
                return false;
            }
            return zip(leftEntries, rightEntries).allSatisfy { (pair) -> Bool in
                pair.0.key == pair.1.key && pair.0.value == pair.1.value;
            };
        default:
            return false;
        }
    }

    /// Serializes like serde_json over its default `BTreeMap` objects: keys
    /// sorted by UTF-8 bytes, no whitespace, minimal string escaping.
    func serializedJson() -> String {
        var outputText = "";
        self.writeJson(into: &outputText);
        return outputText;
    }

    private func writeJson(into outputText: inout String) {
        switch self {
        case .null:
            outputText += "null";
        case let .boolean(booleanValue):
            outputText += booleanValue ? "true" : "false";
        case let .integer(integerValue):
            outputText += String(integerValue);
        case let .number(numberValue):
            outputText += String(numberValue);
        case let .string(stringValue):
            Qwen35ParsedToolValue.writeEscaped(stringValue, into: &outputText);
        case let .array(entryValues):
            outputText += "[";
            for (entryIndex, entryValue) in entryValues.enumerated() {
                if entryIndex > 0 {
                    outputText += ",";
                }
                entryValue.writeJson(into: &outputText);
            }
            outputText += "]";
        case let .object(entryValues):
            outputText += "{";
            let sortedEntries = entryValues.sorted { (left, right) -> Bool in
                Array(left.key.utf8).lexicographicallyPrecedes(Array(right.key.utf8));
            };
            for (entryIndex, entry) in sortedEntries.enumerated() {
                if entryIndex > 0 {
                    outputText += ",";
                }
                Qwen35ParsedToolValue.writeEscaped(entry.key, into: &outputText);
                outputText += ":";
                entry.value.writeJson(into: &outputText);
            }
            outputText += "}";
        }
    }

    private static func writeEscaped(_ sourceText: String, into outputText: inout String) {
        outputText += "\"";
        for scalar in sourceText.unicodeScalars {
            switch scalar {
            case "\"": outputText += "\\\"";
            case "\\": outputText += "\\\\";
            case "\n": outputText += "\\n";
            case "\r": outputText += "\\r";
            case "\t": outputText += "\\t";
            case "\u{08}": outputText += "\\b";
            case "\u{0C}": outputText += "\\f";
            default:
                if scalar.value < 0x20 {
                    outputText += String(
                        format: "\\u%04x", UInt(scalar.value));
                } else {
                    outputText.unicodeScalars.append(scalar);
                }
            }
        }
        outputText += "\"";
    }
}

/// One declared tool's resolved parameter schemas, read once at parser
/// construction. Only the branch shapes that change argument coercion are
/// modeled; anything else degrades to dynamic JSON parsing.
struct Qwen35DeclaredTool {
    let parameterSchemas: Dictionary<String, Qwen35DeclaredParameterSchema>;

    init(toolDefinition: ChatToolDefinition) throws {
        let schemaBytes = Array(toolDefinition.parametersJson.utf8);
        let schemaValue: JsonWireValue;
        do {
            schemaValue = try JsonWireParser.parseDocument(documentBytes: Data(schemaBytes));
        } catch {
            throw Qwen35OutputParserError.invalidDeclaredToolSchema(
                functionName: toolDefinition.name, problem: "the schema is not valid JSON");
        }
        guard case let .object(schemaObject) = schemaValue else {
            throw Qwen35OutputParserError.declaredToolSchemaMustBeObject(
                functionName: toolDefinition.name);
        }
        var resolvedSchemas: Dictionary<String, Qwen35DeclaredParameterSchema> = [:];
        guard case let .object(propertySchemas)? = schemaObject.value(forKey: "properties") else {
            self.parameterSchemas = resolvedSchemas;
            return;
        }
        for propertyEntry in propertySchemas.entries {
            guard case let .object(propertyObject) = propertyEntry.value else {
                throw Qwen35OutputParserError.invalidToolPropertySchema(
                    functionName: toolDefinition.name, parameterName: propertyEntry.key);
            }
            resolvedSchemas[propertyEntry.key] = try Qwen35DeclaredTool.resolveParameterSchema(
                functionName: toolDefinition.name, parameterName: propertyEntry.key,
                propertyObject: propertyObject);
        }
        self.parameterSchemas = resolvedSchemas;
    }

    /// Resolves `type`, the two-branch nullable pair, and the flexible `anyOf`
    /// shapes. An unresolvable union degrades to dynamic JSON parsing because
    /// JSON Schema makes `type` optional and real harness inventories declare
    /// unions no resolver can reduce (#736, #771).
    private static func resolveParameterSchema(
        functionName: String, parameterName: String, propertyObject: JsonWireObject
    ) throws -> Qwen35DeclaredParameterSchema {
        switch propertyObject.value(forKey: "type") {
        case let .string(parameterType):
            return Qwen35DeclaredParameterSchema(
                parameterType: parameterType, isNullable: false);
        case let .array(parameterTypes):
            return try Qwen35DeclaredTool.resolveNullableTypeList(
                functionName: functionName, parameterName: parameterName,
                parameterTypes: parameterTypes);
        case nil:
            if let nullableResolution = Qwen35DeclaredTool.parseNullableAnyOfType(propertyObject) {
                return nullableResolution;
            }
            if let flexibleResolution = Qwen35DeclaredTool.parseFlexibleAnyOfType(propertyObject) {
                return flexibleResolution;
            }
            return Qwen35DeclaredParameterSchema(parameterType: nil, isNullable: false);
        default:
            throw Qwen35OutputParserError.invalidToolParameterTypeDeclaration(
                functionName: functionName, parameterName: parameterName);
        }
    }

    private static func resolveNullableTypeList(
        functionName: String, parameterName: String, parameterTypes: Array<JsonWireValue>
    ) throws -> Qwen35DeclaredParameterSchema {
        let nonNullTypeTexts = parameterTypes.compactMap { (parameterType: JsonWireValue) -> String? in
            guard case let .string(typeText) = parameterType, typeText != "null" else {
                return nil;
            }
            return typeText;
        };
        guard let firstNonNullType = nonNullTypeTexts.first else {
            throw Qwen35OutputParserError.invalidToolParameterTypeDeclaration(
                functionName: functionName, parameterName: parameterName);
        }
        let hasExactNullablePair = nonNullTypeTexts.count == 1
            && parameterTypes.count == 2
            && parameterTypes.contains(.string("null"));
        guard hasExactNullablePair else {
            throw Qwen35OutputParserError.invalidToolParameterTypeDeclaration(
                functionName: functionName, parameterName: parameterName);
        }
        return Qwen35DeclaredParameterSchema(
            parameterType: firstNonNullType, isNullable: true);
    }

    /// `anyOf: [{...}, {"type": "null"}]` where the non-null branch only
    /// carries coercion-neutral constraint keywords.
    private static func parseNullableAnyOfType(
        _ propertyObject: JsonWireObject
    ) -> Qwen35DeclaredParameterSchema? {
        guard case let .array(anyOfSchemas)? = propertyObject.value(forKey: "anyOf"),
            anyOfSchemas.count == 2
        else {
            return nil;
        }
        var nonNullType: String? = nil;
        var hasNullType = false;
        for anyOfSchema in anyOfSchemas {
            guard case let .object(branchObject) = anyOfSchema else {
                return nil;
            }
            switch branchObject.value(forKey: "type") {
            case let .string(typeText) where typeText == "null":
                if branchObject.entries.count != 1 {
                    return nil;
                }
                hasNullType = true;
            case let .string(typeText) where nonNullType == nil:
                let allowedKeywords: Set<String> = [
                    "type", "description", "title", "default", "minimum", "maximum",
                    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
                    "pattern", "format", "minItems", "maxItems", "minProperties",
                    "maxProperties", "uniqueItems",
                ];
                if branchObject.keyNames.contains(where: { (schemaKeyword: String) -> Bool in
                    !allowedKeywords.contains(schemaKeyword)
                }) {
                    return nil;
                }
                nonNullType = typeText;
            default:
                return nil;
            }
        }
        guard hasNullType, let resolvedType = nonNullType else {
            return nil;
        }
        return Qwen35DeclaredParameterSchema(parameterType: resolvedType, isNullable: true);
    }

    /// How a multi-branch `anyOf` declaration maps to argument parsing.
    private static func parseFlexibleAnyOfType(
        _ propertyObject: JsonWireObject
    ) -> Qwen35DeclaredParameterSchema? {
        guard case let .array(anyOfSchemas)? = propertyObject.value(forKey: "anyOf"),
            anyOfSchemas.count >= 2
        else {
            return nil;
        }
        var uniformType: String? = nil;
        for anyOfSchema in anyOfSchemas {
            guard case let .object(branchObject) = anyOfSchema,
                case let .string(branchType)? = branchObject.value(forKey: "type")
            else {
                return nil;
            }
            if branchType == "null" {
                if branchObject.entries.count != 1 {
                    return nil;
                }
                continue;
            }
            let coercibleTypes: Set<String> = [
                "string", "boolean", "integer", "number", "array", "object",
            ];
            guard coercibleTypes.contains(branchType) else {
                return nil;
            }
            if let seenType = uniformType {
                if seenType != branchType {
                    return Qwen35DeclaredParameterSchema(parameterType: nil, isNullable: false);
                }
            } else {
                uniformType = branchType;
            }
        }
        return uniformType.map { (sharedType: String) -> Qwen35DeclaredParameterSchema in
            Qwen35DeclaredParameterSchema(parameterType: sharedType, isNullable: false);
        };
    }
}

struct Qwen35DeclaredParameterSchema {
    let parameterType: String?;
    let isNullable: Bool;
}

/// Parses the raw `<parameter=k>value</parameter>` content of one tool-call
/// body into a JSON object. Duplicate names overwrite (last value wins), and
/// unknown or defective shapes degrade to dynamic parsing or raw strings —
/// the harness owns rejection, so parsing never fails the generation.
enum Qwen35ToolSchemaParser {

    static func parseToolParameters(
        parameterContent: Array<UInt8>, declaredTool: Qwen35DeclaredTool?
    ) -> Dictionary<String, Qwen35ParsedToolValue> {
        var remainingParameters = Qwen35ByteText.trimmed(parameterContent);
        var parsedArguments: Dictionary<String, Qwen35ParsedToolValue> = [:];
        while !remainingParameters.isEmpty {
            remainingParameters = Qwen35ByteText.trimmed(remainingParameters);
            guard
                let parameterWithName = stripQwenParameterOpen(remainingParameters)
            else {
                guard let parameterOpenOffset = nextParameterOpenOffset(remainingParameters) else {
                    break;
                }
                remainingParameters = Array(remainingParameters[parameterOpenOffset...]);
                continue;
            }
            guard
                let parameterNameEnd = Qwen35ByteText.firstOffset(
                    of: [UInt8(ascii: ">")], in: parameterWithName)
            else {
                remainingParameters = skipUnclosedParameterName(parameterWithName);
                continue;
            }
            let parameterName = Qwen35ByteText.trimmed(
                Array(parameterWithName[0..<parameterNameEnd]));
            if parameterName.isEmpty {
                remainingParameters = Array(parameterWithName[(parameterNameEnd + 1)...]);
                continue;
            }
            let parameterWithValue = Array(parameterWithName[(parameterNameEnd + 1)...]);
            let (parameterValue, afterParameter) = splitParameterValue(parameterWithValue);
            let parameterNameKey = Qwen35ByteText.text(parameterName[...]);
            let parameterSchema = declaredTool?.parameterSchemas[parameterNameKey];
            let parsedParameterValue: Qwen35ParsedToolValue;
            if let parameterSchema {
                parsedParameterValue = parseParameterValue(parameterValue, parameterSchema);
            } else {
                parsedParameterValue = parseUntypedParameterValue(parameterValue);
            }
            parsedArguments[parameterNameKey] = parsedParameterValue;
            remainingParameters = Qwen35ByteText.trimmed(afterParameter);
        }
        return parsedArguments;
    }

    private static func stripQwenParameterOpen(_ parameterContent: Array<UInt8>) -> Array<UInt8>? {
        let canonicalOpen = Qwen35ByteText.bytes(Qwen35OutputParserMarkers.bareParameterStart);
        return Qwen35ByteText.stripPrefix(parameterContent, canonicalOpen)
            ?? Qwen35ByteText.stripPrefix(parameterContent, Qwen35ByteText.bytes("parameter="));
    }

    private static let parameterClose = Qwen35ByteText.bytes("</parameter>");

    private static func nextParameterOpenOffset(_ parameterContent: Array<UInt8>) -> Int? {
        let canonicalOpen = Qwen35ByteText.bytes(Qwen35OutputParserMarkers.bareParameterStart);
        let salvagedOpen = Qwen35ByteText.bytes("parameter=");
        let canonicalOffset = Qwen35ByteText.firstOffset(of: canonicalOpen, in: parameterContent);
        let salvagedOffset = Qwen35ByteText.firstOffset(of: salvagedOpen, in: parameterContent);
        switch (canonicalOffset, salvagedOffset) {
        case let (canonical?, salvaged?):
            return min(canonical, salvaged);
        case let (canonical?, nil):
            return canonical;
        case let (nil, salvaged?):
            return salvaged;
        default:
            return nil;
        }
    }

    private static func skipUnclosedParameterName(_ parameterWithName: Array<UInt8>) -> Array<UInt8> {
        if let parameterEnd = Qwen35ByteText.firstOffset(of: parameterClose, in: parameterWithName) {
            return Array(parameterWithName[(parameterEnd + parameterClose.count)...]);
        }
        if let nextParameterOffset = nextParameterOpenOffset(parameterWithName),
            nextParameterOffset > 0
        {
            return Array(parameterWithName[nextParameterOffset...]);
        }
        return [];
    }

    private static func splitParameterValue(
        _ parameterWithValue: Array<UInt8>
    ) -> (parameterValue: Array<UInt8>, afterParameter: Array<UInt8>) {
        guard
            let parameterEnd = Qwen35ByteText.firstOffset(of: parameterClose, in: parameterWithValue)
        else {
            return (trimOneBoundaryNewline(parameterWithValue), []);
        }
        return (
            trimOneBoundaryNewline(Array(parameterWithValue[0..<parameterEnd])),
            Array(parameterWithValue[(parameterEnd + parameterClose.count)...])
        );
    }

    private static func parseParameterValue(
        _ parameterValue: Array<UInt8>, _ parameterSchema: Qwen35DeclaredParameterSchema
    ) -> Qwen35ParsedToolValue {
        let valueText = Qwen35ByteText.text(parameterValue[...]);
        if parameterSchema.isNullable && valueText == "null" {
            return .null;
        }
        guard let parameterType = parameterSchema.parameterType else {
            return parseUntypedParameterValue(parameterValue);
        }
        switch parameterType {
        case "string":
            return .string(valueText);
        case "boolean":
            if valueText == "true" {
                return .boolean(true);
            }
            if valueText == "false" {
                return .boolean(false);
            }
            return .string(valueText);
        case "integer":
            if let integerValue = Int64(valueText) {
                return .integer(integerValue);
            }
            return .string(valueText);
        case "number":
            if let numberValue = Double(valueText), numberValue.isFinite {
                return .number(numberValue);
            }
            return .string(valueText);
        case "array", "object":
            guard let jsonValue = parseJsonParameter(parameterValue) else {
                return .string(valueText);
            }
            switch (jsonValue, parameterType) {
            case (.array, "array"), (.object, "object"):
                return jsonValue;
            default:
                return .string(valueText);
            }
        default:
            return .string(valueText);
        }
    }

    /// Untyped parameters accept any non-string JSON literal; a quoted JSON
    /// string or unparseable text stays the raw parameter text.
    private static func parseUntypedParameterValue(
        _ parameterValue: Array<UInt8>
    ) -> Qwen35ParsedToolValue {
        guard let jsonValue = parseJsonParameter(parameterValue) else {
            return .string(Qwen35ByteText.text(parameterValue[...]));
        }
        if case let .string(stringValue) = jsonValue {
            return .string(stringValue);
        }
        return jsonValue;
    }

    private static func parseJsonParameter(
        _ parameterValue: Array<UInt8>
    ) -> Qwen35ParsedToolValue? {
        guard let wireValue = try? JsonWireParser.parseDocument(documentBytes: Data(parameterValue))
        else {
            return nil;
        }
        return Qwen35ToolSchemaParser.parsedValue(from: wireValue);
    }

    private static func parsedValue(from wireValue: JsonWireValue) -> Qwen35ParsedToolValue? {
        switch wireValue {
        case .null:
            return .null;
        case let .boolean(booleanValue):
            return .boolean(booleanValue);
        case let .unsignedInteger(unsignedValue):
            return Int64(exactly: unsignedValue).map { Qwen35ParsedToolValue.integer($0) }
                ?? .number(Double(unsignedValue));
        case let .signedInteger(signedValue):
            return .integer(signedValue);
        case let .double(doubleValue):
            return doubleValue.isFinite ? .number(doubleValue) : nil;
        case let .float32(floatValue):
            return floatValue.isFinite ? .number(Double(floatValue)) : nil;
        case let .string(stringValue):
            return .string(stringValue);
        case let .array(entryValues):
            let parsedEntries = entryValues.compactMap { parsedValue(from: $0) };
            return parsedEntries.count == entryValues.count ? .array(parsedEntries) : nil;
        case let .object(wireObject):
            var parsedEntries: Array<(key: String, value: Qwen35ParsedToolValue)> = [];
            for entry in wireObject.entries {
                guard let parsedEntryValue = parsedValue(from: entry.value) else {
                    return nil;
                }
                parsedEntries.append((key: entry.key, value: parsedEntryValue));
            }
            return .object(parsedEntries);
        }
    }

    private static func trimOneBoundaryNewline(_ parameterValue: Array<UInt8>) -> Array<UInt8> {
        var trimmedValue = parameterValue;
        if trimmedValue.first == UInt8(ascii: "\n") {
            trimmedValue = Array(trimmedValue[1...]);
        }
        if trimmedValue.last == UInt8(ascii: "\n") {
            trimmedValue = Array(trimmedValue[0..<(trimmedValue.count - 1)]);
        }
        return trimmedValue;
    }
}
