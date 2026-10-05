import Foundation;

/// Worker-enforced structured generation. The supervisor never sends a variant
/// the worker cannot mask at sample time.
/// Wire shape is internally tagged with `kind` in snake_case.
public enum StructuredGenerationConstraint: Equatable {
    /// Bounds the raw JSON schema text one chat request may carry. Schema DFA
    /// compilation cost grows with schema size, so a fixed ceiling keeps worker
    /// compilation predictable.
    public static let maximumChatSchemaJsonBytes: Int = 65_536;

    case jsonObject;
    case jsonSchema(schemaJson: String);
    case choice(choices: Array<String>);
    /// The complete visible answer must match this regular expression.
    case regex(pattern: String);

    private static let expectedVariantNames: Array<String> = ["json_object", "json_schema", "choice", "regex"];

    /// Wraps a caller regex so its DFA matches only from the first answer byte.
    /// The end stays anchored to end of text so completion and viability share
    /// one automaton. The public boundary and the worker compile this identical shape.
    public static func structuredRegexDfaPattern(regexPattern: String) -> String {
        return "\\A(?:\(regexPattern))\\z";
    }

    internal func wireValue() -> JsonWireValue {
        switch self {
        case .jsonObject:
            return .object(StructuredGenerationConstraint.tagOnlyWireObject(variantName: "json_object"));
        case let .jsonSchema(schemaJson):
            var wireObject = StructuredGenerationConstraint.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("json_schema"));
            wireObject.appendEntry(key: "schema_json", value: .string(schemaJson));
            return .object(wireObject);
        case let .choice(choices):
            var wireObject = StructuredGenerationConstraint.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("choice"));
            wireObject.appendEntry(key: "choices", value: StructuredGenerationConstraint.stringArrayWireValue(choices));
            return .object(wireObject);
        case let .regex(pattern):
            var wireObject = StructuredGenerationConstraint.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("regex"));
            wireObject.appendEntry(key: "pattern", value: .string(pattern));
            return .object(wireObject);
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> StructuredGenerationConstraint {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: StructuredGenerationConstraint.expectedVariantNames) {
        case "json_object":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .jsonObject;
        case "json_schema":
            let parsedConstraint = StructuredGenerationConstraint.jsonSchema(schemaJson: try wireObject.decodeString(fieldName: "schema_json"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["schema_json"]);
            return parsedConstraint;
        case "choice":
            let parsedConstraint = StructuredGenerationConstraint.choice(
                choices: try wireObject.decodeArray(fieldName: "choices", mappedElement: { (elementWireValue: JsonWireValue) throws -> String in
                    try JsonWireValue.extractString(elementWireValue)
                }));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["choices"]);
            return parsedConstraint;
        default:
            let parsedConstraint = StructuredGenerationConstraint.regex(pattern: try wireObject.decodeString(fieldName: "pattern"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["pattern"]);
            return parsedConstraint;
        }
    }

    private static func emptyWireObject() -> JsonWireObject {
        return JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
    }

    private static func tagOnlyWireObject(variantName: String) -> JsonWireObject {
        var wireObject = StructuredGenerationConstraint.emptyWireObject();
        wireObject.appendEntry(key: "kind", value: .string(variantName));
        return wireObject;
    }

    private static func stringArrayWireValue(_ textValues: Array<String>) -> JsonWireValue {
        return JsonWireValue.mappedArray(textValues, mappedWireValue: { (textValue: String) -> JsonWireValue in .string(textValue) });
    }
}
