import Foundation;

/// One callable JSON-schema function supplied to the model.
public struct ChatToolDefinition: Equatable {
    /// The declared function name.
    public let name: String;
    /// Optional caller-provided explanation.
    public let description: String?;
    /// Canonical JSON Schema serialized by the supervisor after bounded validation.
    public let parametersJson: String;

    public init(name: String, description: String?, parametersJson: String) {
        self.name = name;
        self.description = description;
        self.parametersJson = parametersJson;
    }

    internal static let wireFieldNames: Array<String> = ["name", "description", "parameters_json"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "name", value: .string(self.name));
        wireObject.appendEntry(key: "description", value: ChatToolDefinition.optionalStringWireValue(self.description));
        wireObject.appendEntry(key: "parameters_json", value: .string(self.parametersJson));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatToolDefinition {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedDefinition = ChatToolDefinition(
            name: try wireObject.decodeString(fieldName: "name"),
            description: try wireObject.decodeOptionalString(fieldName: "description"),
            parametersJson: try wireObject.decodeString(fieldName: "parameters_json"));
        try wireObject.rejectUnknownFields(allowedFieldNames: ChatToolDefinition.wireFieldNames);
        return parsedDefinition;
    }

    private static func optionalStringWireValue(_ rawValue: String?) -> JsonWireValue {
        guard let unwrappedValue = rawValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }
}

/// The caller's tool-selection policy after public validation.
/// Wire shape is internally tagged with `kind` in snake_case.
public enum ChatToolChoice: Equatable {
    /// The model decides whether to call a function.
    case auto;
    /// The model must not call a function.
    case none;
    /// The model must call one declared function.
    case required;
    /// The model must call this declared function.
    case function(name: String);

    private static let expectedVariantNames: Array<String> = ["auto", "none", "required", "function"];

    internal func wireValue() -> JsonWireValue {
        switch self {
        case .auto:
            return .object(ChatToolChoice.tagOnlyWireObject(variantName: "auto"));
        case .none:
            return .object(ChatToolChoice.tagOnlyWireObject(variantName: "none"));
        case .required:
            return .object(ChatToolChoice.tagOnlyWireObject(variantName: "required"));
        case let .function(name):
            var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
            wireObject.appendEntry(key: "kind", value: .string("function"));
            wireObject.appendEntry(key: "name", value: .string(name));
            return .object(wireObject);
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatToolChoice {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: ChatToolChoice.expectedVariantNames) {
        case "auto":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .auto;
        case "none":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .none;
        case "required":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .required;
        default:
            let parsedChoice = ChatToolChoice.function(name: try wireObject.decodeString(fieldName: "name"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["name"]);
            return parsedChoice;
        }
    }

    private static func tagOnlyWireObject(variantName: String) -> JsonWireObject {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "kind", value: .string(variantName));
        return wireObject;
    }
}
