import Foundation;

/// Compact serde_json-shaped serializer for JsonWireValue trees. Key order is
/// insertion order, solidus is never escaped, non-finite floats are rejected,
/// and non-ASCII text passes through as UTF-8 exactly like serde_json.
internal struct JsonWireWriter {
    private var outputText: String = "";

    internal init() {
    }

    internal var serializedText: String {
        return self.outputText;
    }

    internal var serializedUtf8Bytes: Data {
        return Data(self.outputText.utf8);
    }

    internal mutating func appendValue(_ wireValue: JsonWireValue) throws {
        switch wireValue {
        case .null: self.outputText = self.outputText + "null";
        case let .boolean(booleanValue): self.outputText = self.outputText + (booleanValue ? "true" : "false");
        case let .unsignedInteger(numericValue): self.outputText = self.outputText + String(numericValue);
        case let .signedInteger(numericValue): self.outputText = self.outputText + String(numericValue);
        case let .double(numericValue): try self.appendFiniteDouble(numericValue);
        case let .float32(numericValue): try self.appendFiniteFloat32(numericValue);
        case let .string(textValue): self.appendStringValue(textValue);
        case let .array(arrayValue): try self.appendArrayValue(arrayValue);
        case let .object(objectValue): try self.appendObjectValue(objectValue);
        }
    }

    internal mutating func appendStringValue(_ textValue: String) {
        var escapedText: String = "\"";
        for character in textValue.unicodeScalars {
            switch character {
            case "\"": escapedText = escapedText + "\\\"";
            case "\\": escapedText = escapedText + "\\\\";
            case "\n": escapedText = escapedText + "\\n";
            case "\r": escapedText = escapedText + "\\r";
            case "\t": escapedText = escapedText + "\\t";
            case "\u{08}": escapedText = escapedText + "\\b";
            case "\u{0C}": escapedText = escapedText + "\\f";
            default:
                if character.value < 0x20 {
                    escapedText = escapedText + String(format: "\\u%04x", character.value);
                } else {
                    escapedText.unicodeScalars.append(character);
                }
            }
        }
        self.outputText = self.outputText + escapedText + "\"";
    }

    private mutating func appendArrayValue(_ arrayValue: Array<JsonWireValue>) throws {
        self.outputText = self.outputText + "[";
        for (elementIndex, elementValue) in arrayValue.enumerated() {
            if elementIndex > 0 {
                self.outputText = self.outputText + ",";
            }
            try self.appendValue(elementValue);
        }
        self.outputText = self.outputText + "]";
    }

    private mutating func appendObjectValue(_ objectValue: JsonWireObject) throws {
        self.outputText = self.outputText + "{";
        for (entryIndex, entry) in objectValue.entries.enumerated() {
            if entryIndex > 0 {
                self.outputText = self.outputText + ",";
            }
            self.appendStringValue(entry.key);
            self.outputText = self.outputText + ":";
            try self.appendValue(entry.value);
        }
        self.outputText = self.outputText + "}";
    }

    /// serde_json rejects non-finite doubles when serializing.
    private mutating func appendFiniteDouble(_ numericValue: Double) throws {
        guard numericValue.isFinite else {
            throw JsonWireProblem.malformedDocument(problem: "float value out of range");
        }
        self.outputText = self.outputText + String(numericValue);
    }

    private mutating func appendFiniteFloat32(_ numericValue: Float) throws {
        guard numericValue.isFinite else {
            throw JsonWireProblem.malformedDocument(problem: "float value out of range");
        }
        self.outputText = self.outputText + String(numericValue);
    }
}
