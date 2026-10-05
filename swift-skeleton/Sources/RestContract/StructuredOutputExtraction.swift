import Foundation;
import IpcProtocol;

/// Best-effort JSON extraction from model text, port of the free functions at
/// the bottom of crates/rest-contract/src/openai_response_format.rs. The Rust
/// helpers deserialize `serde_json::Value` directly from text, whose object
/// semantics differ from struct deserialization: duplicate keys are
/// last-wins, objects are stored as BTreeMaps (byte-sorted), and the leading
/// value parser tolerates trailing content after the first JSON value. The
/// scanner here reproduces those Value semantics over the shared wire tree.
extension OpenAiStructuredOutput {

    /// Parses a JSON value from model text without coercing fields or filling defaults.
    public static func extract_json_value_from_text(_ visibleText: String) -> JsonWireValue? {
        let trimmedText: String = visibleText.trimmingCharacters(in: .whitespacesAndNewlines);
        if let wholeDocument: JsonWireValue = StructuredOutputTextJson.parseWholeDocument(trimmedText) {
            return wholeDocument;
        }
        if let fencedPayload: String = StructuredOutputTextJson.fencedJsonPayload(from: trimmedText),
            let fencedDocument: JsonWireValue = StructuredOutputTextJson.parseWholeDocument(fencedPayload) {
            return fencedDocument;
        }
        return StructuredOutputTextJson.parseLeadingValue(trimmedText);
    }

    /// Compact JSON text when extraction succeeds.
    public static func compact_extracted_json_text(_ visibleText: String) -> String? {
        guard let parsedJson: JsonWireValue = OpenAiStructuredOutput.extract_json_value_from_text(visibleText) else {
            return nil;
        }
        do {
            return try ChatToolValidation.canonicalSerializedText(parsedJson);
        } catch {
            return nil;
        }
    }
}

/// Recursive-descent scanner with serde_json `Value` semantics.
enum StructuredOutputTextJson {

    private static let maximumRecursionDepth: Int = 128;

    static func parseWholeDocument(_ documentText: String) -> JsonWireValue? {
        var scanner: TextJsonScanner = TextJsonScanner(documentBytes: Array(documentText.utf8));
        scanner.skipWhitespace();
        guard let parsedValue: JsonWireValue = scanner.parseValue(currentDepth: 0) else {
            return nil;
        }
        scanner.skipWhitespace();
        guard scanner.isAtEnd else {
            return nil;
        }
        return parsedValue;
    }

    static func parseLeadingValue(_ documentText: String) -> JsonWireValue? {
        guard let jsonStartIndex: Int = firstValueStartIndex(in: Array(documentText.utf8)) else {
            return nil;
        }
        var scanner: TextJsonScanner = TextJsonScanner(documentBytes: Array(documentText.utf8), startIndex: jsonStartIndex);
        return scanner.parseValue(currentDepth: 0);
    }

    static func fencedJsonPayload(from visibleText: String) -> String? {
        let textUnits: Array<Character> = Array(visibleText);
        guard let fenceStartOffset: Int = firstFenceOffset(in: textUnits) else {
            return nil;
        }
        var cursorOffset: Int = fenceStartOffset + 3;
        let unitCount: Int = textUnits.count;
        if cursorOffset + 4 <= unitCount, visibleTextHasJsonLanguageTag(textUnits, at: cursorOffset) {
            cursorOffset = cursorOffset + 4;
        }
        if cursorOffset + 2 <= unitCount, textUnits[cursorOffset] == "\r", textUnits[cursorOffset + 1] == "\n" {
            cursorOffset = cursorOffset + 2;
        } else if cursorOffset < unitCount, textUnits[cursorOffset] == "\n" {
            cursorOffset = cursorOffset + 1;
        } else {
            return nil;
        }
        var fenceEndOffset: Int = -1;
        var scanOffset: Int = cursorOffset;
        while scanOffset + 3 <= unitCount {
            if textUnits[scanOffset] == "`", textUnits[scanOffset + 1] == "`", textUnits[scanOffset + 2] == "`" {
                fenceEndOffset = scanOffset;
                break;
            }
            scanOffset = scanOffset + 1;
        }
        guard fenceEndOffset >= 0 else {
            return nil;
        }
        let payloadText: String = String(textUnits[cursorOffset..<fenceEndOffset]);
        return payloadText.trimmingCharacters(in: .whitespacesAndNewlines);
    }

    private static func firstFenceOffset(in textUnits: Array<Character>) -> Int? {
        var scanOffset: Int = 0;
        while scanOffset + 3 <= textUnits.count {
            if textUnits[scanOffset] == "`", textUnits[scanOffset + 1] == "`", textUnits[scanOffset + 2] == "`" {
                return scanOffset;
            }
            scanOffset = scanOffset + 1;
        }
        return nil;
    }

    private static func visibleTextHasJsonLanguageTag(_ textUnits: Array<Character>, at offset: Int) -> Bool {
        let languageTag: Array<Character> = Array("json");
        for (tagIndex, tagCharacter) in languageTag.enumerated() {
            if textUnits[offset + tagIndex] != tagCharacter {
                return false;
            }
        }
        return true;
    }

    private static func firstValueStartIndex(in documentBytes: Array<UInt8>) -> Int? {
        for (byteIndex, documentByte) in documentBytes.enumerated() {
            if documentByte == UInt8(ascii: "{") || documentByte == UInt8(ascii: "[") {
                return byteIndex;
            }
        }
        return nil;
    }
}

/// Byte-cursor scanner; `nil` at every parse site mirrors serde_json's
/// `.ok()` discard path — malformed text simply fails extraction.
private struct TextJsonScanner {
    private let documentBytes: Array<UInt8>;
    private var cursorIndex: Int;

    init(documentBytes: Array<UInt8>, startIndex: Int = 0) {
        self.documentBytes = documentBytes;
        self.cursorIndex = startIndex;
    }

    var isAtEnd: Bool {
        return self.cursorIndex >= self.documentBytes.count;
    }

    mutating func skipWhitespace() {
        while self.cursorIndex < self.documentBytes.count {
            let currentByte: UInt8 = self.documentBytes[self.cursorIndex];
            if currentByte == UInt8(ascii: " ") || currentByte == UInt8(ascii: "\t")
                || currentByte == UInt8(ascii: "\n") || currentByte == UInt8(ascii: "\r") {
                self.cursorIndex = self.cursorIndex + 1;
                continue;
            }
            return;
        }
    }

    mutating func parseValue(currentDepth: Int) -> JsonWireValue? {
        guard currentDepth <= StructuredOutputTextJsonMaximumDepthMarker.maximumDepth else {
            return nil;
        }
        self.skipWhitespace();
        guard self.isAtEnd == false else {
            return nil;
        }
        let leadingByte: UInt8 = self.documentBytes[self.cursorIndex];
        switch leadingByte {
        case UInt8(ascii: "{"):
            return self.parseObject(currentDepth: currentDepth);
        case UInt8(ascii: "["):
            return self.parseArray(currentDepth: currentDepth);
        case UInt8(ascii: "\""):
            guard let decodedText: String = self.parseString() else {
                return nil;
            }
            return .string(decodedText);
        case UInt8(ascii: "t"):
            return self.parseLiteral("true", wireValue: .boolean(true));
        case UInt8(ascii: "f"):
            return self.parseLiteral("false", wireValue: .boolean(false));
        case UInt8(ascii: "n"):
            return self.parseLiteral("null", wireValue: .null);
        default:
            return self.parseNumber();
        }
    }

    private mutating func parseObject(currentDepth: Int) -> JsonWireValue? {
        self.cursorIndex = self.cursorIndex + 1;
        var objectEntries: Array<(key: String, value: JsonWireValue)> = Array();
        self.skipWhitespace();
        if self.cursorIndex < self.documentBytes.count, self.documentBytes[self.cursorIndex] == UInt8(ascii: "}") {
            self.cursorIndex = self.cursorIndex + 1;
            return .object(Self.sortedObject(objectEntries));
        }
        while self.isAtEnd == false {
            self.skipWhitespace();
            guard self.documentBytes[self.cursorIndex] == UInt8(ascii: "\"") else {
                return nil;
            }
            guard let entryKey: String = self.parseString() else {
                return nil;
            }
            self.skipWhitespace();
            guard self.cursorIndex < self.documentBytes.count, self.documentBytes[self.cursorIndex] == UInt8(ascii: ":") else {
                return nil;
            }
            self.cursorIndex = self.cursorIndex + 1;
            guard let entryValue: JsonWireValue = self.parseValue(currentDepth: currentDepth + 1) else {
                return nil;
            }
            // serde_json Value maps are last-wins on duplicate keys.
            if let existingIndex: Int = objectEntries.firstIndex(where: { (entry: (key: String, value: JsonWireValue)) -> Bool in entry.key == entryKey }) {
                objectEntries[existingIndex].value = entryValue;
            } else {
                objectEntries.append((entryKey, entryValue));
            }
            self.skipWhitespace();
            guard self.cursorIndex < self.documentBytes.count else {
                return nil;
            }
            let separatorByte: UInt8 = self.documentBytes[self.cursorIndex];
            if separatorByte == UInt8(ascii: ",") {
                self.cursorIndex = self.cursorIndex + 1;
                continue;
            }
            if separatorByte == UInt8(ascii: "}") {
                self.cursorIndex = self.cursorIndex + 1;
                return .object(Self.sortedObject(objectEntries));
            }
            return nil;
        }
        return nil;
    }

    private mutating func parseArray(currentDepth: Int) -> JsonWireValue? {
        self.cursorIndex = self.cursorIndex + 1;
        var elementValues: Array<JsonWireValue> = Array();
        self.skipWhitespace();
        if self.cursorIndex < self.documentBytes.count, self.documentBytes[self.cursorIndex] == UInt8(ascii: "]") {
            self.cursorIndex = self.cursorIndex + 1;
            return .array(elementValues);
        }
        while self.isAtEnd == false {
            guard let elementValue: JsonWireValue = self.parseValue(currentDepth: currentDepth + 1) else {
                return nil;
            }
            elementValues.append(elementValue);
            self.skipWhitespace();
            guard self.cursorIndex < self.documentBytes.count else {
                return nil;
            }
            let separatorByte: UInt8 = self.documentBytes[self.cursorIndex];
            if separatorByte == UInt8(ascii: ",") {
                self.cursorIndex = self.cursorIndex + 1;
                continue;
            }
            if separatorByte == UInt8(ascii: "]") {
                self.cursorIndex = self.cursorIndex + 1;
                return .array(elementValues);
            }
            return nil;
        }
        return nil;
    }

    private mutating func parseString() -> String? {
        self.cursorIndex = self.cursorIndex + 1;
        var decodedBytes: Array<UInt8> = Array();
        while self.cursorIndex < self.documentBytes.count {
            let currentByte: UInt8 = self.documentBytes[self.cursorIndex];
            if currentByte == UInt8(ascii: "\"") {
                self.cursorIndex = self.cursorIndex + 1;
                guard let decodedText: String = String(bytes: decodedBytes, encoding: String.Encoding.utf8) else {
                    return nil;
                }
                return decodedText;
            }
            if currentByte == UInt8(ascii: "\\") {
                self.cursorIndex = self.cursorIndex + 1;
                guard self.cursorIndex < self.documentBytes.count else {
                    return nil;
                }
                let escapeByte: UInt8 = self.documentBytes[self.cursorIndex];
                switch escapeByte {
                case UInt8(ascii: "\""): decodedBytes.append(UInt8(ascii: "\""));
                case UInt8(ascii: "\\"): decodedBytes.append(UInt8(ascii: "\\"));
                case UInt8(ascii: "/"): decodedBytes.append(UInt8(ascii: "/"));
                case UInt8(ascii: "b"): decodedBytes.append(0x08);
                case UInt8(ascii: "f"): decodedBytes.append(0x0C);
                case UInt8(ascii: "n"): decodedBytes.append(0x0A);
                case UInt8(ascii: "r"): decodedBytes.append(0x0D);
                case UInt8(ascii: "t"): decodedBytes.append(0x09);
                case UInt8(ascii: "u"):
                    guard let escapedScalar: UInt16 = self.parseHexQuad() else {
                        return nil;
                    }
                    var combinedScalar: UInt32 = UInt32(escapedScalar);
                    if escapedScalar >= 0xD800 && escapedScalar <= 0xDBFF {
                        guard self.cursorIndex + 1 < self.documentBytes.count,
                            self.documentBytes[self.cursorIndex] == UInt8(ascii: "\\"),
                            self.documentBytes[self.cursorIndex + 1] == UInt8(ascii: "u") else {
                            return nil;
                        }
                        self.cursorIndex = self.cursorIndex + 2;
                        guard let lowScalar: UInt16 = self.parseHexQuad(), lowScalar >= 0xDC00, lowScalar <= 0xDFFF else {
                            return nil;
                        }
                        combinedScalar = 0x10000 + ((UInt32(escapedScalar) - 0xD800) << 10) + (UInt32(lowScalar) - 0xDC00);
                    } else if escapedScalar >= 0xDC00 && escapedScalar <= 0xDFFF {
                        return nil;
                    }
                    guard let scalarCharacter: Character = Unicode.Scalar(combinedScalar).map({ (scalar: Unicode.Scalar) -> Character in Character(scalar) }) else {
                        return nil;
                    }
                    for utf8Byte: UInt8 in Array(String(scalarCharacter).utf8) {
                        decodedBytes.append(utf8Byte);
                    }
                default:
                    return nil;
                }
                self.cursorIndex = self.cursorIndex + 1;
                continue;
            }
            decodedBytes.append(currentByte);
            self.cursorIndex = self.cursorIndex + 1;
        }
        return nil;
    }

    private mutating func parseHexQuad() -> UInt16? {
        guard self.cursorIndex + 4 < self.documentBytes.count else {
            return nil;
        }
        var quadValue: UInt16 = 0;
        for hexOffset in 1...4 {
            let hexByte: UInt8 = self.documentBytes[self.cursorIndex + hexOffset];
            let digitValue: UInt16;
            switch hexByte {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): digitValue = UInt16(hexByte - UInt8(ascii: "0"));
            case UInt8(ascii: "a")...UInt8(ascii: "f"): digitValue = UInt16(hexByte - UInt8(ascii: "a") + 10);
            case UInt8(ascii: "A")...UInt8(ascii: "F"): digitValue = UInt16(hexByte - UInt8(ascii: "A") + 10);
            default: return nil;
            }
            quadValue = (quadValue << 4) | digitValue;
        }
        self.cursorIndex = self.cursorIndex + 4;
        return quadValue;
    }

    private mutating func parseLiteral(_ literalText: String, wireValue: JsonWireValue) -> JsonWireValue? {
        let literalBytes: Array<UInt8> = Array(literalText.utf8);
        guard self.cursorIndex + literalBytes.count <= self.documentBytes.count else {
            return nil;
        }
        for (literalOffset, literalByte) in literalBytes.enumerated() {
            if self.documentBytes[self.cursorIndex + literalOffset] != literalByte {
                return nil;
            }
        }
        self.cursorIndex = self.cursorIndex + literalBytes.count;
        return wireValue;
    }

    private mutating func parseNumber() -> JsonWireValue? {
        let numberStartIndex: Int = self.cursorIndex;
        var isFloat: Bool = false;
        if self.cursorIndex < self.documentBytes.count, self.documentBytes[self.cursorIndex] == UInt8(ascii: "-") {
            self.cursorIndex = self.cursorIndex + 1;
        }
        while self.cursorIndex < self.documentBytes.count {
            let currentByte: UInt8 = self.documentBytes[self.cursorIndex];
            if currentByte >= UInt8(ascii: "0") && currentByte <= UInt8(ascii: "9") {
                self.cursorIndex = self.cursorIndex + 1;
                continue;
            }
            if currentByte == UInt8(ascii: ".") || currentByte == UInt8(ascii: "e") || currentByte == UInt8(ascii: "E") {
                isFloat = true;
                self.cursorIndex = self.cursorIndex + 1;
                continue;
            }
            if currentByte == UInt8(ascii: "+") || currentByte == UInt8(ascii: "-") {
                self.cursorIndex = self.cursorIndex + 1;
                continue;
            }
            break;
        }
        guard self.cursorIndex > numberStartIndex, let numberText: String = String(bytes: self.documentBytes[numberStartIndex..<self.cursorIndex], encoding: String.Encoding.utf8) else {
            return nil;
        }
        if isFloat == false {
            if let unsignedValue: UInt64 = UInt64(numberText) {
                return .unsignedInteger(unsignedValue);
            }
            if let signedValue: Int64 = Int64(numberText) {
                return .signedInteger(signedValue);
            }
        }
        guard let doubleValue: Double = Double(numberText), doubleValue.isFinite else {
            return nil;
        }
        return .double(doubleValue);
    }

    /// serde_json Value objects iterate in BTreeMap (byte-sorted) order, which
    /// also makes equality and re-serialization match the Rust shapes.
    private static func sortedObject(_ entries: Array<(key: String, value: JsonWireValue)>) -> JsonWireObject {
        var sortedEntries: Array<(key: String, value: JsonWireValue)> = entries;
        sortedEntries.sort { (leftEntry: (key: String, value: JsonWireValue), rightEntry: (key: String, value: JsonWireValue)) -> Bool in
            return Array(leftEntry.key.utf8).lexicographicallyPrecedes(Array(rightEntry.key.utf8));
        };
        return JsonWireObject(entries: sortedEntries);
    }
}

/// Depth-cap holder kept outside the scanner so the parser reads like the
/// serde_json recursion limit it mirrors.
private enum StructuredOutputTextJsonMaximumDepthMarker {
    static let maximumDepth: Int = 128;
}
