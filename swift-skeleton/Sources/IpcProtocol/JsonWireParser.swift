import Foundation;

/// Strict recursive-descent JSON parser over UTF-8 bytes producing ordered
/// JsonWireValue trees. Rejects duplicate object keys the way serde struct
/// deserializers do and enforces the same 128-level recursion cap as serde_json.
public enum JsonWireParser {

    private static let maximumRecursionDepth: Int = 128;

    public static func parseDocument(documentBytes: Data) throws -> JsonWireValue {
        var documentScanner: DocumentScanner = DocumentScanner(documentBytes: Array(documentBytes));
        documentScanner.skipWhitespaceBytes();
        guard documentScanner.isAtEnd == false else {
            throw JsonWireProblem.malformedDocument(problem: "EOF while parsing a value");
        }
        let parsedRootValue = try documentScanner.parseValueAtCursor(currentDepth: 0);
        documentScanner.skipWhitespaceBytes();
        if documentScanner.isAtEnd == false {
            throw JsonWireProblem.malformedDocument(problem: "trailing characters");
        }
        return parsedRootValue;
    }

    private struct DocumentScanner {
        private let documentBytes: Array<UInt8>;
        private var cursorIndex: Int = 0;

        internal init(documentBytes: Array<UInt8>) {
            self.documentBytes = documentBytes;
        }

        internal var isAtEnd: Bool {
            return self.cursorIndex >= self.documentBytes.count;
        }

        internal mutating func skipWhitespaceBytes() {
            while self.isAtEnd == false {
                let currentByte: UInt8 = self.documentBytes[self.cursorIndex];
                if currentByte == 0x20 || currentByte == 0x09 || currentByte == 0x0A || currentByte == 0x0D {
                    self.cursorIndex = self.cursorIndex + 1;
                } else {
                    return;
                }
            }
        }

        internal mutating func parseValueAtCursor(currentDepth: Int) throws -> JsonWireValue {
            if currentDepth > JsonWireParser.maximumRecursionDepth {
                throw JsonWireProblem.recursionLimitExceeded;
            }
            self.skipWhitespaceBytes();
            guard self.isAtEnd == false else {
                throw JsonWireProblem.malformedDocument(problem: "EOF while parsing a value");
            }
            switch self.documentBytes[self.cursorIndex] {
            case UInt8(ascii: "{"): return try self.parseObjectAtCursor(currentDepth: currentDepth);
            case UInt8(ascii: "["): return try self.parseArrayAtCursor(currentDepth: currentDepth);
            case UInt8(ascii: "\""): return .string(try self.parseStringAtCursor());
            case UInt8(ascii: "t"): try self.expectLiteral(literalBytes: Array("true".utf8)); return .boolean(true);
            case UInt8(ascii: "f"): try self.expectLiteral(literalBytes: Array("false".utf8)); return .boolean(false);
            case UInt8(ascii: "n"): try self.expectLiteral(literalBytes: Array("null".utf8)); return .null;
            case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"): return try self.parseNumberAtCursor();
            default: throw JsonWireProblem.malformedDocument(problem: "expected value");
            }
        }

        internal mutating func parseObjectAtCursor(currentDepth: Int) throws -> JsonWireValue {
            self.cursorIndex = self.cursorIndex + 1;
            var parsedEntries: Array<(key: String, value: JsonWireValue)> = Array<(key: String, value: JsonWireValue)>();
            var seenKeyNames: Set<String> = Set<String>();
            self.skipWhitespaceBytes();
            if self.isAtEnd == false && self.documentBytes[self.cursorIndex] == UInt8(ascii: "}") {
                self.cursorIndex = self.cursorIndex + 1;
                return .object(JsonWireObject(entries: parsedEntries));
            }
            while true {
                self.skipWhitespaceBytes();
                guard self.isAtEnd == false && self.documentBytes[self.cursorIndex] == UInt8(ascii: "\"") else {
                    throw JsonWireProblem.malformedDocument(problem: "key must be a string");
                }
                let entryKey: String = try self.parseStringAtCursor();
                if seenKeyNames.contains(entryKey) {
                    throw JsonWireProblem.duplicateField(fieldName: entryKey);
                }
                seenKeyNames.insert(entryKey);
                self.skipWhitespaceBytes();
                guard self.isAtEnd == false && self.documentBytes[self.cursorIndex] == UInt8(ascii: ":") else {
                    throw JsonWireProblem.malformedDocument(problem: "expected `:`");
                }
                self.cursorIndex = self.cursorIndex + 1;
                let entryValue: JsonWireValue = try self.parseValueAtCursor(currentDepth: currentDepth + 1);
                parsedEntries.append((entryKey, entryValue));
                self.skipWhitespaceBytes();
                guard self.isAtEnd == false else {
                    throw JsonWireProblem.malformedDocument(problem: "EOF while parsing an object");
                }
                let separatorByte: UInt8 = self.documentBytes[self.cursorIndex];
                self.cursorIndex = self.cursorIndex + 1;
                if separatorByte == UInt8(ascii: "}") {
                    return .object(JsonWireObject(entries: parsedEntries));
                }
                if separatorByte != UInt8(ascii: ",") {
                    throw JsonWireProblem.malformedDocument(problem: "expected `,` or `}`");
                }
            }
        }

        private mutating func parseArrayAtCursor(currentDepth: Int) throws -> JsonWireValue {
            self.cursorIndex = self.cursorIndex + 1;
            var parsedElements: Array<JsonWireValue> = Array<JsonWireValue>();
            self.skipWhitespaceBytes();
            if self.isAtEnd == false && self.documentBytes[self.cursorIndex] == UInt8(ascii: "]") {
                self.cursorIndex = self.cursorIndex + 1;
                return .array(parsedElements);
            }
            while true {
                let parsedElement: JsonWireValue = try self.parseValueAtCursor(currentDepth: currentDepth + 1);
                parsedElements.append(parsedElement);
                self.skipWhitespaceBytes();
                guard self.isAtEnd == false else {
                    throw JsonWireProblem.malformedDocument(problem: "EOF while parsing a list");
                }
                let separatorByte: UInt8 = self.documentBytes[self.cursorIndex];
                self.cursorIndex = self.cursorIndex + 1;
                if separatorByte == UInt8(ascii: "]") {
                    return .array(parsedElements);
                }
                if separatorByte != UInt8(ascii: ",") {
                    throw JsonWireProblem.malformedDocument(problem: "expected `,` or `]`");
                }
            }
        }

        private mutating func parseStringAtCursor() throws -> String {
            self.cursorIndex = self.cursorIndex + 1;
            var decodedScalars: Array<UInt8> = Array<UInt8>();
            while true {
                guard self.isAtEnd == false else {
                    throw JsonWireProblem.malformedDocument(problem: "EOF while parsing a string");
                }
                let currentByte: UInt8 = self.documentBytes[self.cursorIndex];
                self.cursorIndex = self.cursorIndex + 1;
                if currentByte == UInt8(ascii: "\"") {
                    guard let decodedText = String(bytes: decodedScalars, encoding: .utf8) else {
                        throw JsonWireProblem.malformedDocument(problem: "invalid unicode code point");
                    }
                    return decodedText;
                }
                if currentByte != UInt8(ascii: "\\") {
                    decodedScalars.append(currentByte);
                    continue;
                }
                guard self.isAtEnd == false else {
                    throw JsonWireProblem.malformedDocument(problem: "EOF while parsing a string");
                }
                let escapeByte: UInt8 = self.documentBytes[self.cursorIndex];
                self.cursorIndex = self.cursorIndex + 1;
                switch escapeByte {
                case UInt8(ascii: "\""): decodedScalars.append(UInt8(ascii: "\""));
                case UInt8(ascii: "\\"): decodedScalars.append(UInt8(ascii: "\\"));
                case UInt8(ascii: "/"): decodedScalars.append(UInt8(ascii: "/"));
                case UInt8(ascii: "b"): decodedScalars.append(0x08);
                case UInt8(ascii: "f"): decodedScalars.append(0x0C);
                case UInt8(ascii: "n"): decodedScalars.append(0x0A);
                case UInt8(ascii: "r"): decodedScalars.append(0x0D);
                case UInt8(ascii: "t"): decodedScalars.append(0x09);
                case UInt8(ascii: "u"): decodedScalars.append(contentsOf: try self.parseUnicodeEscapeAtCursor());
                default: throw JsonWireProblem.malformedDocument(problem: "invalid escape");
                }
            }
        }

        private mutating func parseUnicodeEscapeAtCursor() throws -> Array<UInt8> {
            let firstCodeUnit: UInt16 = try self.parseHexQuadAtCursor();
            if firstCodeUnit >= 0xD800 && firstCodeUnit <= 0xDBFF {
                guard self.isAtEnd == false && self.documentBytes[self.cursorIndex] == UInt8(ascii: "\\") else {
                    throw JsonWireProblem.malformedDocument(problem: "lone leading surrogate in hex escape");
                }
                self.cursorIndex = self.cursorIndex + 1;
                guard self.isAtEnd == false && self.documentBytes[self.cursorIndex] == UInt8(ascii: "u") else {
                    throw JsonWireProblem.malformedDocument(problem: "unexpected end of hex escape");
                }
                self.cursorIndex = self.cursorIndex + 1;
                let secondCodeUnit: UInt16 = try self.parseHexQuadAtCursor();
                guard secondCodeUnit >= 0xDC00 && secondCodeUnit <= 0xDFFF else {
                    throw JsonWireProblem.malformedDocument(problem: "lone leading surrogate in hex escape");
                }
                let combinedScalarValue: UInt32 = 0x10000 + (UInt32(firstCodeUnit - 0xD800) << 10) + UInt32(secondCodeUnit - 0xDC00);
                guard let combinedScalar = Unicode.Scalar(combinedScalarValue) else {
                    throw JsonWireProblem.malformedDocument(problem: "invalid unicode code point");
                }
                return Array(String(Character(combinedScalar)).utf8);
            }
            if firstCodeUnit >= 0xDC00 && firstCodeUnit <= 0xDFFF {
                throw JsonWireProblem.malformedDocument(problem: "lone leading surrogate in hex escape");
            }
            guard let decodedScalar = Unicode.Scalar(UInt32(firstCodeUnit)) else {
                throw JsonWireProblem.malformedDocument(problem: "invalid unicode code point");
            }
            return Array(String(Character(decodedScalar)).utf8);
        }

        private mutating func parseHexQuadAtCursor() throws -> UInt16 {
            var parsedQuad: UInt16 = 0;
            for _ in 0..<4 {
                guard self.isAtEnd == false else {
                    throw JsonWireProblem.malformedDocument(problem: "EOF while parsing a string");
                }
                let digitByte: UInt8 = self.documentBytes[self.cursorIndex];
                self.cursorIndex = self.cursorIndex + 1;
                let digitValue: UInt16;
                switch digitByte {
                case UInt8(ascii: "0")...UInt8(ascii: "9"): digitValue = UInt16(digitByte - UInt8(ascii: "0"));
                case UInt8(ascii: "a")...UInt8(ascii: "f"): digitValue = UInt16(digitByte - UInt8(ascii: "a")) + 10;
                case UInt8(ascii: "A")...UInt8(ascii: "F"): digitValue = UInt16(digitByte - UInt8(ascii: "A")) + 10;
                default: throw JsonWireProblem.malformedDocument(problem: "invalid escape");
                }
                parsedQuad = (parsedQuad << 4) | digitValue;
            }
            return parsedQuad;
        }

        private mutating func parseNumberAtCursor() throws -> JsonWireValue {
            let numberStartIndex: Int = self.cursorIndex;
            var isNegative: Bool = false;
            if self.documentBytes[self.cursorIndex] == UInt8(ascii: "-") {
                isNegative = true;
                self.cursorIndex = self.cursorIndex + 1;
                guard self.isAtEnd == false && self.documentBytes[self.cursorIndex] >= UInt8(ascii: "0") && self.documentBytes[self.cursorIndex] <= UInt8(ascii: "9") else {
                    throw JsonWireProblem.malformedDocument(problem: "invalid number");
                }
            }
            guard self.isAtEnd == false else {
                throw JsonWireProblem.malformedDocument(problem: "EOF while parsing a value");
            }
            if self.documentBytes[self.cursorIndex] == UInt8(ascii: "0") {
                self.cursorIndex = self.cursorIndex + 1;
            } else {
                while self.isAtEnd == false && self.documentBytes[self.cursorIndex] >= UInt8(ascii: "0") && self.documentBytes[self.cursorIndex] <= UInt8(ascii: "9") {
                    self.cursorIndex = self.cursorIndex + 1;
                }
            }
            var hasFractionPart: Bool = false;
            if self.isAtEnd == false && self.documentBytes[self.cursorIndex] == UInt8(ascii: ".") {
                hasFractionPart = true;
                self.cursorIndex = self.cursorIndex + 1;
                guard self.isAtEnd == false && self.documentBytes[self.cursorIndex] >= UInt8(ascii: "0") && self.documentBytes[self.cursorIndex] <= UInt8(ascii: "9") else {
                    throw JsonWireProblem.malformedDocument(problem: "invalid number");
                }
                while self.isAtEnd == false && self.documentBytes[self.cursorIndex] >= UInt8(ascii: "0") && self.documentBytes[self.cursorIndex] <= UInt8(ascii: "9") {
                    self.cursorIndex = self.cursorIndex + 1;
                }
            }
            var hasExponentPart: Bool = false;
            if self.isAtEnd == false && (self.documentBytes[self.cursorIndex] == UInt8(ascii: "e") || self.documentBytes[self.cursorIndex] == UInt8(ascii: "E")) {
                hasExponentPart = true;
                self.cursorIndex = self.cursorIndex + 1;
                if self.isAtEnd == false && (self.documentBytes[self.cursorIndex] == UInt8(ascii: "+") || self.documentBytes[self.cursorIndex] == UInt8(ascii: "-")) {
                    self.cursorIndex = self.cursorIndex + 1;
                }
                guard self.isAtEnd == false && self.documentBytes[self.cursorIndex] >= UInt8(ascii: "0") && self.documentBytes[self.cursorIndex] <= UInt8(ascii: "9") else {
                    throw JsonWireProblem.malformedDocument(problem: "invalid number");
                }
                while self.isAtEnd == false && self.documentBytes[self.cursorIndex] >= UInt8(ascii: "0") && self.documentBytes[self.cursorIndex] <= UInt8(ascii: "9") {
                    self.cursorIndex = self.cursorIndex + 1;
                }
            }
            let numberText = String(bytes: Array(self.documentBytes[numberStartIndex..<self.cursorIndex]), encoding: .utf8) ?? "";
            if isNegative == false && hasFractionPart == false && hasExponentPart == false {
                if let unsignedValue = UInt64(numberText) {
                    return .unsignedInteger(unsignedValue);
                }
            }
            if hasFractionPart == false && hasExponentPart == false {
                if let signedValue = Int64(numberText) {
                    return .signedInteger(signedValue);
                }
            }
            guard let doubleValue = Double(numberText), doubleValue.isFinite else {
                throw JsonWireProblem.malformedDocument(problem: "number out of range");
            }
            return .double(doubleValue);
        }

        private mutating func expectLiteral(literalBytes: Array<UInt8>) throws {
            for expectedByte in literalBytes {
                guard self.isAtEnd == false && self.documentBytes[self.cursorIndex] == expectedByte else {
                    throw JsonWireProblem.malformedDocument(problem: "expected ident");
                }
                self.cursorIndex = self.cursorIndex + 1;
            }
        }
    }
}
