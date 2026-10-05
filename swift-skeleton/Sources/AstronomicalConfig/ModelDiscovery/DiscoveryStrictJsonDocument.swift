import Foundation;

/**
 * Strict RFC 8259 JSON scanner producing a JSONSerialization-shaped value
 * tree (Dictionary/Array/String/NSNumber/NSNull).
 *
 * JSONSerialization silently keeps the last of duplicate object keys, while
 * serde rejects a duplicated field with a "duplicate field" error. Discovery
 * documents are decoded against serde-shaped contracts, so this scanner
 * rejects a duplicated key at any object depth instead of letting one shadow
 * another.
 */
internal enum DiscoveryStrictJsonDocument {
    internal enum ParseError: Error, CustomStringConvertible {
        case malformedJson(description: String);
        case duplicateObjectKey(keyName: String);

        internal var description: String {
            switch (self) {
            case .malformedJson(let problemDescription): return "malformed JSON document: \(problemDescription)";
            case .duplicateObjectKey(let keyName): return "duplicate object key \"\(keyName)\"";
            }
        }
    }

    internal static func parseDocument(bytes: Data) throws -> Any {
        var documentParser: DocumentParser = DocumentParser(documentBytes: [UInt8](bytes));
        documentParser.skipInsignificantWhitespace();
        let parsedDocumentValue: Any = try documentParser.parseValue();
        documentParser.skipInsignificantWhitespace();
        guard documentParser.isAtEnd else {
            throw ParseError.malformedJson(description: "trailing content after the JSON document");
        }
        return parsedDocumentValue;
    }

    private struct DocumentParser {
        private let documentBytes: Array<UInt8>;
        private var readOffset: Int;

        fileprivate init(documentBytes: Array<UInt8>) {
            self.documentBytes = documentBytes;
            self.readOffset = 0;
        }

        fileprivate var isAtEnd: Bool {
            return self.readOffset >= self.documentBytes.count;
        }

        fileprivate mutating func skipInsignificantWhitespace() {
            while !self.isAtEnd {
                let whitespaceByte: UInt8 = self.documentBytes[self.readOffset];
                if whitespaceByte == 0x20 || whitespaceByte == 0x09 || whitespaceByte == 0x0A || whitespaceByte == 0x0D {
                    self.readOffset += 1;
                    continue;
                }
                return;
            }
        }

        fileprivate mutating func parseValue() throws -> Any {
            guard !self.isAtEnd else {
                throw ParseError.malformedJson(description: "unexpected end of document");
            }
            switch (self.documentBytes[self.readOffset]) {
            case UInt8(ascii: "{"): return try self.parseObject();
            case UInt8(ascii: "["): return try self.parseArray();
            case UInt8(ascii: "\""): return try self.parseString();
            case UInt8(ascii: "t"): return try self.parseLiteral(literalBytes: Array<UInt8>("true".utf8), parsedValue: true);
            case UInt8(ascii: "f"): return try self.parseLiteral(literalBytes: Array<UInt8>("false".utf8), parsedValue: false);
            case UInt8(ascii: "n"): return try self.parseLiteral(literalBytes: Array<UInt8>("null".utf8), parsedValue: NSNull());
            default: return try self.parseNumber();
            }
        }

        private mutating func parseObject() throws -> Dictionary<String, Any> {
            self.readOffset += 1;
            var parsedMembers: Dictionary<String, Any> = Dictionary<String, Any>();
            self.skipInsignificantWhitespace();
            if !self.isAtEnd && self.documentBytes[self.readOffset] == UInt8(ascii: "}") {
                self.readOffset += 1;
                return parsedMembers;
            }
            while true {
                self.skipInsignificantWhitespace();
                guard !self.isAtEnd, self.documentBytes[self.readOffset] == UInt8(ascii: "\"") else {
                    throw ParseError.malformedJson(description: "expected an object key");
                }
                let memberKeyName: String = try self.parseString();
                self.skipInsignificantWhitespace();
                guard !self.isAtEnd, self.documentBytes[self.readOffset] == UInt8(ascii: ":") else {
                    throw ParseError.malformedJson(description: "expected ':' after an object key");
                }
                self.readOffset += 1;
                self.skipInsignificantWhitespace();
                let memberValue: Any = try self.parseValue();
                if parsedMembers.keys.contains(memberKeyName) {
                    throw ParseError.duplicateObjectKey(keyName: memberKeyName);
                }
                parsedMembers[memberKeyName] = memberValue;
                self.skipInsignificantWhitespace();
                guard !self.isAtEnd else {
                    throw ParseError.malformedJson(description: "unterminated object");
                }
                let separatorByte: UInt8 = self.documentBytes[self.readOffset];
                self.readOffset += 1;
                if separatorByte == UInt8(ascii: "}") {
                    return parsedMembers;
                }
                guard separatorByte == UInt8(ascii: ",") else {
                    throw ParseError.malformedJson(description: "expected ',' or '}' in an object");
                }
            }
        }

        private mutating func parseArray() throws -> Array<Any> {
            self.readOffset += 1;
            var parsedElements: Array<Any> = Array<Any>();
            self.skipInsignificantWhitespace();
            if !self.isAtEnd && self.documentBytes[self.readOffset] == UInt8(ascii: "]") {
                self.readOffset += 1;
                return parsedElements;
            }
            while true {
                self.skipInsignificantWhitespace();
                parsedElements.append(try self.parseValue());
                self.skipInsignificantWhitespace();
                guard !self.isAtEnd else {
                    throw ParseError.malformedJson(description: "unterminated array");
                }
                let separatorByte: UInt8 = self.documentBytes[self.readOffset];
                self.readOffset += 1;
                if separatorByte == UInt8(ascii: "]") {
                    return parsedElements;
                }
                guard separatorByte == UInt8(ascii: ",") else {
                    throw ParseError.malformedJson(description: "expected ',' or ']' in an array");
                }
            }
        }

        private mutating func parseString() throws -> String {
            self.readOffset += 1;
            var stringBytes: Array<UInt8> = Array<UInt8>();
            while true {
                guard !self.isAtEnd else {
                    throw ParseError.malformedJson(description: "unterminated string");
                }
                let currentByte: UInt8 = self.documentBytes[self.readOffset];
                if currentByte == UInt8(ascii: "\"") {
                    self.readOffset += 1;
                    return String(decoding: stringBytes, as: UTF8.self);
                }
                if currentByte == UInt8(ascii: "\\") {
                    try self.parseEscapedByte(into: &stringBytes);
                    continue;
                }
                if currentByte < 0x20 {
                    throw ParseError.malformedJson(description: "unescaped control character in a string");
                }
                try self.parseRawUtf8Sequence(into: &stringBytes);
            }
        }

        private mutating func parseEscapedByte(into stringBytes: inout Array<UInt8>) throws -> Void {
            guard self.readOffset + 1 < self.documentBytes.count else {
                throw ParseError.malformedJson(description: "truncated escape sequence");
            }
            self.readOffset += 1;
            let escapeByte: UInt8 = self.documentBytes[self.readOffset];
            self.readOffset += 1;
            switch (escapeByte) {
            case UInt8(ascii: "\""): stringBytes.append(UInt8(ascii: "\""));
            case UInt8(ascii: "\\"): stringBytes.append(UInt8(ascii: "\\"));
            case UInt8(ascii: "/"): stringBytes.append(UInt8(ascii: "/"));
            case UInt8(ascii: "b"): stringBytes.append(0x08);
            case UInt8(ascii: "f"): stringBytes.append(0x0C);
            case UInt8(ascii: "n"): stringBytes.append(0x0A);
            case UInt8(ascii: "r"): stringBytes.append(0x0D);
            case UInt8(ascii: "t"): stringBytes.append(0x09);
            case UInt8(ascii: "u"): try self.parseUnicodeEscape(into: &stringBytes);
            default: throw ParseError.malformedJson(description: "invalid escape sequence");
            }
        }

        private mutating func parseUnicodeEscape(into stringBytes: inout Array<UInt8>) throws -> Void {
            let leadingCodeUnit: UInt16 = try self.parseHexadecimalCodeUnit();
            if leadingCodeUnit >= 0xD800 && leadingCodeUnit <= 0xDBFF {
                guard self.readOffset + 1 < self.documentBytes.count,
                    self.documentBytes[self.readOffset] == UInt8(ascii: "\\"),
                    self.documentBytes[self.readOffset + 1] == UInt8(ascii: "u")
                else {
                    throw ParseError.malformedJson(description: "unpaired surrogate escape");
                }
                self.readOffset += 2;
                let trailingCodeUnit: UInt16 = try self.parseHexadecimalCodeUnit();
                guard trailingCodeUnit >= 0xDC00 && trailingCodeUnit <= 0xDFFF else {
                    throw ParseError.malformedJson(description: "unpaired surrogate escape");
                }
                let combinedCodePoint: UInt32 = 0x10000
                    + (UInt32(leadingCodeUnit - 0xD800) << 10)
                    + UInt32(trailingCodeUnit - 0xDC00);
                guard let combinedScalar: Unicode.Scalar = Unicode.Scalar(combinedCodePoint) else {
                    throw ParseError.malformedJson(description: "invalid surrogate pair escape");
                }
                stringBytes.append(contentsOf: Array<UInt8>(combinedScalar.utf8));
                return;
            }
            if leadingCodeUnit >= 0xDC00 && leadingCodeUnit <= 0xDFFF {
                throw ParseError.malformedJson(description: "unpaired surrogate escape");
            }
            guard let escapedScalar: Unicode.Scalar = Unicode.Scalar(UInt32(leadingCodeUnit)) else {
                throw ParseError.malformedJson(description: "invalid unicode escape");
            }
            stringBytes.append(contentsOf: Array<UInt8>(escapedScalar.utf8));
        }

        private mutating func parseHexadecimalCodeUnit() throws -> UInt16 {
            var parsedCodeUnit: UInt16 = 0;
            for _ in 0..<4 {
                guard !self.isAtEnd else {
                    throw ParseError.malformedJson(description: "truncated unicode escape");
                }
                let hexadecimalByte: UInt8 = self.documentBytes[self.readOffset];
                let hexadecimalDigit: UInt16;
                switch (hexadecimalByte) {
                case UInt8(ascii: "0")...UInt8(ascii: "9"): hexadecimalDigit = UInt16(hexadecimalByte - UInt8(ascii: "0"));
                case UInt8(ascii: "a")...UInt8(ascii: "f"): hexadecimalDigit = UInt16(hexadecimalByte - UInt8(ascii: "a")) + 10;
                case UInt8(ascii: "A")...UInt8(ascii: "F"): hexadecimalDigit = UInt16(hexadecimalByte - UInt8(ascii: "A")) + 10;
                default: throw ParseError.malformedJson(description: "invalid unicode escape digit");
                }
                parsedCodeUnit = parsedCodeUnit << 4 | hexadecimalDigit;
                self.readOffset += 1;
            }
            return parsedCodeUnit;
        }

        /**
         * Validates and appends one non-ASCII UTF-8 sequence; serde rejects
         * malformed byte input, so replacement-decoding would be unfaithful.
         */
        private mutating func parseRawUtf8Sequence(into stringBytes: inout Array<UInt8>) throws -> Void {
            let leadingByte: UInt8 = self.documentBytes[self.readOffset];
            let continuationByteCount: Int;
            let firstContinuationLowerBound: UInt8;
            let firstContinuationUpperBound: UInt8;
            switch (leadingByte) {
            case 0xC2...0xDF:
                continuationByteCount = 1;
                firstContinuationLowerBound = 0x80;
                firstContinuationUpperBound = 0xBF;
            case 0xE0:
                continuationByteCount = 2;
                firstContinuationLowerBound = 0xA0;
                firstContinuationUpperBound = 0xBF;
            case 0xE1...0xEC, 0xEE...0xEF:
                continuationByteCount = 2;
                firstContinuationLowerBound = 0x80;
                firstContinuationUpperBound = 0xBF;
            case 0xED:
                continuationByteCount = 2;
                firstContinuationLowerBound = 0x80;
                firstContinuationUpperBound = 0x9F;
            case 0xF0:
                continuationByteCount = 3;
                firstContinuationLowerBound = 0x90;
                firstContinuationUpperBound = 0xBF;
            case 0xF1...0xF3:
                continuationByteCount = 3;
                firstContinuationLowerBound = 0x80;
                firstContinuationUpperBound = 0xBF;
            case 0xF4:
                continuationByteCount = 3;
                firstContinuationLowerBound = 0x80;
                firstContinuationUpperBound = 0x8F;
            default:
                throw ParseError.malformedJson(description: "invalid UTF-8 sequence in a string");
            }
            guard self.readOffset + continuationByteCount < self.documentBytes.count else {
                throw ParseError.malformedJson(description: "truncated UTF-8 sequence in a string");
            }
            let sequenceEndOffset: Int = self.readOffset + continuationByteCount + 1;
            for sequenceOffset: Int in (self.readOffset + 1)..<sequenceEndOffset {
                let continuationByte: UInt8 = self.documentBytes[sequenceOffset];
                if sequenceOffset == self.readOffset + 1 {
                    guard continuationByte >= firstContinuationLowerBound && continuationByte <= firstContinuationUpperBound else {
                        throw ParseError.malformedJson(description: "invalid UTF-8 sequence in a string");
                    }
                    continue;
                }
                guard continuationByte >= 0x80 && continuationByte <= 0xBF else {
                    throw ParseError.malformedJson(description: "invalid UTF-8 sequence in a string");
                }
            }
            stringBytes.append(contentsOf: self.documentBytes[self.readOffset..<sequenceEndOffset]);
            self.readOffset = sequenceEndOffset;
        }

        private mutating func parseNumber() throws -> Any {
            let numberStartOffset: Int = self.readOffset;
            if !self.isAtEnd && self.documentBytes[self.readOffset] == UInt8(ascii: "-") {
                self.readOffset += 1;
            }
            guard !self.isAtEnd, self.documentBytes[self.readOffset] >= UInt8(ascii: "0"),
                self.documentBytes[self.readOffset] <= UInt8(ascii: "9")
            else {
                throw ParseError.malformedJson(description: "expected a digit in a number");
            }
            if self.documentBytes[self.readOffset] == UInt8(ascii: "0") {
                self.readOffset += 1;
            } else {
                while !self.isAtEnd, self.isDigitByte(self.documentBytes[self.readOffset]) {
                    self.readOffset += 1;
                }
            }
            var isDecimalNumber: Bool = false;
            if !self.isAtEnd && self.documentBytes[self.readOffset] == UInt8(ascii: ".") {
                isDecimalNumber = true;
                self.readOffset += 1;
                guard !self.isAtEnd, self.isDigitByte(self.documentBytes[self.readOffset]) else {
                    throw ParseError.malformedJson(description: "expected a digit after a decimal point");
                }
                while !self.isAtEnd, self.isDigitByte(self.documentBytes[self.readOffset]) {
                    self.readOffset += 1;
                }
            }
            if !self.isAtEnd,
                self.documentBytes[self.readOffset] == UInt8(ascii: "e") || self.documentBytes[self.readOffset] == UInt8(ascii: "E")
            {
                isDecimalNumber = true;
                self.readOffset += 1;
                if !self.isAtEnd,
                    self.documentBytes[self.readOffset] == UInt8(ascii: "+") || self.documentBytes[self.readOffset] == UInt8(ascii: "-")
                {
                    self.readOffset += 1;
                }
                guard !self.isAtEnd, self.isDigitByte(self.documentBytes[self.readOffset]) else {
                    throw ParseError.malformedJson(description: "expected an exponent digit");
                }
                while !self.isAtEnd, self.isDigitByte(self.documentBytes[self.readOffset]) {
                    self.readOffset += 1;
                }
            }
            let numberText: String = String(
                decoding: self.documentBytes[numberStartOffset..<self.readOffset],
                as: UTF8.self
            );
            if !isDecimalNumber {
                if let unsignedNumberValue: UInt64 = UInt64(numberText) {
                    return NSNumber(value: unsignedNumberValue);
                }
                if let signedNumberValue: Int64 = Int64(numberText) {
                    return NSNumber(value: signedNumberValue);
                }
            }
            guard let doubleNumberValue: Double = Double(numberText) else {
                throw ParseError.malformedJson(description: "invalid number");
            }
            return NSNumber(value: doubleNumberValue);
        }

        private mutating func parseLiteral(literalBytes: Array<UInt8>, parsedValue: Any) throws -> Any {
            let literalEndOffset: Int = self.readOffset + literalBytes.count;
            guard literalEndOffset <= self.documentBytes.count else {
                throw ParseError.malformedJson(description: "truncated literal");
            }
            guard Array<UInt8>(self.documentBytes[self.readOffset..<literalEndOffset]) == literalBytes else {
                throw ParseError.malformedJson(description: "invalid literal");
            }
            self.readOffset = literalEndOffset;
            return parsedValue;
        }

        private func isDigitByte(_ candidateByte: UInt8) -> Bool {
            return candidateByte >= UInt8(ascii: "0") && candidateByte <= UInt8(ascii: "9");
        }
    }
}
