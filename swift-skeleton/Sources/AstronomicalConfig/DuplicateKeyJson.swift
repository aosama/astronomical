import Foundation;

/// Strict JSON parser that rejects objects containing duplicate keys, porting
/// crates/config/src/duplicate_key_json.rs. serde_json is streaming, so its
/// duplicate rejection happens while parsing; this Swift port implements the
/// same behavior with a hand-written recursive-descent parser over UTF-8
/// bytes. The parser produces JSONSerialization-compatible trees (NSNumber for
/// numbers and booleans, NSNull for null) so downstream StrictJson helpers and
/// UserConfigFile.fromJsonObject keep working unchanged.
internal enum DuplicateKeyJson {

    private static let maximumJsonDepth: Int = 128;

    private enum ParserFailure: Error {
        case duplicateKey(propertyName: String);
        case malformed(problem: String);
        case trailingContent;
        case emptyDocument;
    }

    internal static func parseJsonRejectingDuplicates(configFilePath: FilePath, configBytes: Data, performanceAttributionEnabled: Bool = false) throws -> Any {
        let passStart = ConfigPerformanceAttribution.startedPass(operationName: "parse_json_rejecting_duplicates", performanceAttributionEnabled: performanceAttributionEnabled);
        do {
            let parsedJson = try parseJsonBytes(configBytes: configBytes);
            ConfigPerformanceAttribution.finishedPass(operationName: "parse_json_rejecting_duplicates", passStart: passStart, passOutcome: "success", performanceAttributionEnabled: performanceAttributionEnabled);
            return parsedJson;
        } catch let parserFailure as ParserFailure {
            ConfigPerformanceAttribution.finishedPass(operationName: "parse_json_rejecting_duplicates", passStart: passStart, passOutcome: "failure", performanceAttributionEnabled: performanceAttributionEnabled);
            throw mapParserFailure(parserFailure: parserFailure, configFilePath: configFilePath);
        }
    }

    private static func parseJsonBytes(configBytes: Data) throws -> Any {
        var documentParser: JsonParser = JsonParser(configBytes: configBytes, rejectDuplicateKeys: true);
        documentParser.skipWhitespaceBytes();
        guard documentParser.isAtEnd == false else {
            throw ParserFailure.emptyDocument;
        }
        let parsedRootJson = try documentParser.parseValue(currentDepth: 0);
        documentParser.skipWhitespaceBytes();
        if documentParser.isAtEnd == false {
            throw ParserFailure.trailingContent;
        }
        return parsedRootJson;
    }

    private static func mapParserFailure(parserFailure: ParserFailure, configFilePath: FilePath) -> Error {
        switch parserFailure {
        case let .duplicateKey(propertyName):
            return ConfigResolutionError.duplicateConfigKey(configFilePath: configFilePath, duplicateKey: propertyName);
        case let .malformed(problem):
            return AstronomicalConfigError.parseConfigFile(configFilePath: configFilePath, underlyingDescription: problem);
        case .trailingContent:
            return AstronomicalConfigError.parseConfigFile(configFilePath: configFilePath, underlyingDescription: "trailing characters after the configuration document");
        case .emptyDocument:
            return AstronomicalConfigError.parseConfigFile(configFilePath: configFilePath, underlyingDescription: "the configuration file is empty");
        }
    }

    private struct JsonParser {

        private let configBytes: [UInt8];
        private let rejectDuplicateKeys: Bool;
        private var parseOffset: Int = 0;

        fileprivate init(configBytes: Data, rejectDuplicateKeys: Bool) {
            self.configBytes = Array(configBytes);
            self.rejectDuplicateKeys = rejectDuplicateKeys;
        }

        fileprivate var isAtEnd: Bool {
            return parseOffset >= configBytes.count;
        }

        fileprivate mutating func skipWhitespaceBytes() {
            while isAtEnd == false {
                let currentByte = configBytes[parseOffset];
                if currentByte == 0x20 || currentByte == 0x09 || currentByte == 0x0A || currentByte == 0x0D {
                    parseOffset += 1;
                } else {
                    break;
                }
            }
        }

        private mutating func expectNextByte(expectedByte: UInt8, problem: String) throws {
            if isAtEnd {
                throw ParserFailure.malformed(problem: problem);
            }
            if configBytes[parseOffset] != expectedByte {
                throw ParserFailure.malformed(problem: problem);
            }
            parseOffset += 1;
        }

        fileprivate mutating func parseValue(currentDepth: Int) throws -> Any {
            // serde_json::Deserializer caps nesting at 128 by default; the
            // check runs when a value starts, exactly like the Rust seed
            // recursion.
            if currentDepth > maximumJsonDepth {
                throw ParserFailure.malformed(problem: "configuration JSON exceeds the maximum nesting depth");
            }
            skipWhitespaceBytes();
            if isAtEnd {
                throw ParserFailure.malformed(problem: "unexpected end of the configuration document");
            }
            let valueByte = configBytes[parseOffset];
            switch valueByte {
            case UInt8(ascii: "{"):
                return try parseObjectBody(currentDepth: currentDepth);
            case UInt8(ascii: "["):
                return try parseArrayBody(currentDepth: currentDepth);
            case UInt8(ascii: "\""):
                return try parseStringValue();
            case UInt8(ascii: "t"):
                try expectLiteral(literalBytes: Array("true".utf8), problem: "expected true");
                return NSNumber(value: true);
            case UInt8(ascii: "f"):
                try expectLiteral(literalBytes: Array("false".utf8), problem: "expected false");
                return NSNumber(value: false);
            case UInt8(ascii: "n"):
                try expectLiteral(literalBytes: Array("null".utf8), problem: "expected null");
                return NSNull();
            case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"):
                return try parseNumberValue();
            default:
                throw ParserFailure.malformed(problem: "expected a JSON value");
            }
        }

        private mutating func expectLiteral(literalBytes: [UInt8], problem: String) throws {
            for literalByte in literalBytes {
                if isAtEnd || configBytes[parseOffset] != literalByte {
                    throw ParserFailure.malformed(problem: problem);
                }
                parseOffset += 1;
            }
        }

        private mutating func parseObjectBody(currentDepth: Int) throws -> Dictionary<String, Any> {
            try expectNextByte(expectedByte: UInt8(ascii: "{"), problem: "expected an object");
            var parsedObject: Dictionary<String, Any> = Dictionary<String, Any>();
            skipWhitespaceBytes();
            if isAtEnd {
                throw ParserFailure.malformed(problem: "unterminated object");
            }
            if configBytes[parseOffset] == UInt8(ascii: "}") {
                parseOffset += 1;
                return parsedObject;
            }
            while true {
                skipWhitespaceBytes();
                if isAtEnd {
                    throw ParserFailure.malformed(problem: "unterminated object");
                }
                if configBytes[parseOffset] != UInt8(ascii: "\"") {
                    throw ParserFailure.malformed(problem: "expected a quoted object key");
                }
                let memberKey = try parseStringValue();
                skipWhitespaceBytes();
                try expectNextByte(expectedByte: UInt8(ascii: ":"), problem: "expected : between object key and value");
                let memberValue = try parseValue(currentDepth: currentDepth + 1);
                if rejectDuplicateKeys {
                    if parsedObject.keys.contains(memberKey) {
                        throw ParserFailure.duplicateKey(propertyName: memberKey);
                    }
                }
                parsedObject[memberKey] = memberValue;
                skipWhitespaceBytes();
                if isAtEnd {
                    throw ParserFailure.malformed(problem: "unterminated object");
                }
                if configBytes[parseOffset] == UInt8(ascii: ",") {
                    parseOffset += 1;
                    continue;
                }
                if configBytes[parseOffset] == UInt8(ascii: "}") {
                    parseOffset += 1;
                    return parsedObject;
                }
                throw ParserFailure.malformed(problem: "expected , or } in object");
            }
        }

        private mutating func parseArrayBody(currentDepth: Int) throws -> Array<Any> {
            try expectNextByte(expectedByte: UInt8(ascii: "["), problem: "expected an array");
            var parsedElements: Array<Any> = Array<Any>();
            skipWhitespaceBytes();
            if isAtEnd {
                throw ParserFailure.malformed(problem: "unterminated array");
            }
            if configBytes[parseOffset] == UInt8(ascii: "]") {
                parseOffset += 1;
                return parsedElements;
            }
            while true {
                let parsedElement = try parseValue(currentDepth: currentDepth + 1);
                parsedElements.append(parsedElement);
                skipWhitespaceBytes();
                if isAtEnd {
                    throw ParserFailure.malformed(problem: "unterminated array");
                }
                if configBytes[parseOffset] == UInt8(ascii: ",") {
                    parseOffset += 1;
                    continue;
                }
                if configBytes[parseOffset] == UInt8(ascii: "]") {
                    parseOffset += 1;
                    return parsedElements;
                }
                throw ParserFailure.malformed(problem: "expected , or ] in array");
            }
        }

        private mutating func parseStringValue() throws -> String {
            try expectNextByte(expectedByte: UInt8(ascii: "\""), problem: "expected a string");
            var decodedScalars: Array<Character> = Array<Character>();
            while true {
                if isAtEnd {
                    throw ParserFailure.malformed(problem: "unterminated string");
                }
                let stringByte = configBytes[parseOffset];
                if stringByte == UInt8(ascii: "\"") {
                    parseOffset += 1;
                    return String(decodedScalars);
                }
                if stringByte == UInt8(ascii: "\\") {
                    parseOffset += 1;
                    try parseEscapeSequence(decodedScalars: &decodedScalars);
                    continue;
                }
                if stringByte < 0x20 {
                    throw ParserFailure.malformed(problem: "control character in string");
                }
                let (decodedCharacter, decodedByteCount) = decodeUtf8CharacterAt(parseOffset: parseOffset);
                guard let unwrappedDecodedCharacter = decodedCharacter else {
                    throw ParserFailure.malformed(problem: "invalid UTF-8 in string");
                }
                decodedScalars.append(unwrappedDecodedCharacter);
                parseOffset += decodedByteCount;
            }
        }

        private mutating func parseEscapeSequence(decodedScalars: inout Array<Character>) throws {
            if isAtEnd {
                throw ParserFailure.malformed(problem: "unterminated escape sequence");
            }
            let escapeByte = configBytes[parseOffset];
            parseOffset += 1;
            switch escapeByte {
            case UInt8(ascii: "\""): decodedScalars.append("\"");
            case UInt8(ascii: "\\"): decodedScalars.append("\\");
            case UInt8(ascii: "/"): decodedScalars.append("/");
            case UInt8(ascii: "b"): decodedScalars.append("\u{0008}");
            case UInt8(ascii: "f"): decodedScalars.append("\u{000C}");
            case UInt8(ascii: "n"): decodedScalars.append("\n");
            case UInt8(ascii: "r"): decodedScalars.append("\r");
            case UInt8(ascii: "t"): decodedScalars.append("\t");
            case UInt8(ascii: "u"):
                let firstUnit = try parseHexadecimalUnit();
                if firstUnit >= 0xD800 && firstUnit <= 0xDBFF {
                    try expectNextByte(expectedByte: UInt8(ascii: "\\"), problem: "unpaired surrogate half");
                    try expectNextByte(expectedByte: UInt8(ascii: "u"), problem: "unpaired surrogate half");
                    let secondUnit = try parseHexadecimalUnit();
                    if secondUnit < 0xDC00 || secondUnit > 0xDFFF {
                        throw ParserFailure.malformed(problem: "unpaired surrogate half");
                    }
                    let combinedScalar = 0x10000 + ((firstUnit - 0xD800) << 10) + (secondUnit - 0xDC00);
                    guard let combinedScalarValue = Unicode.Scalar(combinedScalar) else {
                        throw ParserFailure.malformed(problem: "invalid unicode escape");
                    }
                    decodedScalars.append(Character(combinedScalarValue));
                } else if firstUnit >= 0xDC00 && firstUnit <= 0xDFFF {
                    throw ParserFailure.malformed(problem: "unpaired surrogate half");
                } else {
                    guard let singleScalarValue = Unicode.Scalar(firstUnit) else {
                        throw ParserFailure.malformed(problem: "invalid unicode escape");
                    }
                    decodedScalars.append(Character(singleScalarValue));
                }
            default:
                throw ParserFailure.malformed(problem: "invalid escape sequence");
            }
        }

        private mutating func parseHexadecimalUnit() throws -> Int {
            var hexadecimalUnit: Int = 0;
            for _ in 0..<4 {
                if isAtEnd {
                    throw ParserFailure.malformed(problem: "truncated unicode escape");
                }
                let digitByte = configBytes[parseOffset];
                let digitValue: Int;
                if digitByte >= UInt8(ascii: "0") && digitByte <= UInt8(ascii: "9") {
                    digitValue = Int(digitByte - UInt8(ascii: "0"));
                } else if digitByte >= UInt8(ascii: "a") && digitByte <= UInt8(ascii: "f") {
                    digitValue = Int(digitByte - UInt8(ascii: "a")) + 10;
                } else if digitByte >= UInt8(ascii: "A") && digitByte <= UInt8(ascii: "F") {
                    digitValue = Int(digitByte - UInt8(ascii: "A")) + 10;
                } else {
                    throw ParserFailure.malformed(problem: "invalid unicode escape digit");
                }
                hexadecimalUnit = (hexadecimalUnit << 4) | digitValue;
                parseOffset += 1;
            }
            return hexadecimalUnit;
        }

        private func decodeUtf8CharacterAt(parseOffset: Int) -> (decodedCharacter: Character?, decodedByteCount: Int) {
            let leadByte = configBytes[parseOffset];
            let expectedByteCount: Int;
            if leadByte < 0x80 {
                expectedByteCount = 1;
            } else if leadByte >= 0xC2 && leadByte <= 0xDF {
                expectedByteCount = 2;
            } else if leadByte >= 0xE0 && leadByte <= 0xEF {
                expectedByteCount = 3;
            } else if leadByte >= 0xF0 && leadByte <= 0xF4 {
                expectedByteCount = 4;
            } else {
                return (nil, 0);
            }
            if parseOffset + expectedByteCount > configBytes.count {
                return (nil, 0);
            }
            let sequenceBytes = Array(configBytes[parseOffset..<(parseOffset + expectedByteCount)]);
            let decodedSequence = String(bytes: sequenceBytes, encoding: .utf8);
            if let unwrappedDecodedSequence = decodedSequence, unwrappedDecodedSequence.count == 1 {
                return (unwrappedDecodedSequence.first, expectedByteCount);
            }
            return (nil, 0);
        }

        private mutating func parseNumberValue() throws -> Any {
            let numberStartOffset = parseOffset;
            if isAtEnd {
                throw ParserFailure.malformed(problem: "expected a number");
            }
            if configBytes[parseOffset] == UInt8(ascii: "-") {
                parseOffset += 1;
            }
            if isAtEnd {
                throw ParserFailure.malformed(problem: "truncated number");
            }
            if configBytes[parseOffset] == UInt8(ascii: "0") {
                parseOffset += 1;
            } else if configBytes[parseOffset] >= UInt8(ascii: "1") && configBytes[parseOffset] <= UInt8(ascii: "9") {
                while isAtEnd == false && configBytes[parseOffset] >= UInt8(ascii: "0") && configBytes[parseOffset] <= UInt8(ascii: "9") {
                    parseOffset += 1;
                }
            } else {
                throw ParserFailure.malformed(problem: "invalid number");
            }
            var hasFractionPart: Bool = false;
            if isAtEnd == false && configBytes[parseOffset] == UInt8(ascii: ".") {
                hasFractionPart = true;
                parseOffset += 1;
                if isAtEnd || configBytes[parseOffset] < UInt8(ascii: "0") || configBytes[parseOffset] > UInt8(ascii: "9") {
                    throw ParserFailure.malformed(problem: "invalid number fraction");
                }
                while isAtEnd == false && configBytes[parseOffset] >= UInt8(ascii: "0") && configBytes[parseOffset] <= UInt8(ascii: "9") {
                    parseOffset += 1;
                }
            }
            var hasExponentPart: Bool = false;
            if isAtEnd == false && (configBytes[parseOffset] == UInt8(ascii: "e") || configBytes[parseOffset] == UInt8(ascii: "E")) {
                hasExponentPart = true;
                parseOffset += 1;
                if isAtEnd == false && (configBytes[parseOffset] == UInt8(ascii: "+") || configBytes[parseOffset] == UInt8(ascii: "-")) {
                    parseOffset += 1;
                }
                if isAtEnd || configBytes[parseOffset] < UInt8(ascii: "0") || configBytes[parseOffset] > UInt8(ascii: "9") {
                    throw ParserFailure.malformed(problem: "invalid number exponent");
                }
                while isAtEnd == false && configBytes[parseOffset] >= UInt8(ascii: "0") && configBytes[parseOffset] <= UInt8(ascii: "9") {
                    parseOffset += 1;
                }
            }
            let numberText = String(bytes: configBytes[numberStartOffset..<parseOffset], encoding: .utf8) ?? "";
            if hasFractionPart == false && hasExponentPart == false {
                if numberText.hasPrefix("-") == false, let unsignedParsed = UInt64(numberText) {
                    return NSNumber(value: unsignedParsed);
                }
                if let signedParsed = Int64(numberText) {
                    return NSNumber(value: signedParsed);
                }
                // Very large integer literals lose integer precision exactly
                // the way serde_json's arbitrary-precision-free default does
                // not, but no config value reaches that magnitude; keep the
                // document parseable as a float.
                if let oversizedParsed = Double(numberText) {
                    return NSNumber(value: oversizedParsed);
                }
                throw ParserFailure.malformed(problem: "invalid number");
            }
            guard let floatParsed = Double(numberText) else {
                throw ParserFailure.malformed(problem: "invalid number");
            }
            return NSNumber(value: floatParsed);
        }
    }
}
