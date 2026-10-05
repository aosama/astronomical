import Foundation;

/// Ordered JSON value tree used as the single wire representation for every
/// IpcProtocol message. Hand-rolled because the platform JSONEncoder both
/// reorders object keys and escapes solidus characters, which breaks the
/// byte-exact serde_json compatibility contracts this module guards.
internal enum JsonWireValue: Equatable {
    case null;
    case boolean(Bool);
    case unsignedInteger(UInt64);
    case signedInteger(Int64);
    case double(Double);
    case float32(Float);
    case string(String);
    case array(Array<JsonWireValue>);
    case object(JsonWireObject);

    internal var isNull: Bool {
        if case .null = self {
            return true;
        }
        return false;
    }

    /// serde_json prints finite float32 values with shortest round-trip
    /// formatting; Swift's Float description matches for all values the
    /// protocol carries (non-finite floats are rejected by the writer).
    internal var serializedText: String {
        get throws {
            var writer: JsonWireWriter = JsonWireWriter();
            try writer.appendValue(self);
            return writer.serializedText;
        }
    }
}

/// Insertion-ordered JSON object preserving the exact key order the Rust
/// serde derives emit, with rejection of duplicate keys while parsing.
internal struct JsonWireObject: Equatable {
    private(set) var entries: Array<(key: String, value: JsonWireValue)>;

    internal init(entries: Array<(key: String, value: JsonWireValue)>) {
        self.entries = entries;
    }

    internal static func == (lhsValue: JsonWireObject, rhsValue: JsonWireObject) -> Bool {
        if lhsValue.entries.count != rhsValue.entries.count {
            return false;
        }
        for entryIndex in lhsValue.entries.indices {
            let leftEntry = lhsValue.entries[entryIndex];
            let rightEntry = rhsValue.entries[entryIndex];
            if leftEntry.key != rightEntry.key || leftEntry.value != rightEntry.value {
                return false;
            }
        }
        return true;
    }

    internal var keyNames: Array<String> {
        return self.entries.map({ (entry: (key: String, value: JsonWireValue)) -> String in entry.key });
    }

    internal func value(forKey propertyName: String) -> JsonWireValue? {
        for entry in self.entries {
            if entry.key == propertyName {
                return entry.value;
            }
        }
        return nil;
    }

    internal mutating func appendEntry(key propertyName: String, value wireValue: JsonWireValue) {
        self.entries.append((propertyName, wireValue));
    }
}

/// Serde-shaped decode problems. Messages mirror serde_json wording closely
/// enough for the compatibility tests that assert on field names. Public because
/// ProtocolError carries encode/decode failures as associated payloads.
public enum JsonWireProblem: Error, CustomStringConvertible {
    case malformedDocument(problem: String);
    case recursionLimitExceeded;
    case duplicateField(fieldName: String);
    case missingField(fieldName: String);
    case unknownField(fieldName: String, expectedFields: Array<String>);
    case invalidType(expectedTypeName: String, found: String);
    case expectedObject(found: String);
    case expectedString(found: String);

    public var description: String {
        switch self {
        case let .malformedDocument(problem): return problem;
        case .recursionLimitExceeded: return "recursion limit exceeded";
        case let .duplicateField(fieldName): return "duplicate field `\(fieldName)`";
        case let .missingField(fieldName): return "missing field `\(fieldName)`";
        case let .unknownField(fieldName, expectedFields): return "unknown field `\(fieldName)`, expected one of \(JsonWireProblem.formattedFieldList(expectedFields))";
        case let .invalidType(expectedTypeName, found): return "invalid type: \(found), expected \(expectedTypeName)";
        case let .expectedObject(found): return "invalid type: \(found), expected a map";
        case let .expectedString(found): return "invalid type: \(found), expected a string";
        }
    }

    internal static func formattedFieldList(_ fieldNames: Array<String>) -> String {
        return fieldNames.map({ (fieldName: String) -> String in "`\(fieldName)`" }).joined(separator: ", ");
    }
}

/// Typed extraction helpers shared by every wire-mapped payload type.
extension JsonWireObject {

    internal func requireObjectValue(fieldName propertyName: String) throws -> JsonWireValue {
        guard let fieldValue = self.value(forKey: propertyName) else {
            throw JsonWireProblem.missingField(fieldName: propertyName);
        }
        return fieldValue;
    }

    internal func decodeString(fieldName propertyName: String) throws -> String {
        return try JsonWireValue.extractString(try self.requireObjectValue(fieldName: propertyName));
    }

    internal func decodeBool(fieldName propertyName: String) throws -> Bool {
        return try JsonWireValue.extractBool(try self.requireObjectValue(fieldName: propertyName));
    }

    internal func decodeArray<Element>(fieldName propertyName: String, mappedElement: (JsonWireValue) throws -> Element) throws -> Array<Element> {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        return try JsonWireValue.extractArray(fieldValue, mappedElement: mappedElement);
    }

    internal func decodeArrayAllowingAbsent<Element>(fieldName propertyName: String, mappedElement: (JsonWireValue) throws -> Element) throws -> Array<Element> {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return Array<Element>();
        }
        return try JsonWireValue.extractArray(fieldValue, mappedElement: mappedElement);
    }

    internal func decodeOptionalStringAllowingAbsent(fieldName propertyName: String) throws -> String? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractString(fieldValue);
    }

    internal func decodeOptionalUInt16AllowingAbsent(fieldName propertyName: String) throws -> UInt16? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.clampToUInt16(try JsonWireValue.extractUInt64(fieldValue));
    }

    internal func decodeOptionalObjectAllowingAbsent(fieldName propertyName: String) throws -> JsonWireObject? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractObject(fieldValue);
    }

    internal func decodeBoolAllowingAbsent(fieldName propertyName: String) throws -> Bool {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return false;
        }
        return try JsonWireValue.extractBool(fieldValue);
    }

    /// Reads the variant tag of an internally tagged enum payload and rejects
    /// unrecognized variant names with serde's wording.
    internal func decodeTaggedVariantName(tagFieldName: String, expectedVariantNames: Array<String>) throws -> String {
        let tagWireValue = try self.requireObjectValue(fieldName: tagFieldName);
        let variantName = try JsonWireValue.extractString(tagWireValue);
        guard expectedVariantNames.contains(variantName) else {
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(variantName)`, expected one of \(JsonWireProblem.formattedFieldList(expectedVariantNames))");
        }
        return variantName;
    }

    /// Rejects keys outside a tagged variant's field set while leaving the
    /// variant tag itself in place, mirroring how serde strips the tag before
    /// deny_unknown_fields sees the variant payload.
    internal func rejectUnknownFieldsBesidesTag(tagFieldName: String, allowedFieldNames: Array<String>) throws {
        for propertyName in self.keyNames {
            if propertyName != tagFieldName && allowedFieldNames.contains(propertyName) == false {
                throw JsonWireProblem.unknownField(fieldName: propertyName, expectedFields: allowedFieldNames);
            }
        }
    }

    internal func decodeOptionalString(fieldName propertyName: String) throws -> String? {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractString(fieldValue);
    }

    internal func decodeUInt64(fieldName propertyName: String) throws -> UInt64 {
        return try JsonWireValue.extractUInt64(try self.requireObjectValue(fieldName: propertyName));
    }

    internal func decodeOptionalUInt64(fieldName propertyName: String) throws -> UInt64? {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractUInt64(fieldValue);
    }

    internal func decodeUInt32(fieldName propertyName: String) throws -> UInt32 {
        return try JsonWireValue.clampToUInt32(try self.decodeUInt64(fieldName: propertyName));
    }

    internal func decodeOptionalUInt32(fieldName propertyName: String) throws -> UInt32? {
        guard let unwrappedValue = try self.decodeOptionalUInt64(fieldName: propertyName) else {
            return nil;
        }
        return try JsonWireValue.clampToUInt32(unwrappedValue);
    }

    internal func decodeUInt16(fieldName propertyName: String) throws -> UInt16 {
        return try JsonWireValue.clampToUInt16(try self.decodeUInt64(fieldName: propertyName));
    }

    internal func decodeUInt8(fieldName propertyName: String) throws -> UInt8 {
        return try JsonWireValue.clampToUInt8(try self.decodeUInt64(fieldName: propertyName));
    }

    internal func decodeOptionalUInt8(fieldName propertyName: String) throws -> UInt8? {
        guard let unwrappedValue = try self.decodeOptionalUInt64(fieldName: propertyName) else {
            return nil;
        }
        return try JsonWireValue.clampToUInt8(unwrappedValue);
    }

    internal func decodeOptionalUInt16(fieldName propertyName: String) throws -> UInt16? {
        guard let unwrappedValue = try self.decodeOptionalUInt64(fieldName: propertyName) else {
            return nil;
        }
        return try JsonWireValue.clampToUInt16(unwrappedValue);
    }

    internal func decodeInt64(fieldName propertyName: String) throws -> Int64 {
        return try JsonWireValue.extractInt64(try self.requireObjectValue(fieldName: propertyName));
    }

    internal func decodeOptionalInt64(fieldName propertyName: String) throws -> Int64? {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractInt64(fieldValue);
    }

    internal func decodeObject(fieldName propertyName: String) throws -> JsonWireObject {
        return try JsonWireValue.extractObject(try self.requireObjectValue(fieldName: propertyName));
    }

    internal func decodeOptionalObject(fieldName propertyName: String) throws -> JsonWireObject? {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractObject(fieldValue);
    }

    /// Rejects keys outside the variant's field set after the tagged payload
    /// decoded, mirroring serde deny_unknown_fields.
    internal func rejectUnknownFields(allowedFieldNames: Array<String>) throws {
        for propertyName in self.keyNames {
            if allowedFieldNames.contains(propertyName) == false {
                throw JsonWireProblem.unknownField(fieldName: propertyName, expectedFields: allowedFieldNames);
            }
        }
    }
}

extension JsonWireValue {

    internal static func extractString(_ wireValue: JsonWireValue) throws -> String {
        guard case let .string(textValue) = wireValue else {
            throw JsonWireProblem.expectedString(found: wireValue.foundDescription);
        }
        return textValue;
    }

    internal static func extractUInt64(_ wireValue: JsonWireValue) throws -> UInt64 {
        switch wireValue {
        case let .unsignedInteger(numericValue): return numericValue;
        default: throw JsonWireProblem.invalidType(expectedTypeName: "u64", found: wireValue.foundDescription);
        }
    }

    internal static func extractInt64(_ wireValue: JsonWireValue) throws -> Int64 {
        switch wireValue {
        case let .signedInteger(numericValue): return numericValue;
        default: throw JsonWireProblem.invalidType(expectedTypeName: "i64", found: wireValue.foundDescription);
        }
    }

    internal static func extractBool(_ wireValue: JsonWireValue) throws -> Bool {
        guard case let .boolean(booleanValue) = wireValue else {
            throw JsonWireProblem.invalidType(expectedTypeName: "a boolean", found: wireValue.foundDescription);
        }
        return booleanValue;
    }

    internal static func extractArray<Element>(_ wireValue: JsonWireValue, mappedElement: (JsonWireValue) throws -> Element) throws -> Array<Element> {        guard case let .array(elementValues) = wireValue else {
            throw JsonWireProblem.invalidType(expectedTypeName: "a sequence", found: wireValue.foundDescription);
        }
        var decodedElements = Array<Element>();
        decodedElements.reserveCapacity(elementValues.count);
        for elementValue in elementValues {
            let decodedElement = try mappedElement(elementValue);
            decodedElements.append(decodedElement);
        }
        return decodedElements;
    }

    internal static func extractObject(_ wireValue: JsonWireValue) throws -> JsonWireObject {
        guard case let .object(objectValue) = wireValue else {
            throw JsonWireProblem.expectedObject(found: wireValue.foundDescription);
        }
        return objectValue;
    }

    internal static func clampToUInt32(_ rawValue: UInt64) throws -> UInt32 {
        guard rawValue <= UInt64(UInt32.max) else {
            throw JsonWireProblem.invalidType(expectedTypeName: "u32", found: "integer `\(rawValue)`");
        }
        return UInt32(rawValue);
    }

    internal static func clampToUInt16(_ rawValue: UInt64) throws -> UInt16 {
        guard rawValue <= UInt64(UInt16.max) else {
            throw JsonWireProblem.invalidType(expectedTypeName: "u16", found: "integer `\(rawValue)`");
        }
        return UInt16(rawValue);
    }

    internal static func clampToUInt8(_ rawValue: UInt64) throws -> UInt8 {
        guard rawValue <= UInt64(UInt8.max) else {
            throw JsonWireProblem.invalidType(expectedTypeName: "u8", found: "integer `\(rawValue)`");
        }
        return UInt8(rawValue);
    }

    internal var foundDescription: String {
        switch self {
        case .null: return "null";
        case let .boolean(booleanValue): return "boolean `\(booleanValue)`";
        case let .unsignedInteger(numericValue): return "integer `\(numericValue)`";
        case let .signedInteger(numericValue): return "integer `\(numericValue)`";
        case let .double(numericValue): return "floating point `\(numericValue)`";
        case let .float32(numericValue): return "floating point `\(numericValue)`";
        case let .string(textValue): return "string \(textValue.serializedJsonQuoted)";
        case .array: return "sequence";
        case .object: return "map";
        }
    }

    internal static func mappedArray<Element>(_ values: Array<Element>, mappedWireValue: (Element) -> JsonWireValue) -> JsonWireValue {
        var elementWireValues = Array<JsonWireValue>();
        elementWireValues.reserveCapacity(values.count);
        for elementValue in values {
            elementWireValues.append(mappedWireValue(elementValue));
        }
        return .array(elementWireValues);
    }
}

extension String {
    internal var serializedJsonQuoted: String {
        var writer: JsonWireWriter = JsonWireWriter();
        writer.appendStringValue(self);
        return writer.serializedText;
    }
}
