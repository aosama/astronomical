import Foundation;

/// Ordered JSON value tree used as the single wire representation for every
/// IpcProtocol message. Hand-rolled because the platform JSONEncoder both
/// reorders object keys and escapes solidus characters, which breaks the
/// byte-exact serde_json compatibility contracts this module guards.
public enum JsonWireValue: Equatable {
    case null;
    case boolean(Bool);
    case unsignedInteger(UInt64);
    case signedInteger(Int64);
    case double(Double);
    case float32(Float);
    case string(String);
    case array(Array<JsonWireValue>);
    case object(JsonWireObject);

    public var isNull: Bool {
        if case .null = self {
            return true;
        }
        return false;
    }

    /// serde_json prints finite float32 values with shortest round-trip
    /// formatting; Swift's Float description matches for all values the
    /// protocol carries (non-finite floats are rejected by the writer).
    public var serializedText: String {
        get throws {
            var writer: JsonWireWriter = JsonWireWriter();
            try writer.appendValue(self);
            return writer.serializedText;
        }
    }
}

/// Insertion-ordered JSON object preserving the exact key order the Rust
/// serde derives emit, with rejection of duplicate keys while parsing.
public struct JsonWireObject: Equatable {
    public private(set) var entries: Array<(key: String, value: JsonWireValue)>;

    public init(entries: Array<(key: String, value: JsonWireValue)>) {
        self.entries = entries;
    }

    /// Content equality mirroring serde's map semantics: key order never
    /// participates in Value equality. Serialization stays insertion-ordered.
    public static func == (lhsValue: JsonWireObject, rhsValue: JsonWireObject) -> Bool {
        if lhsValue.entries.count != rhsValue.entries.count {
            return false;
        }
        for leftEntry in lhsValue.entries {
            guard let rightValue: JsonWireValue = rhsValue.value(forKey: leftEntry.key) else {
                return false;
            }
            if leftEntry.value != rightValue {
                return false;
            }
        }
        return true;
    }

    public var keyNames: Array<String> {
        return self.entries.map({ (entry: (key: String, value: JsonWireValue)) -> String in entry.key });
    }

    public func value(forKey propertyName: String) -> JsonWireValue? {
        for entry in self.entries {
            if entry.key == propertyName {
                return entry.value;
            }
        }
        return nil;
    }

    public mutating func appendEntry(key propertyName: String, value wireValue: JsonWireValue) {
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

    public static func formattedFieldList(_ fieldNames: Array<String>) -> String {
        return fieldNames.map({ (fieldName: String) -> String in "`\(fieldName)`" }).joined(separator: ", ");
    }
}

/// Typed extraction helpers shared by every wire-mapped payload type.
extension JsonWireObject {

    public func requireObjectValue(fieldName propertyName: String) throws -> JsonWireValue {
        guard let fieldValue = self.value(forKey: propertyName) else {
            throw JsonWireProblem.missingField(fieldName: propertyName);
        }
        return fieldValue;
    }

    public func decodeString(fieldName propertyName: String) throws -> String {
        return try JsonWireValue.extractString(try self.requireObjectValue(fieldName: propertyName));
    }

    public func decodeBool(fieldName propertyName: String) throws -> Bool {
        return try JsonWireValue.extractBool(try self.requireObjectValue(fieldName: propertyName));
    }

    public func decodeArray<Element>(fieldName propertyName: String, mappedElement: (JsonWireValue) throws -> Element) throws -> Array<Element> {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        return try JsonWireValue.extractArray(fieldValue, mappedElement: mappedElement);
    }

    public func decodeArrayAllowingAbsent<Element>(fieldName propertyName: String, mappedElement: (JsonWireValue) throws -> Element) throws -> Array<Element> {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return Array<Element>();
        }
        return try JsonWireValue.extractArray(fieldValue, mappedElement: mappedElement);
    }

    public func decodeOptionalStringAllowingAbsent(fieldName propertyName: String) throws -> String? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractString(fieldValue);
    }

    public func decodeOptionalUInt16AllowingAbsent(fieldName propertyName: String) throws -> UInt16? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.clampToUInt16(try JsonWireValue.extractUInt64(fieldValue));
    }

    public func decodeOptionalUInt32AllowingAbsent(fieldName propertyName: String) throws -> UInt32? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.clampToUInt32(try JsonWireValue.extractUInt64(fieldValue));
    }

    public func decodeOptionalUInt64AllowingAbsent(fieldName propertyName: String) throws -> UInt64? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractUInt64(fieldValue);
    }

    public func decodeOptionalObjectAllowingAbsent(fieldName propertyName: String) throws -> JsonWireObject? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractObject(fieldValue);
    }

    public func decodeOptionalRawValueAllowingAbsent(fieldName propertyName: String) throws -> JsonWireValue? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return fieldValue;
    }

    public func decodeBoolAllowingAbsent(fieldName propertyName: String) throws -> Bool {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return false;
        }
        return try JsonWireValue.extractBool(fieldValue);
    }

    public func decodeOptionalBoolAllowingAbsent(fieldName propertyName: String) throws -> Bool? {
        guard let fieldValue = self.value(forKey: propertyName) else {
            return nil;
        }
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractBool(fieldValue);
    }

    /// Reads the variant tag of an internally tagged enum payload and rejects
    /// unrecognized variant names with serde's wording.
    public func decodeTaggedVariantName(tagFieldName: String, expectedVariantNames: Array<String>) throws -> String {
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
    public func rejectUnknownFieldsBesidesTag(tagFieldName: String, allowedFieldNames: Array<String>) throws {
        for propertyName in self.keyNames {
            if propertyName != tagFieldName && allowedFieldNames.contains(propertyName) == false {
                throw JsonWireProblem.unknownField(fieldName: propertyName, expectedFields: allowedFieldNames);
            }
        }
    }

    public func decodeOptionalString(fieldName propertyName: String) throws -> String? {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractString(fieldValue);
    }

    public func decodeUInt64(fieldName propertyName: String) throws -> UInt64 {
        return try JsonWireValue.extractUInt64(try self.requireObjectValue(fieldName: propertyName));
    }

    public func decodeOptionalUInt64(fieldName propertyName: String) throws -> UInt64? {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractUInt64(fieldValue);
    }

    public func decodeUInt32(fieldName propertyName: String) throws -> UInt32 {
        return try JsonWireValue.clampToUInt32(try self.decodeUInt64(fieldName: propertyName));
    }

    public func decodeOptionalUInt32(fieldName propertyName: String) throws -> UInt32? {
        guard let unwrappedValue = try self.decodeOptionalUInt64(fieldName: propertyName) else {
            return nil;
        }
        return try JsonWireValue.clampToUInt32(unwrappedValue);
    }

    public func decodeUInt16(fieldName propertyName: String) throws -> UInt16 {
        return try JsonWireValue.clampToUInt16(try self.decodeUInt64(fieldName: propertyName));
    }

    public func decodeUInt8(fieldName propertyName: String) throws -> UInt8 {
        return try JsonWireValue.clampToUInt8(try self.decodeUInt64(fieldName: propertyName));
    }

    public func decodeOptionalUInt8(fieldName propertyName: String) throws -> UInt8? {
        guard let unwrappedValue = try self.decodeOptionalUInt64(fieldName: propertyName) else {
            return nil;
        }
        return try JsonWireValue.clampToUInt8(unwrappedValue);
    }

    public func decodeOptionalUInt16(fieldName propertyName: String) throws -> UInt16? {
        guard let unwrappedValue = try self.decodeOptionalUInt64(fieldName: propertyName) else {
            return nil;
        }
        return try JsonWireValue.clampToUInt16(unwrappedValue);
    }

    public func decodeInt64(fieldName propertyName: String) throws -> Int64 {
        return try JsonWireValue.extractInt64(try self.requireObjectValue(fieldName: propertyName));
    }

    public func decodeOptionalInt64(fieldName propertyName: String) throws -> Int64? {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractInt64(fieldValue);
    }

    public func decodeObject(fieldName propertyName: String) throws -> JsonWireObject {
        return try JsonWireValue.extractObject(try self.requireObjectValue(fieldName: propertyName));
    }

    public func decodeOptionalObject(fieldName propertyName: String) throws -> JsonWireObject? {
        let fieldValue = try self.requireObjectValue(fieldName: propertyName);
        if fieldValue.isNull {
            return nil;
        }
        return try JsonWireValue.extractObject(fieldValue);
    }

    /// Rejects keys outside the variant's field set after the tagged payload
    /// decoded, mirroring serde deny_unknown_fields.
    public func rejectUnknownFields(allowedFieldNames: Array<String>) throws {
        for propertyName in self.keyNames {
            if allowedFieldNames.contains(propertyName) == false {
                throw JsonWireProblem.unknownField(fieldName: propertyName, expectedFields: allowedFieldNames);
            }
        }
    }
}

extension JsonWireValue {

    public static func extractString(_ wireValue: JsonWireValue) throws -> String {
        guard case let .string(textValue) = wireValue else {
            throw JsonWireProblem.expectedString(found: wireValue.foundDescription);
        }
        return textValue;
    }

    public static func extractUInt64(_ wireValue: JsonWireValue) throws -> UInt64 {
        switch wireValue {
        case let .unsignedInteger(numericValue): return numericValue;
        default: throw JsonWireProblem.invalidType(expectedTypeName: "u64", found: wireValue.foundDescription);
        }
    }

    public static func extractInt64(_ wireValue: JsonWireValue) throws -> Int64 {
        switch wireValue {
        case let .signedInteger(numericValue): return numericValue;
        default: throw JsonWireProblem.invalidType(expectedTypeName: "i64", found: wireValue.foundDescription);
        }
    }

    public static func extractBool(_ wireValue: JsonWireValue) throws -> Bool {
        guard case let .boolean(booleanValue) = wireValue else {
            throw JsonWireProblem.invalidType(expectedTypeName: "a boolean", found: wireValue.foundDescription);
        }
        return booleanValue;
    }

    public static func extractArray<Element>(_ wireValue: JsonWireValue, mappedElement: (JsonWireValue) throws -> Element) throws -> Array<Element> {        guard case let .array(elementValues) = wireValue else {
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

    public static func extractObject(_ wireValue: JsonWireValue) throws -> JsonWireObject {
        guard case let .object(objectValue) = wireValue else {
            throw JsonWireProblem.expectedObject(found: wireValue.foundDescription);
        }
        return objectValue;
    }

    public static func clampToUInt32(_ rawValue: UInt64) throws -> UInt32 {
        guard rawValue <= UInt64(UInt32.max) else {
            throw JsonWireProblem.invalidType(expectedTypeName: "u32", found: "integer `\(rawValue)`");
        }
        return UInt32(rawValue);
    }

    public static func clampToUInt16(_ rawValue: UInt64) throws -> UInt16 {
        guard rawValue <= UInt64(UInt16.max) else {
            throw JsonWireProblem.invalidType(expectedTypeName: "u16", found: "integer `\(rawValue)`");
        }
        return UInt16(rawValue);
    }

    public static func clampToUInt8(_ rawValue: UInt64) throws -> UInt8 {
        guard rawValue <= UInt64(UInt8.max) else {
            throw JsonWireProblem.invalidType(expectedTypeName: "u8", found: "integer `\(rawValue)`");
        }
        return UInt8(rawValue);
    }

    /// serde_json accepts every JSON number shape when deserializing an f32
    /// field: integer tokens narrow losslessly and float tokens cast, and an
    /// out-of-range token fails instead of becoming infinity.
    public static func extractFloat32(_ wireValue: JsonWireValue) throws -> Float {
        let convertedValue: Float;
        switch wireValue {
        case let .float32(numericValue): convertedValue = numericValue;
        case let .double(numericValue): convertedValue = Float(numericValue);
        case let .unsignedInteger(numericValue): convertedValue = Float(numericValue);
        case let .signedInteger(numericValue): convertedValue = Float(numericValue);
        default: throw JsonWireProblem.invalidType(expectedTypeName: "f32", found: wireValue.foundDescription);
        }
        guard convertedValue.isFinite else {
            throw JsonWireProblem.invalidType(
                expectedTypeName: "f32", found: "floating point `\(wireValue.foundDescription)`");
        }
        return convertedValue;
    }

    /// Mirrors serde_json f64 deserialization across every number shape.
    public static func extractFloat64(_ wireValue: JsonWireValue) throws -> Double {
        let convertedValue: Double;
        switch wireValue {
        case let .double(numericValue): convertedValue = numericValue;
        case let .float32(numericValue): convertedValue = Double(numericValue);
        case let .unsignedInteger(numericValue): convertedValue = Double(numericValue);
        case let .signedInteger(numericValue): convertedValue = Double(numericValue);
        default: throw JsonWireProblem.invalidType(expectedTypeName: "f64", found: wireValue.foundDescription);
        }
        guard convertedValue.isFinite else {
            throw JsonWireProblem.invalidType(
                expectedTypeName: "f64", found: "floating point `\(wireValue.foundDescription)`");
        }
        return convertedValue;
    }

    public var foundDescription: String {
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

    public static func mappedArray<Element>(_ values: Array<Element>, mappedWireValue: (Element) -> JsonWireValue) -> JsonWireValue {
        var elementWireValues = Array<JsonWireValue>();
        elementWireValues.reserveCapacity(values.count);
        for elementValue in values {
            elementWireValues.append(mappedWireValue(elementValue));
        }
        return .array(elementWireValues);
    }

    public static func stringArray(_ values: Array<String>) -> JsonWireValue {
        return mappedArray(values, mappedWireValue: { (textValue: String) -> JsonWireValue in
            return .string(textValue);
        });
    }
}

extension String {
    internal var serializedJsonQuoted: String {
        var writer: JsonWireWriter = JsonWireWriter();
        writer.appendStringValue(self);
        return writer.serializedText;
    }
}
