import Foundation;

/**
 * Failure to extract one strictly typed field from the untyped JSON object
 * tree. The config-file store maps this to its structured parse error.
 */
internal struct StrictJsonError: Error {
    internal let fieldName: String;
    internal let problem: String;

    internal init(fieldName: String, problem: String) {
        self.fieldName = fieldName;
        self.problem = problem;
    }
}

/**
 * Strict JSON value extractors over JSONSerialization output.
 *
 * Codable's KeyedDecodingContainer cannot observe unknown keys, so the
 * deny_unknown_fields contract of the v1 configuration document is enforced
 * against JSONSerialization's untyped object tree instead. Every extractor
 * either returns the declared type or throws StrictJsonError; NSNull counts
 * as an absent optional, matching how serde decodes null into None.
 */
internal enum StrictJson {
    internal static func requireKnownKeys(
        object: Dictionary<String, Any>,
        knownKeys: Set<String>,
        fieldName: String
    ) throws -> Void {
        for presentKeyName: String in object.keys {
            guard (knownKeys.contains(presentKeyName)) else {
                let fieldPath: String = fieldName.isEmpty ? presentKeyName : fieldName + "." + presentKeyName;
                throw StrictJsonError(fieldName: fieldPath, problem: "unknown field");
            }
        }
    }

    internal static func requiredString(object: Dictionary<String, Any>, fieldName: String) throws -> String {
        guard let stringValue: String = object[fieldName] as? String else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a string");
        }
        return stringValue;
    }

    internal static func optionalString(object: Dictionary<String, Any>, fieldName: String) throws -> String? {
        guard let presentValue: Any = object[fieldName] else {
            return nil;
        }
        if (presentValue is NSNull) {
            return nil;
        }
        guard let stringValue: String = presentValue as? String else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a string");
        }
        return stringValue;
    }

    internal static func requiredUnsignedInteger<UnsignedIntegerField: FixedWidthInteger>(
        object: Dictionary<String, Any>,
        fieldName: String
    ) throws -> UnsignedIntegerField {
        let numberValue: NSNumber = try StrictJson.requireNumber(object: object, fieldName: fieldName);
        return try StrictJson.unsignedIntegerValue(of: numberValue, fieldName: fieldName);
    }

    internal static func optionalUnsignedInteger<UnsignedIntegerField: FixedWidthInteger>(
        object: Dictionary<String, Any>,
        fieldName: String
    ) throws -> UnsignedIntegerField? {
        guard let presentValue: Any = object[fieldName] else {
            return nil;
        }
        if (presentValue is NSNull) {
            return nil;
        }
        guard let numberValue: NSNumber = presentValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an unsigned integer");
        }
        return try StrictJson.unsignedIntegerValue(of: numberValue, fieldName: fieldName);
    }

    internal static func optionalSignedInteger<SignedIntegerField: FixedWidthInteger>(
        object: Dictionary<String, Any>,
        fieldName: String
    ) throws -> SignedIntegerField? {
        guard let presentValue: Any = object[fieldName] else {
            return nil;
        }
        if (presentValue is NSNull) {
            return nil;
        }
        guard let numberValue: NSNumber = presentValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an integer");
        }
        return try StrictJson.signedIntegerValue(of: numberValue, fieldName: fieldName);
    }

    internal static func requiredBoolean(object: Dictionary<String, Any>, fieldName: String) throws -> Bool {
        let numberValue: NSNumber = try StrictJson.requireNumber(object: object, fieldName: fieldName);
        return try StrictJson.booleanValue(of: numberValue, fieldName: fieldName);
    }

    internal static func optionalBoolean(object: Dictionary<String, Any>, fieldName: String) throws -> Bool? {
        guard let presentValue: Any = object[fieldName] else {
            return nil;
        }
        if (presentValue is NSNull) {
            return nil;
        }
        guard let numberValue: NSNumber = presentValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a boolean");
        }
        return try StrictJson.booleanValue(of: numberValue, fieldName: fieldName);
    }

    internal static func optionalFloat(object: Dictionary<String, Any>, fieldName: String) throws -> Float? {
        guard let presentValue: Any = object[fieldName] else {
            return nil;
        }
        if (presentValue is NSNull) {
            return nil;
        }
        guard let numberValue: NSNumber = presentValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a number");
        }
        if (StrictJson.isBoolean(numberValue)) {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a number, got a boolean");
        }
        // serde_json rounds f32 fields to the nearest representable value but
        // rejects values that would overflow to infinity.
        let floatValue: Float = Float(numberValue.doubleValue);
        if (floatValue.isInfinite) {
            throw StrictJsonError(fieldName: fieldName, problem: "number is out of range for a 32-bit float");
        }
        return floatValue;
    }

    internal static func requiredStringArray(object: Dictionary<String, Any>, fieldName: String) throws -> Array<String> {
        guard let arrayValue: Array<Any> = object[fieldName] as? Array<Any> else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an array of strings");
        }
        var stringValues: Array<String> = Array<String>();
        for (offset: elementIndex, element: elementValue) in arrayValue.enumerated() {
            guard let elementString: String = elementValue as? String else {
                throw StrictJsonError(
                    fieldName: fieldName + "[" + String(elementIndex) + "]",
                    problem: "must be a string"
                );
            }
            stringValues.append(elementString);
        }
        return stringValues;
    }

    internal static func objectValue(object: Dictionary<String, Any>, fieldName: String) throws -> Dictionary<String, Any> {
        guard let dictionaryValue: Dictionary<String, Any> = object[fieldName] as? Dictionary<String, Any> else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an object");
        }
        return dictionaryValue;
    }

    internal static func optionalObject(object: Dictionary<String, Any>, fieldName: String) throws -> Dictionary<String, Any>? {
        guard let presentValue: Any = object[fieldName] else {
            return nil;
        }
        if (presentValue is NSNull) {
            return nil;
        }
        guard let dictionaryValue: Dictionary<String, Any> = presentValue as? Dictionary<String, Any> else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an object");
        }
        return dictionaryValue;
    }

    /**
     * Decodes an optional nested object with the section's own decoder,
     * keeping the absent-null / present-object decision at the call site the
     * same way serde's Option<T> deserialization behaves.
     */
    internal static func decodeOptional<DecodedValue>(
        _ jsonObject: Dictionary<String, Any>?,
        decode: (Dictionary<String, Any>) throws -> DecodedValue
    ) throws -> DecodedValue? {
        guard let presentObject: Dictionary<String, Any> = jsonObject else {
            return nil;
        }
        return try decode(presentObject);
    }

    private static func requireNumber(object: Dictionary<String, Any>, fieldName: String) throws -> NSNumber {
        guard let presentValue: Any = object[fieldName] else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be present");
        }
        guard let numberValue: NSNumber = presentValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a number");
        }
        return numberValue;
    }

    private static func unsignedIntegerValue<UnsignedIntegerField: FixedWidthInteger>(
        of numberValue: NSNumber,
        fieldName: String
    ) throws -> UnsignedIntegerField {
        if (StrictJson.isBoolean(numberValue)) {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an unsigned integer, got a boolean");
        }
        // Int64 round-trip equality rejects fractions and values outside the
        // Int64 range, which NSNumber would otherwise silently clamp.
        let integerValue: Int64 = numberValue.int64Value;
        guard (NSNumber(value: integerValue) == numberValue) else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an unsigned integer");
        }
        guard (integerValue >= 0) else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an unsigned integer, got a negative number");
        }
        guard let convertedValue: UnsignedIntegerField = UnsignedIntegerField(exactly: integerValue) else {
            throw StrictJsonError(fieldName: fieldName, problem: "unsigned integer is out of range for this field");
        }
        return convertedValue;
    }

    private static func signedIntegerValue<SignedIntegerField: FixedWidthInteger>(
        of numberValue: NSNumber,
        fieldName: String
    ) throws -> SignedIntegerField {
        if (StrictJson.isBoolean(numberValue)) {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an integer, got a boolean");
        }
        let integerValue: Int64 = numberValue.int64Value;
        guard (NSNumber(value: integerValue) == numberValue) else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an integer");
        }
        guard let convertedValue: SignedIntegerField = SignedIntegerField(exactly: integerValue) else {
            throw StrictJsonError(fieldName: fieldName, problem: "integer is out of range for this field");
        }
        return convertedValue;
    }

    private static func booleanValue(of numberValue: NSNumber, fieldName: String) throws -> Bool {
        // NSNumber equality considers booleans and 0/1 integers equal, so
        // booleans are identified by their Core Foundation type first.
        guard (StrictJson.isBoolean(numberValue)) else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a boolean");
        }
        return numberValue.boolValue;
    }

    private static func isBoolean(_ numberValue: NSNumber) -> Bool {
        return CFGetTypeID(numberValue as CFTypeRef) == CFBooleanGetTypeID();
    }
}
