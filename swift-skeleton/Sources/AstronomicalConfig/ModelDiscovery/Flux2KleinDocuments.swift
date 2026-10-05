import Foundation;

/**
 * Strict deserialization documents for reviewed FLUX.2 Klein package evidence,
 * porting crates/config/src/model_discovery/flux2_klein_documents.rs. Keeping
 * wire-shaped JSON documents separate leaves discovery focused on validation
 * policy and filesystem evidence rather than serialization mechanics.
 *
 * The Rust documents rely on serde's default of ignoring unknown fields (no
 * `deny_unknown_fields` on any struct here), so `requireKnownKeys` is
 * deliberately not called; only declared fields are extracted.
 */
internal struct PipelineClass: Equatable, Sendable {
    internal let className: String?;

    internal init(className: String?) {
        self.className = className;
    }

    /** `_class_name` defaults to absent in serde, so a missing key decodes to nil. */
    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> PipelineClass {
        return PipelineClass(
            className: try StrictJson.optionalString(object: jsonObject, fieldName: "_class_name")
        );
    }
}

internal struct PipelineIndex: Equatable, Sendable {
    internal let className: String;
    internal let isDistilled: Bool;
    internal let scheduler: Array<String>;
    internal let textEncoder: Array<String>;
    internal let tokenizer: Array<String>;
    internal let transformer: Array<String>;
    internal let vae: Array<String>;

    internal init(
        className: String,
        isDistilled: Bool,
        scheduler: Array<String>,
        textEncoder: Array<String>,
        tokenizer: Array<String>,
        transformer: Array<String>,
        vae: Array<String>
    ) {
        self.className = className;
        self.isDistilled = isDistilled;
        self.scheduler = scheduler;
        self.textEncoder = textEncoder;
        self.tokenizer = tokenizer;
        self.transformer = transformer;
        self.vae = vae;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> PipelineIndex {
        return PipelineIndex(
            className: try StrictJson.requiredString(object: jsonObject, fieldName: "_class_name"),
            isDistilled: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "is_distilled"),
            scheduler: try Flux2KleinWireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "scheduler",
                exactElementCount: 2
            ),
            textEncoder: try Flux2KleinWireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "text_encoder",
                exactElementCount: 2
            ),
            tokenizer: try Flux2KleinWireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "tokenizer",
                exactElementCount: 2
            ),
            transformer: try Flux2KleinWireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "transformer",
                exactElementCount: 2
            ),
            vae: try Flux2KleinWireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "vae",
                exactElementCount: 2
            )
        );
    }
}

internal struct TextEncoderIndex: Equatable, Sendable {
    internal let metadata: TextEncoderIndexMetadata;
    internal let weightMap: Dictionary<String, String>;

    internal init(metadata: TextEncoderIndexMetadata, weightMap: Dictionary<String, String>) {
        self.metadata = metadata;
        self.weightMap = weightMap;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> TextEncoderIndex {
        let metadataObject: Dictionary<String, Any> = try StrictJson.objectValue(object: jsonObject, fieldName: "metadata");
        return TextEncoderIndex(
            metadata: try TextEncoderIndexMetadata.fromJsonObject(metadataObject),
            weightMap: try Flux2KleinWireJson.requiredStringDictionary(object: jsonObject, fieldName: "weight_map")
        );
    }
}

internal struct TextEncoderIndexMetadata: Equatable, Sendable {
    internal let totalSize: UInt64;

    internal init(totalSize: UInt64) {
        self.totalSize = totalSize;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> TextEncoderIndexMetadata {
        return TextEncoderIndexMetadata(
            totalSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "total_size")
        );
    }
}

/**
 * serde's `Option<ValueMarker>` from flux2_klein_documents.rs: the payload is
 * never read, only its presence, so any non-null value decodes as present and
 * null decodes as absent. A present marker always fails the reviewed-profile
 * check exactly like the Rust parse-or-check outcome does.
 */
internal struct ValueMarker: Equatable, Sendable {
    internal init() {
    }
}

/**
 * Private extraction helpers this document set needs that StrictJson does not
 * offer yet: f64 fields and fixed-size numeric arrays. A future port should
 * generalize them into StrictJson next to its string counterparts.
 */
internal enum Flux2KleinWireJson {
    internal static func requiredDouble(object: Dictionary<String, Any>, fieldName: String) throws -> Double {
        guard let presentValue: Any = object[fieldName] else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be present");
        }
        guard let numberValue: NSNumber = presentValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a number");
        }
        if (Flux2KleinWireJson.isBoolean(numberValue)) {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a number, got a boolean");
        }
        return numberValue.doubleValue;
    }

    internal static func optionalDouble(object: Dictionary<String, Any>, fieldName: String) throws -> Double? {
        guard let presentValue: Any = object[fieldName] else {
            return nil;
        }
        if (presentValue is NSNull) {
            return nil;
        }
        return try Flux2KleinWireJson.requiredDouble(object: jsonObjectMaskingAbsence(presentValue), fieldName: fieldName);
    }

    internal static func requiredFixedStringArray(
        object: Dictionary<String, Any>,
        fieldName: String,
        exactElementCount: Int
    ) throws -> Array<String> {
        let stringElements: Array<String> = try StrictJson.requiredStringArray(object: object, fieldName: fieldName);
        guard stringElements.count == exactElementCount else {
            throw StrictJsonError(
                fieldName: fieldName,
                problem: "must be an array of exactly " + String(exactElementCount) + " strings"
            );
        }
        return stringElements;
    }

    internal static func requiredFixedUnsignedIntegerArray<UnsignedIntegerField: FixedWidthInteger>(
        object: Dictionary<String, Any>,
        fieldName: String,
        exactElementCount: Int
    ) throws -> Array<UnsignedIntegerField> {
        guard let arrayValue: Array<Any> = object[fieldName] as? Array<Any> else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an array of unsigned integers");
        }
        guard arrayValue.count == exactElementCount else {
            throw StrictJsonError(
                fieldName: fieldName,
                problem: "must be an array of exactly " + String(exactElementCount) + " unsigned integers"
            );
        }
        var integerElements: Array<UnsignedIntegerField> = Array<UnsignedIntegerField>();
        for (offset: elementIndex, element: elementValue) in arrayValue.enumerated() {
            let elementFieldName: String = fieldName + "[" + String(elementIndex) + "]";
            integerElements.append(try Flux2KleinWireJson.requiredUnsignedIntegerElement(
                elementValue: elementValue,
                fieldName: elementFieldName
            ));
        }
        return integerElements;
    }

    internal static func requiredStringDictionary(
        object: Dictionary<String, Any>,
        fieldName: String
    ) throws -> Dictionary<String, String> {
        guard let dictionaryValue: Dictionary<String, Any> = object[fieldName] as? Dictionary<String, Any> else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an object of strings");
        }
        var stringDictionary: Dictionary<String, String> = Dictionary<String, String>();
        for (key: entryKeyName, value: entryValue) in dictionaryValue {
            guard let entryString: String = entryValue as? String else {
                throw StrictJsonError(fieldName: fieldName + "." + entryKeyName, problem: "must be a string");
            }
            stringDictionary[entryKeyName] = entryString;
        }
        return stringDictionary;
    }

    /**
     * Option<ValueMarker> semantics: absent and JSON null mean absent; every
     * other present value means present, because the marker carries no data.
     */
    internal static func optionalValueMarker(object: Dictionary<String, Any>, fieldName: String) throws -> ValueMarker? {
        guard let presentValue: Any = object[fieldName] else {
            return nil;
        }
        if (presentValue is NSNull) {
            return nil;
        }
        return ValueMarker();
    }

    private static func jsonObjectMaskingAbsence(_ presentValue: Any) -> Dictionary<String, Any> {
        var maskedObject: Dictionary<String, Any> = Dictionary<String, Any>();
        maskedObject["value"] = presentValue;
        return maskedObject;
    }

    private static func requiredUnsignedIntegerElement<UnsignedIntegerField: FixedWidthInteger>(
        elementValue: Any,
        fieldName: String
    ) throws -> UnsignedIntegerField {
        guard let numberValue: NSNumber = elementValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an unsigned integer");
        }
        if (Flux2KleinWireJson.isBoolean(numberValue)) {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an unsigned integer, got a boolean");
        }
        // Int64 round-trip equality rejects fractions and values outside the
        // Int64 range, mirroring StrictJson's scalar unsigned extraction.
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

    private static func isBoolean(_ numberValue: NSNumber) -> Bool {
        return CFGetTypeID(numberValue as CFTypeRef) == CFBooleanGetTypeID();
    }
}
