import Foundation;

/**
 * Wire documents for the reviewed Qwen-Image-2.1 MLX package, porting
 * crates/config/src/model_discovery/qwen_image_21_documents.rs. Field names
 * intentionally mirror the Hugging Face serialization, including its
 * upstream `temperal_downsample` spelling. Discovery owns profile policy;
 * this file owns only JSON shape.
 *
 * The Rust documents rely on serde's default of ignoring unknown fields (no
 * `deny_unknown_fields` on any struct here), so `requireKnownKeys` is
 * deliberately not called; only declared fields are extracted.
 */
internal struct QwenImage21PipelineClass: Equatable, Sendable {
    internal let className: String?;

    internal init(className: String?) {
        self.className = className;
    }

    /** `_class_name` defaults to absent in serde, so a missing key decodes to nil. */
    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> QwenImage21PipelineClass {
        return QwenImage21PipelineClass(
            className: try StrictJson.optionalString(object: jsonObject, fieldName: "_class_name")
        );
    }
}

internal struct QwenImage21PipelineIndex: Equatable, Sendable {
    internal let className: String;
    internal let processor: Array<String>;
    internal let scheduler: Array<String>;
    internal let textEncoder: Array<String>;
    internal let transformer: Array<String>;
    internal let vae: Array<String>;

    internal init(
        className: String,
        processor: Array<String>,
        scheduler: Array<String>,
        textEncoder: Array<String>,
        transformer: Array<String>,
        vae: Array<String>
    ) {
        self.className = className;
        self.processor = processor;
        self.scheduler = scheduler;
        self.textEncoder = textEncoder;
        self.transformer = transformer;
        self.vae = vae;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> QwenImage21PipelineIndex {
        return QwenImage21PipelineIndex(
            className: try StrictJson.requiredString(object: jsonObject, fieldName: "_class_name"),
            processor: try QwenImage21WireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "processor",
                exactElementCount: 2
            ),
            scheduler: try QwenImage21WireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "scheduler",
                exactElementCount: 2
            ),
            textEncoder: try QwenImage21WireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "text_encoder",
                exactElementCount: 2
            ),
            transformer: try QwenImage21WireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "transformer",
                exactElementCount: 2
            ),
            vae: try QwenImage21WireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "vae",
                exactElementCount: 2
            )
        );
    }
}
internal struct ComponentSafetensorsIndex: Equatable, Sendable {
    internal let metadata: ComponentSafetensorsIndexMetadata;
    internal let weightMap: Dictionary<String, String>;

    internal init(metadata: ComponentSafetensorsIndexMetadata, weightMap: Dictionary<String, String>) {
        self.metadata = metadata;
        self.weightMap = weightMap;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> ComponentSafetensorsIndex {
        let metadataObject: Dictionary<String, Any> = try StrictJson.objectValue(object: jsonObject, fieldName: "metadata");
        return ComponentSafetensorsIndex(
            metadata: try ComponentSafetensorsIndexMetadata.fromJsonObject(metadataObject),
            weightMap: try QwenImage21WireJson.requiredStringDictionary(object: jsonObject, fieldName: "weight_map")
        );
    }
}

internal struct ComponentSafetensorsIndexMetadata: Equatable, Sendable {
    internal let totalSize: UInt64;

    internal init(totalSize: UInt64) {
        self.totalSize = totalSize;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> ComponentSafetensorsIndexMetadata {
        return ComponentSafetensorsIndexMetadata(
            totalSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "total_size")
        );
    }
}

/**
 * Private extraction helpers this document set needs that StrictJson does not
 * offer yet: f64 fields and arrays, fixed-size arrays of unsigned integers
 * and booleans, and string dictionaries. A future port should generalize
 * them into StrictJson next to its string counterparts.
 */
internal enum QwenImage21WireJson {
    internal static func requiredDouble(object: Dictionary<String, Any>, fieldName: String) throws -> Double {
        guard let presentValue: Any = object[fieldName] else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be present");
        }
        guard let numberValue: NSNumber = presentValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a number");
        }
        if (QwenImage21WireJson.isBoolean(numberValue)) {
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
        return try QwenImage21WireJson.requiredDouble(
            object: QwenImage21WireJson.jsonObjectMaskingAbsence(presentValue),
            fieldName: fieldName
        );
    }

    internal static func requiredDoubleArray(object: Dictionary<String, Any>, fieldName: String) throws -> Array<Double> {
        guard let arrayValue: Array<Any> = object[fieldName] as? Array<Any> else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an array of numbers");
        }
        var doubleElements: Array<Double> = Array<Double>();
        for (offset: elementIndex, element: elementValue) in arrayValue.enumerated() {
            let elementFieldName: String = fieldName + "[" + String(elementIndex) + "]";
            doubleElements.append(try QwenImage21WireJson.requiredDoubleElement(
                elementValue: elementValue,
                fieldName: elementFieldName
            ));
        }
        return doubleElements;
    }

    internal static func requiredUnsignedIntegerArray<UnsignedIntegerField: FixedWidthInteger>(
        object: Dictionary<String, Any>,
        fieldName: String
    ) throws -> Array<UnsignedIntegerField> {
        guard let arrayValue: Array<Any> = object[fieldName] as? Array<Any> else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an array of unsigned integers");
        }
        var integerElements: Array<UnsignedIntegerField> = Array<UnsignedIntegerField>();
        for (offset: elementIndex, element: elementValue) in arrayValue.enumerated() {
            let elementFieldName: String = fieldName + "[" + String(elementIndex) + "]";
            integerElements.append(try QwenImage21WireJson.unsignedIntegerElement(
                elementValue: elementValue,
                fieldName: elementFieldName
            ));
        }
        return integerElements;
    }

    internal static func requiredFixedUnsignedIntegerArray<UnsignedIntegerField: FixedWidthInteger>(
        object: Dictionary<String, Any>,
        fieldName: String,
        exactElementCount: Int
    ) throws -> Array<UnsignedIntegerField> {
        let integerElements: Array<UnsignedIntegerField> = try QwenImage21WireJson.requiredUnsignedIntegerArray(
            object: object,
            fieldName: fieldName
        );
        guard integerElements.count == exactElementCount else {
            throw StrictJsonError(
                fieldName: fieldName,
                problem: "must be an array of exactly " + String(exactElementCount) + " unsigned integers"
            );
        }
        return integerElements;
    }

    internal static func requiredFixedBooleanArray(
        object: Dictionary<String, Any>,
        fieldName: String,
        exactElementCount: Int
    ) throws -> Array<Bool> {
        guard let arrayValue: Array<Any> = object[fieldName] as? Array<Any> else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an array of booleans");
        }
        guard arrayValue.count == exactElementCount else {
            throw StrictJsonError(
                fieldName: fieldName,
                problem: "must be an array of exactly " + String(exactElementCount) + " booleans"
            );
        }
        var booleanElements: Array<Bool> = Array<Bool>();
        for (offset: elementIndex, element: elementValue) in arrayValue.enumerated() {
            let elementFieldName: String = fieldName + "[" + String(elementIndex) + "]";
            booleanElements.append(try QwenImage21WireJson.booleanElement(
                elementValue: elementValue,
                fieldName: elementFieldName
            ));
        }
        return booleanElements;
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

    private static func requiredDoubleElement(elementValue: Any, fieldName: String) throws -> Double {
        guard let numberValue: NSNumber = elementValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a number");
        }
        if (QwenImage21WireJson.isBoolean(numberValue)) {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a number, got a boolean");
        }
        return numberValue.doubleValue;
    }

    private static func unsignedIntegerElement<UnsignedIntegerField: FixedWidthInteger>(
        elementValue: Any,
        fieldName: String
    ) throws -> UnsignedIntegerField {
        guard let numberValue: NSNumber = elementValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be an unsigned integer");
        }
        if (QwenImage21WireJson.isBoolean(numberValue)) {
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

    private static func booleanElement(elementValue: Any, fieldName: String) throws -> Bool {
        guard let numberValue: NSNumber = elementValue as? NSNumber else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a boolean");
        }
        guard (QwenImage21WireJson.isBoolean(numberValue)) else {
            throw StrictJsonError(fieldName: fieldName, problem: "must be a boolean");
        }
        return numberValue.boolValue;
    }

    private static func jsonObjectMaskingAbsence(_ presentValue: Any) -> Dictionary<String, Any> {
        var maskedObject: Dictionary<String, Any> = Dictionary<String, Any>();
        maskedObject["value"] = presentValue;
        return maskedObject;
    }

    private static func isBoolean(_ numberValue: NSNumber) -> Bool {
        return CFGetTypeID(numberValue as CFTypeRef) == CFBooleanGetTypeID();
    }
}
