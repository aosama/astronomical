import Foundation;
import IpcProtocol;

/// A config-decoding or implemented-execution-contract failure for Qwen3.5.
/// Port of Qwen3_5ConfigError from
/// crates/model-serving/src/qwen3_5/configuration/config_validation.rs.
public enum Qwen3_5ConfigError: Error, Equatable {
    /// The retained config bytes were not valid JSON for the expected document shape.
    case deserializeConfig(problem: String);
    /// A required string field selected behavior this executor does not implement.
    case unexpectedStringValue(fieldName: String, expectedValue: String, actualValue: String);
    /// A required boolean field selected behavior this executor does not implement.
    case unexpectedBooleanValue(fieldName: String, expectedValue: Bool, actualValue: Bool);
    /// The decoder did not declare exactly one attention kind for every text layer.
    case layerTypeCountMismatch(actualLayerTypeCount: Int, expectedLayerTypeCount: Int);
    /// An affine override used a bit width unsupported by MLX.
    case unsupportedQuantizationOverrideBits(moduleName: String, actualValue: UInt32);
    /// The multimodal rotary section differed from the pinned text configuration.
    case mropeSectionMismatch(actualSection: Array<UInt32>);
    /// The duplicated MLX quantization documents did not match exactly.
    case quantizationCopiesDiffer;
    /// The retained config bytes were not valid JSON for the expected vision_config shape.
    case deserializeVisionConfig(problem: String);
    case missingVisionConfig;
    /// A structural sanity check failed (e.g., non-positive dimension, inconsistent counts).
    case invalidConfigValue(description: String);
    /// A structural sanity check failed with a dynamic description.
    case invalidConfigValueDynamic(description: String);
    /// The activation dtype (`dtype` or `torch_dtype`) was absent from both the
    /// top level and `text_config`. At least one must be present.
    case missingActivationDtype;
    /// The `eos_token_id` field was absent from both the top level and `text_config`.
    /// At least one must be present so the model can identify stop tokens.
    case missingEosTokenId;

    public var errorDescription: String? {
        switch self {
        case .deserializeConfig:
            return "failed to decode Qwen3.5 config JSON";
        case .unexpectedStringValue(let fieldName, let expectedValue, let actualValue):
            return "Qwen3.5 config field '\(fieldName)' must be '\(expectedValue)', got '\(actualValue)'";
        case .unexpectedBooleanValue(let fieldName, let expectedValue, let actualValue):
            return "Qwen3.5 config field '\(fieldName)' must be \(expectedValue), got \(actualValue)";
        case .layerTypeCountMismatch(let actualLayerTypeCount, let expectedLayerTypeCount):
            return "Qwen3.5 config declares \(actualLayerTypeCount) layer types, expected \(expectedLayerTypeCount)";
        case .unsupportedQuantizationOverrideBits(let moduleName, let actualValue):
            return "module '\(moduleName)' uses unsupported \(actualValue)-bit affine quantization";
        case .mropeSectionMismatch(let actualSection):
            return "Qwen3.5 mrope section \(rustDebugArrayText(actualSection)) differs from [11, 11, 10]";
        case .quantizationCopiesDiffer:
            return "Qwen3.5 quantization and quantization_config fields differ";
        case .deserializeVisionConfig:
            return "failed to decode Qwen3.5 vision config JSON";
        case .missingVisionConfig:
            return "Qwen3.5 config does not declare vision_config";
        case .invalidConfigValue(let description):
            return "invalid Qwen3.5 config: \(description)";
        case .invalidConfigValueDynamic(let description):
            return "invalid Qwen3.5 config: \(description)";
        case .missingActivationDtype:
            return "Qwen3.5 config must specify `dtype` at the top level or inside `text_config`";
        case .missingEosTokenId:
            return "Qwen3.5 config must specify `eos_token_id` at the top level or inside `text_config`";
        }
    }

    /// Reproduces Rust's `{:?}` slice rendering, e.g. `[11, 11, 10]`.
    private func rustDebugArrayText(_ values: Array<UInt32>) -> String {
        return "[\(values.map({ (value: UInt32) -> String in String(value) }).joined(separator: ", "))]";
    }
}

/// Config validation helpers shared by the document and quantization layers,
/// port of the free functions in config_validation.rs.
public enum Qwen3_5ConfigValidation {

    /// The standard Qwen chat end-of-sequence token ID.
    /// When `eos_token_id` is absent at the top level and resolved from `text_config`,
    /// this token is appended to the list if not already present.
    public static let QWEN_CHAT_EOS_TOKEN_ID: UInt32 = 248046;

    public static func validateExactValue(
        fieldName: String, actualValue: String, expectedValue: String) throws -> Void {
        if actualValue == expectedValue {
            return;
        }
        throw Qwen3_5ConfigError.unexpectedStringValue(
            fieldName: fieldName, expectedValue: expectedValue, actualValue: actualValue);
    }

    public static func validateExactBoolean(
        fieldName: String, actualValue: Bool, expectedValue: Bool) throws -> Void {
        if actualValue == expectedValue {
            return;
        }
        throw Qwen3_5ConfigError.unexpectedBooleanValue(
            fieldName: fieldName, expectedValue: expectedValue, actualValue: actualValue);
    }

    /// Converts a JSON number into exact IEEE-754 float32 bits, mirroring the
    /// Rust `deserialize_f32_bits` serde helper. serde_json rejects
    /// out-of-range and non-numeric tokens before the conversion.
    public static func float32Bits(fromNumber wireValue: JsonWireValue) throws -> UInt32 {
        return try JsonWireValue.extractFloat32(wireValue).bitPattern;
    }

    public static func optionalFloat32Bits(fromNumber wireValue: JsonWireValue) throws -> UInt32? {
        if wireValue.isNull {
            return nil;
        }
        return try float32Bits(fromNumber: wireValue);
    }

    /// Decodes `eos_token_id` from either a single integer or an array of
    /// integers; a missing field decodes to nil.
    public static func optionalEosTokenIds(fromObject objectValue: JsonWireObject, fieldName propertyName: String) throws -> Array<UInt32>? {
        guard let fieldValue: JsonWireValue = try objectValue.decodeOptionalRawValueAllowingAbsent(fieldName: propertyName) else {
            return nil;
        }
        if case let .unsignedInteger(singleTokenValue) = fieldValue {
            return [try JsonWireValue.clampToUInt32(singleTokenValue)];
        }
        if case let .array(tokenValues) = fieldValue {
            return try tokenValues.map({ (tokenWireValue: JsonWireValue) throws -> UInt32 in
                return try tokenWireValue.uint32TokenId();
            });
        }
        throw JsonWireProblem.invalidType(
            expectedTypeName: "u32 or a sequence", found: fieldValue.foundDescription);
    }
}

extension JsonWireValue {

    /// serde narrows a JSON number token into u32 exactly when it is a
    /// non-negative integer within range; every other shape is a type error.
    internal func uint32TokenId() throws -> UInt32 {
        switch self {
        case let .unsignedInteger(numericValue):
            return try JsonWireValue.clampToUInt32(numericValue);
        default:
            throw JsonWireProblem.invalidType(expectedTypeName: "u32", found: self.foundDescription);
        }
    }
}
