import Foundation;

/**
 * Family-owned shallow discovery rules for `qwen4_exp` artifacts, porting
 * crates/config/src/model_discovery/qwen4_exp.rs. The caseless enum is the
 * Swift equivalent of the Rust module of free functions. The family is
 * recognized so download preflight and diagnostics can name it, but it is
 * deliberately not executable and never advertised: no engine exists yet.
 * Reads stay bounded to the configuration document, and no rule here may
 * assume one artifact's packaging, because published checkpoints differ on
 * expert count, quantization profile, and lookup-table form.
 */
internal enum Qwen4Exp {

    /**
     * Recognizes the conditional-generation wrapper and a text-only
     * distribution of the same family.
     */
    internal static func recognizesModelType(_ modelType: String?) -> Bool {
        guard let presentModelType: String = modelType else {
            return false;
        }
        return presentModelType == "qwen4_exp" || presentModelType == "qwen4_exp_text";
    }

    /**
     * Reads the family's text configuration summary from a parsed config
     * document. Returns nil when the document lacks the nested text
     * configuration or any required field, which keeps a malformed artifact
     * from producing a partial summary that diagnostics could present as
     * authoritative.
     */
    internal static func describeConfiguration(configObject: Dictionary<String, Any>) -> DiscoveryQwen4ExpConfigurationSummary? {
        guard let textConfigObject: Dictionary<String, Any> = configObject["text_config"] as? Dictionary<String, Any> else {
            return nil;
        }
        guard let decoderLayersValue: Any = textConfigObject["num_hidden_layers"] else {
            return nil;
        }
        guard let decoderLayers: UInt32 = Qwen4Exp.boundedUnsigned32Value(of: decoderLayersValue) else {
            return nil;
        }
        guard let contextWindowTokensValue: Any = textConfigObject["max_position_embeddings"] else {
            return nil;
        }
        guard let contextWindowTokens: UInt64 = Qwen4Exp.unsignedInteger64Value(of: contextWindowTokensValue) else {
            return nil;
        }
        guard let routedExpertsValue: Any = textConfigObject["num_experts"] else {
            return nil;
        }
        guard let routedExperts: UInt32 = Qwen4Exp.boundedUnsigned32Value(of: routedExpertsValue) else {
            return nil;
        }
        return DiscoveryQwen4ExpConfigurationSummary(
            decoderLayers: decoderLayers,
            contextWindowTokens: contextWindowTokens,
            routedExperts: routedExperts
        );
    }

    /** Rust's `u32::try_from(value.as_u64()?)`: nil when the declared number exceeds the u32 range. */
    private static func boundedUnsigned32Value(of jsonValue: Any) -> UInt32? {
        guard let rawUnsignedValue: UInt64 = Qwen4Exp.unsignedInteger64Value(of: jsonValue) else {
            return nil;
        }
        return UInt32(exactly: rawUnsignedValue);
    }

    /**
     * serde_json's `Value::as_u64` over the JSONSerialization-shaped tree:
     * only numbers stored as unsigned integers qualify, so a whole-valued
     * float, a negative integer, a boolean, and a non-number all miss.
     */
    private static func unsignedInteger64Value(of jsonValue: Any) -> UInt64? {
        if (jsonValue is NSNull) {
            return nil;
        }
        guard let numberValue: NSNumber = jsonValue as? NSNumber else {
            return nil;
        }
        if CFGetTypeID(numberValue as CFTypeRef) == CFBooleanGetTypeID() {
            return nil;
        }
        let storageTypeDescription: String = String(cString: numberValue.objCType);
        if storageTypeDescription == "d" || storageTypeDescription == "f" {
            return nil;
        }
        if storageTypeDescription == "Q" {
            return numberValue.uint64Value;
        }
        let storedSignedValue: Int64 = numberValue.int64Value;
        guard storedSignedValue >= 0 else {
            return nil;
        }
        return UInt64(storedSignedValue);
    }
}
