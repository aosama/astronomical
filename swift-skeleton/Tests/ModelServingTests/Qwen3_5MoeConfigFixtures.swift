import Foundation;
import ModelServing;
import IpcProtocol;
import Testing;

/// Frozen Qwen3.5-MoE test fixtures, port of the fixture half of
/// crates/model-serving/tests/common/qwen3_5_moe.rs, plus small JSON-mutation
/// helpers mirroring the serde_json::Value edits the Rust tests perform.
enum Qwen3_5MoeConfigFixtures {

    static func wireValue(_ jsonText: String) throws -> JsonWireValue {
        return try JsonWireParser.parseDocument(documentBytes: Data(jsonText.utf8));
    }

    static func frozenOrnith10ConfigBytes() throws -> Array<UInt8> {
        var configValue: JsonWireValue = try wireValue("""
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "torch_dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }
    }
    """);
        var quantization: JsonWireValue = try wireValue(
            #"{"group_size": 64, "bits": 6, "mode": "affine"}"#);
        for decoderLayerIndex in 0..<40 {
            for gateName in ["gate", "shared_expert_gate"] {
                quantization = quantization.settingObjectKey(
                    path: ["language_model.model.layers.\(decoderLayerIndex).mlp.\(gateName)"],
                    newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
            }
        }
        // Embedding and lm_head must be 8-bit even with a 6-bit default.
        quantization = quantization.settingObjectKey(
            path: ["language_model.model.embed_tokens"],
            newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
        quantization = quantization.settingObjectKey(
            path: ["language_model.lm_head"],
            newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
        configValue = configValue.settingObjectKey(path: ["quantization"], newValue: quantization);
        configValue = configValue.settingObjectKey(path: ["quantization_config"], newValue: quantization);
        configValue = configValue.settingObjectKey(
            path: ["text_config", "layer_types"],
            newValue: JsonWireValue.stringArray(expectedLayerTypes()));
        return try serializedBytes(configValue);
    }

    static func frozenOrnith10OptiQConfigBytes() throws -> Array<UInt8> {
        var configValue: JsonWireValue = try wireValue("""
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "type": "default"
            },
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }
    }
    """);
        var quantization: JsonWireValue = try wireValue(
            #"{"group_size": 64, "bits": 4, "mode": "affine"}"#);
        let quantizedModuleNames: Array<String> = expectedQuantizedModuleNames();
        for (quantizedModuleIndex, quantizedModuleName): (Int, String) in quantizedModuleNames.enumerated() {
            let quantizationBits: UInt32 = quantizedModuleIndex < 113 ? 4 : 8;
            quantization = quantization.settingObjectKey(
                path: [quantizedModuleName],
                newValue: try wireValue(
                    #"{"group_size": 64, "bits": \#(quantizationBits)}"#));
        }
        configValue = configValue.settingObjectKey(path: ["quantization"], newValue: quantization);
        configValue = configValue.settingObjectKey(path: ["quantization_config"], newValue: quantization);
        configValue = configValue.settingObjectKey(
            path: ["text_config", "layer_types"],
            newValue: JsonWireValue.stringArray(expectedLayerTypes()));
        return try serializedBytes(configValue);
    }

    static func frozenOptiQMetadataBytes() throws -> Array<UInt8> {
        let optiQConfigValue: JsonWireValue = try wireValue(
            String(bytes: try frozenOrnith10OptiQConfigBytes(), encoding: String.Encoding.utf8)!);
        guard case .object(let quantizationObject) = optiQConfigValue.objectValue(forKey: "quantization")
            ?? .null else {
            Issue.record("the quantization map should be an object");
            return Array();
        }
        var measuredModuleBits: JsonWireObject = quantizationObject;
        for nonModuleFieldName in [
            "group_size", "bits", "mode",
            "language_model.model.embed_tokens", "language_model.lm_head",
        ] {
            measuredModuleBits = measuredModuleBits.removingKey(nonModuleFieldName);
        }
        let metadataDocument: JsonWireValue = try wireValue("""
    {
        "method": "optiq_mixed_precision_transferred",
        "base_model": "example/Ornith-1.0-35B",
        "reference": "bit map transferred from a community OptiQ 4-bit artifact",
        "sensitivity_measured_on": "Qwen/Qwen3.5-35B-A3B",
        "target_bpw": 4.5,
        "achieved_bpw": 4.5131342941951385,
        "n_high_bits": 397,
        "n_low_bits": 113,
        "threshold": 0.0,
        "per_layer": null
    }
    """);
        return try serializedBytes(metadataDocument.settingObjectKey(
            path: ["per_layer"], newValue: .object(measuredModuleBits)));
    }

    /// Freezes a sparse mixed-precision quantization configuration shape:
    /// a 6-bit affine default with a sparse override map whose unlisted expected
    /// modules (router gates) are stored as native floating point and
    /// resolved through the shard-index scan.
    static func frozenSparseMixedPrecisionConfigBytes() throws -> Array<UInt8> {
        var configValue: JsonWireValue = try wireValue("""
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "type": "default"
            },
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }
    }
    """);
        var quantization: JsonWireValue = try wireValue(
            #"{"group_size": 64, "bits": 6, "mode": "affine"}"#);
        // Every layer lifts the shared expert to 8-bit; the gate stays at group 64
        // while its projections widen to group 128.
        for decoderLayerIndex: Int in 0..<40 {
            let layerPrefix: String = "language_model.model.layers.\(decoderLayerIndex)";
            quantization = quantization.settingObjectKey(
                path: ["\(layerPrefix).mlp.shared_expert_gate"],
                newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
            for projectionName: String in ["gate_proj", "up_proj", "down_proj"] {
                quantization = quantization.settingObjectKey(
                    path: ["\(layerPrefix).mlp.shared_expert.\(projectionName)"],
                    newValue: try wireValue(#"{"group_size": 128, "bits": 8}"#));
            }
        }
        // Linear-attention output projections all widen to group 128 at the default
        // 6-bit width.
        for decoderLayerIndex: Int in 0..<40 where decoderLayerIndex % 4 != 3 {
            quantization = quantization.settingObjectKey(
                path: ["language_model.model.layers.\(decoderLayerIndex).linear_attn.out_proj"],
                newValue: try wireValue(#"{"group_size": 128, "bits": 6}"#));
        }
        // Only a sparse subset of attention input projections is lifted to 8-bit;
        // the remaining full-attention and linear-attention layers keep the 6-bit
        // group-64 default.
        for decoderLayerIndex: Int in [3, 7, 31, 35, 39] {
            for projectionName: String in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                quantization = quantization.settingObjectKey(
                    path: ["language_model.model.layers.\(decoderLayerIndex).self_attn.\(projectionName)"],
                    newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
            }
        }
        for decoderLayerIndex: Int in [0, 10] {
            for projectionName: String in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"] {
                quantization = quantization.settingObjectKey(
                    path: ["language_model.model.layers.\(decoderLayerIndex).linear_attn.\(projectionName)"],
                    newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
            }
        }
        quantization = quantization.settingObjectKey(
            path: ["language_model.model.embed_tokens"],
            newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
        quantization = quantization.settingObjectKey(
            path: ["language_model.lm_head"],
            newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
        configValue = configValue.settingObjectKey(path: ["quantization"], newValue: quantization);
        configValue = configValue.settingObjectKey(path: ["quantization_config"], newValue: quantization);
        configValue = configValue.settingObjectKey(
            path: ["text_config", "layer_types"],
            newValue: JsonWireValue.stringArray(expectedLayerTypes()));
        return try serializedBytes(configValue);
    }

    static func minimalValidConfigJson() throws -> JsonWireValue {
        var configValue: JsonWireValue = try wireValue("""
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "type": "default"
            },
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        },
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"}
    }
    """);
        // Add the decoder attention schedule.
        let layerTypes: Array<String> = (0..<40).map({ (decoderLayerIndex: Int) -> String in
            return decoderLayerIndex % 4 == 3 ? "full_attention" : "linear_attention";
        });
        configValue = configValue.settingObjectKey(
            path: ["text_config", "layer_types"], newValue: JsonWireValue.stringArray(layerTypes));
        // Add quantization per-layer overrides (required by the parser).
        var quantization: JsonWireValue = try wireValue(
            #"{"group_size": 64, "bits": 4, "mode": "affine"}"#);
        quantization = quantization.settingObjectKey(
            path: ["language_model.model.embed_tokens"],
            newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
        quantization = quantization.settingObjectKey(
            path: ["language_model.lm_head"],
            newValue: try wireValue(#"{"group_size": 64, "bits": 8}"#));
        configValue = configValue.settingObjectKey(path: ["quantization"], newValue: quantization);
        configValue = configValue.settingObjectKey(path: ["quantization_config"], newValue: quantization);
        return configValue;
    }

    static func expectedQuantizedModuleNames() -> Array<String> {
        var quantizedModuleNames: Array<String> = Array();
        for decoderLayerIndex in 0..<40 {
            let layerPrefix: String = "language_model.model.layers.\(decoderLayerIndex)";
            if decoderLayerIndex % 4 == 3 {
                for projectionName in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    quantizedModuleNames.append("\(layerPrefix).self_attn.\(projectionName)");
                }
            } else {
                for projectionName in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"] {
                    quantizedModuleNames.append("\(layerPrefix).linear_attn.\(projectionName)");
                }
            }
            for moduleSuffix in [
                "mlp.gate",
                "mlp.switch_mlp.gate_proj",
                "mlp.switch_mlp.up_proj",
                "mlp.switch_mlp.down_proj",
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj",
                "mlp.shared_expert_gate",
            ] {
                quantizedModuleNames.append("\(layerPrefix).\(moduleSuffix)");
            }
        }
        quantizedModuleNames.append("language_model.model.embed_tokens");
        quantizedModuleNames.append("language_model.lm_head");
        return quantizedModuleNames;
    }

    static func expectedLayerTypes() -> Array<String> {
        return (0..<40).map({ (decoderLayerIndex: Int) -> String in
            return decoderLayerIndex % 4 == 3 ? "full_attention" : "linear_attention";
        });
    }

    static func serializedBytes(_ wireValue: JsonWireValue) throws -> Array<UInt8> {
        return Array(try wireValue.serializedText.utf8);
    }
}

extension JsonWireValue {

    /// Reads an object member as a value, or nil when absent or not an object.
    func objectValue(forKey propertyName: String) -> JsonWireValue? {
        guard case let .object(objectValue) = self else {
            return nil;
        }
        return objectValue.value(forKey: propertyName);
    }

    /// Replaces or inserts a value at the object-key path, mirroring
    /// `serde_json::Value` index assignment in the Rust tests.
    func settingObjectKey(path: Array<String>, newValue: JsonWireValue) -> JsonWireValue {
        guard let firstKey: String = path.first else {
            return newValue;
        }
        switch self {
        case .object(let objectValue):
            var rebuiltObject: JsonWireObject = JsonWireObject(entries: Array());
            var replacedEntry: Bool = false;
            for entry: (key: String, value: JsonWireValue) in objectValue.entries {
                if entry.key == firstKey {
                    replacedEntry = true;
                    rebuiltObject.appendEntry(
                        key: entry.key,
                        value: entry.value.settingObjectKey(path: Array(path.dropFirst()), newValue: newValue));
                } else {
                    rebuiltObject.appendEntry(key: entry.key, value: entry.value);
                }
            }
            if replacedEntry == false {
                rebuiltObject.appendEntry(
                    key: firstKey,
                    value: JsonWireValue.null.settingObjectKey(
                        path: Array(path.dropFirst()), newValue: newValue));
            }
            return .object(rebuiltObject);
        case .array(let elementValues):
            // Array path elements are numeric indices encoded as decimal strings.
            guard let elementIndex: Int = Int(firstKey), elementIndex < elementValues.count else {
                return self;
            }
            var rebuiltElements: Array<JsonWireValue> = elementValues;
            rebuiltElements[elementIndex] = elementValues[elementIndex]
                .settingObjectKey(path: Array(path.dropFirst()), newValue: newValue);
            return .array(rebuiltElements);
        default:
            if path.count == 1 {
                return newValue;
            }
            return self;
        }
    }

    /// Removes one object key, mirroring `serde_json::Value::as_object_mut().remove`.
    func removingObjectKey(path: Array<String>) -> JsonWireValue {
        guard let firstKey: String = path.first else {
            return self;
        }
        switch self {
        case .object(let objectValue):
            var rebuiltObject: JsonWireObject = JsonWireObject(entries: Array());
            for entry: (key: String, value: JsonWireValue) in objectValue.entries {
                if entry.key == firstKey && path.count == 1 {
                    continue;
                }
                rebuiltObject.appendEntry(
                    key: entry.key,
                    value: entry.value.removingObjectKey(path: Array(path.dropFirst())));
            }
            return .object(rebuiltObject);
        default:
            return self;
        }
    }
}

extension JsonWireObject {

    /// Removes one key, returning the rebuilt insertion-ordered object.
    func removingKey(_ propertyName: String) -> JsonWireObject {
        var rebuiltObject: JsonWireObject = JsonWireObject(entries: Array());
        for entry: (key: String, value: JsonWireValue) in self.entries {
            if entry.key == propertyName {
                continue;
            }
            rebuiltObject.appendEntry(key: entry.key, value: entry.value);
        }
        return rebuiltObject;
    }
}
