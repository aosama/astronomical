import Foundation;
import IpcProtocol;

/// The validated Qwen3.5 configuration wire document, port of
/// crates/model-serving/src/qwen3_5/configuration/config_document.rs.

private let EXPECTED_MOE_TEXT_MODEL_TYPE: String = "qwen3_5_moe_text";
private let EXPECTED_DENSE_TEXT_MODEL_TYPE: String = "qwen3_5_text";
private let EXPECTED_HIDDEN_ACTIVATION: String = "silu";
private let MAXIMUM_MLX_SHAPE_DIMENSION: UInt64 = UInt64(Int32.max);

/// Qwen MTP sidecar declaration parsed from `mlx_lm_extra_tensors`.
struct MlxLmExtraTensors: Equatable {
    /// Path to the MTP sidecar safetensors file, relative to the model directory.
    /// E.g., "mtp.safetensors" or "optiq/mtp.safetensors".
    var mtpFile: String?;
}

/// Global MTP quantization parameters declared in the top-level config.
/// Provides default bit width and group size for MTP modules that lack
/// per-module overrides in the `quantization` dict. Absent when the
/// model does not declare quantized MTP.
struct MtplxMtpQuantization: Equatable {
    var bits: UInt32 = 0;
    var groupSize: UInt32 = 0;

    static func decoded(wireValue: JsonWireValue) throws -> MtplxMtpQuantization {
        let quantizationObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var parsedQuantization: MtplxMtpQuantization = MtplxMtpQuantization();
        parsedQuantization.bits = try quantizationObject.decodeOptionalUInt32AllowingAbsent(fieldName: "bits") ?? 0;
        parsedQuantization.groupSize = try quantizationObject.decodeOptionalUInt32AllowingAbsent(fieldName: "group_size") ?? 0;
        return parsedQuantization;
    }
}

/// Private wire schema retained only while validating one model config document.
struct Qwen3_5ConfigDocument {
    var architectures: Array<String> = Array();
    var eosTokenId: Array<UInt32>?;
    var modelType: String = "";
    var padTokenId: UInt32?;
    var quantization: OptiQQuantizationConfig?;
    var quantizationConfig: OptiQQuantizationConfig?;
    var textConfig: Qwen3_5TextConfig;
    var tieWordEmbeddings: Bool = false;
    var activationDtype: String?;
    /// Sidecar file declarations from config.json's `mlx_lm_extra_tensors` field.
    /// Absent when models store all tensors in the shard index.
    var mlxLmExtraTensors: MlxLmExtraTensors?;
    /// Top-level MTP sidecar path declared directly in config.json.
    /// Provides a fallback when `mlx_lm_extra_tensors` is absent.
    var mtpFile: String?;
    /// Global MTP quantization parameters for prequantized MTP sidecars.
    /// E.g. `{"bits": 4, "group_size": 64, "mode": "affine", "prequantized": true}`.
    /// Absent when the model does not declare quantized MTP.
    var mtxplxMtpQuantization: MtplxMtpQuantization?;

    static func decoded(wireValue: JsonWireValue) throws -> Qwen3_5ConfigDocument {
        let documentObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var parsedDocument: Qwen3_5ConfigDocument = Qwen3_5ConfigDocument(
            textConfig: try Qwen3_5TextConfig.decoded(
                wireValue: try documentObject.requireObjectValue(fieldName: "text_config")));
        parsedDocument.architectures = try documentObject.decodeArray(
            fieldName: "architectures",
            mappedElement: { (architectureWireValue: JsonWireValue) throws -> String in
                return try JsonWireValue.extractString(architectureWireValue);
            });
        parsedDocument.eosTokenId = try Qwen3_5ConfigValidation.optionalEosTokenIds(
            fromObject: documentObject, fieldName: "eos_token_id");
        parsedDocument.modelType = try documentObject.decodeString(fieldName: "model_type");
        parsedDocument.padTokenId = try documentObject.decodeOptionalUInt32AllowingAbsent(fieldName: "pad_token_id");
        parsedDocument.quantization = try documentObject.decodeOptionalRawValueAllowingAbsent(fieldName: "quantization")
            .map({ (quantizationWireValue: JsonWireValue) throws -> OptiQQuantizationConfig in
                return try OptiQQuantizationConfig.decoded(wireValue: quantizationWireValue);
            });
        parsedDocument.quantizationConfig = try documentObject.decodeOptionalRawValueAllowingAbsent(fieldName: "quantization_config")
            .map({ (quantizationWireValue: JsonWireValue) throws -> OptiQQuantizationConfig in
                return try OptiQQuantizationConfig.decoded(wireValue: quantizationWireValue);
            });
        parsedDocument.tieWordEmbeddings = try documentObject.decodeBool(fieldName: "tie_word_embeddings");
        // serde alias: `dtype` preferred, `torch_dtype` accepted when absent.
        parsedDocument.activationDtype = try documentObject.decodeOptionalStringAllowingAbsent(fieldName: "dtype")
            ?? documentObject.decodeOptionalStringAllowingAbsent(fieldName: "torch_dtype");
        if let extraTensorsObject: JsonWireObject = try documentObject.decodeOptionalObjectAllowingAbsent(fieldName: "mlx_lm_extra_tensors") {
            var parsedExtraTensors: MlxLmExtraTensors = MlxLmExtraTensors();
            parsedExtraTensors.mtpFile = try extraTensorsObject.decodeOptionalStringAllowingAbsent(fieldName: "mtp_file");
            parsedDocument.mlxLmExtraTensors = parsedExtraTensors;
        }
        parsedDocument.mtpFile = try documentObject.decodeOptionalStringAllowingAbsent(fieldName: "mtp_file");
        parsedDocument.mtxplxMtpQuantization = try documentObject.decodeOptionalRawValueAllowingAbsent(fieldName: "mtplx_mtp_quantization")
            .map({ (mtpQuantizationWireValue: JsonWireValue) throws -> MtplxMtpQuantization in
                return try MtplxMtpQuantization.decoded(wireValue: mtpQuantizationWireValue);
            });
        return parsedDocument;
    }
}

/// The validated Qwen3.5 text configuration half of the model document.
struct Qwen3_5TextConfig: Equatable {
    var attentionBias: Bool = false;
    var modelType: String = "";
    var textConfigDtype: String?;
    var textConfigEosTokenId: Array<UInt32>?;
    var hiddenAct: String = "";
    var hiddenSize: UInt32 = 0;
    var numHiddenLayers: UInt32 = 0;
    var numAttentionHeads: UInt32 = 0;
    var numKeyValueHeads: UInt32 = 0;
    var headDim: UInt32 = 0;
    /// Exact IEEE-754 float32 bits of `rms_norm_eps`.
    var rmsNormEpsilonBits: UInt32 = 0;
    /// Exact IEEE-754 float32 bits of the legacy `rope_theta`.
    var legacyRopeThetaBits: UInt32?;
    var ropeParameters: Qwen3_5RopeParameters?;
    /// Exact IEEE-754 float32 bits of `partial_rotary_factor`.
    var partialRotaryFactorBits: UInt32?;
    var mlpBias: Bool = false;
    var normTopkProb: Bool = true;
    var vocabSize: UInt32 = 0;
    var maxPositionEmbeddings: UInt32 = 0;
    var layerTypes: Array<String> = Array();
    var linearConvKernelDim: UInt32 = 0;
    var linearNumKeyHeads: UInt32 = 0;
    var linearNumValueHeads: UInt32 = 0;
    var linearKeyHeadDim: UInt32 = 0;
    var linearValueHeadDim: UInt32 = 0;
    var numExperts: UInt32 = 0;
    var numExpertsPerTok: UInt32 = 0;
    var moeIntermediateSize: UInt32 = 0;
    var sharedExpertIntermediateSize: UInt32 = 0;
    var intermediateSize: UInt32 = 0;
    var mtpNumHiddenLayers: UInt32 = 0;
    var mambaSsmDtype: String?;

    static func decoded(wireValue: JsonWireValue) throws -> Qwen3_5TextConfig {
        let textConfigObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var parsedTextConfig: Qwen3_5TextConfig = Qwen3_5TextConfig();
        parsedTextConfig.attentionBias = try textConfigObject.decodeBool(fieldName: "attention_bias");
        parsedTextConfig.modelType = try textConfigObject.decodeString(fieldName: "model_type");
        parsedTextConfig.textConfigDtype = try textConfigObject.decodeOptionalStringAllowingAbsent(fieldName: "dtype")
            ?? textConfigObject.decodeOptionalStringAllowingAbsent(fieldName: "torch_dtype");
        parsedTextConfig.textConfigEosTokenId = try Qwen3_5ConfigValidation.optionalEosTokenIds(
            fromObject: textConfigObject, fieldName: "eos_token_id");
        parsedTextConfig.hiddenAct = try textConfigObject.decodeString(fieldName: "hidden_act");
        parsedTextConfig.hiddenSize = try textConfigObject.decodeUInt32(fieldName: "hidden_size");
        parsedTextConfig.numHiddenLayers = try textConfigObject.decodeUInt32(fieldName: "num_hidden_layers");
        parsedTextConfig.numAttentionHeads = try textConfigObject.decodeUInt32(fieldName: "num_attention_heads");
        parsedTextConfig.numKeyValueHeads = try textConfigObject.decodeUInt32(fieldName: "num_key_value_heads");
        parsedTextConfig.headDim = try textConfigObject.decodeUInt32(fieldName: "head_dim");
        parsedTextConfig.rmsNormEpsilonBits = try Qwen3_5ConfigValidation.float32Bits(
            fromNumber: try textConfigObject.requireObjectValue(fieldName: "rms_norm_eps"));
        parsedTextConfig.legacyRopeThetaBits = try textConfigObject
            .decodeOptionalRawValueAllowingAbsent(fieldName: "rope_theta")
            .map({ (ropeThetaWireValue: JsonWireValue) throws -> UInt32 in
                return try Qwen3_5ConfigValidation.optionalFloat32Bits(fromNumber: ropeThetaWireValue) ?? 0;
            });
        parsedTextConfig.ropeParameters = try textConfigObject.decodeOptionalRawValueAllowingAbsent(fieldName: "rope_parameters")
            .map({ (ropeParametersWireValue: JsonWireValue) throws -> Qwen3_5RopeParameters in
                return try Qwen3_5RopeParameters.decoded(wireValue: ropeParametersWireValue);
            });
        parsedTextConfig.partialRotaryFactorBits = try textConfigObject
            .decodeOptionalRawValueAllowingAbsent(fieldName: "partial_rotary_factor")
            .map({ (partialRotaryWireValue: JsonWireValue) throws -> UInt32 in
                return try Qwen3_5ConfigValidation.optionalFloat32Bits(fromNumber: partialRotaryWireValue) ?? 0;
            });
        parsedTextConfig.mlpBias = try textConfigObject.decodeBoolAllowingAbsent(fieldName: "mlp_bias");
        parsedTextConfig.normTopkProb = try textConfigObject.decodeOptionalBoolAllowingAbsent(fieldName: "norm_topk_prob") ?? true;
        parsedTextConfig.vocabSize = try textConfigObject.decodeUInt32(fieldName: "vocab_size");
        parsedTextConfig.maxPositionEmbeddings = try textConfigObject.decodeUInt32(fieldName: "max_position_embeddings");
        parsedTextConfig.layerTypes = try textConfigObject.decodeArray(
            fieldName: "layer_types",
            mappedElement: { (layerTypeWireValue: JsonWireValue) throws -> String in
                return try JsonWireValue.extractString(layerTypeWireValue);
            });
        parsedTextConfig.linearConvKernelDim = try textConfigObject.decodeUInt32(fieldName: "linear_conv_kernel_dim");
        parsedTextConfig.linearNumKeyHeads = try textConfigObject.decodeUInt32(fieldName: "linear_num_key_heads");
        parsedTextConfig.linearNumValueHeads = try textConfigObject.decodeUInt32(fieldName: "linear_num_value_heads");
        parsedTextConfig.linearKeyHeadDim = try textConfigObject.decodeUInt32(fieldName: "linear_key_head_dim");
        parsedTextConfig.linearValueHeadDim = try textConfigObject.decodeUInt32(fieldName: "linear_value_head_dim");
        parsedTextConfig.numExperts = try textConfigObject.decodeOptionalUInt32AllowingAbsent(fieldName: "num_experts") ?? 0;
        parsedTextConfig.numExpertsPerTok = try textConfigObject.decodeOptionalUInt32AllowingAbsent(fieldName: "num_experts_per_tok") ?? 0;
        parsedTextConfig.moeIntermediateSize = try textConfigObject.decodeOptionalUInt32AllowingAbsent(fieldName: "moe_intermediate_size") ?? 0;
        parsedTextConfig.sharedExpertIntermediateSize = try textConfigObject.decodeOptionalUInt32AllowingAbsent(fieldName: "shared_expert_intermediate_size") ?? 0;
        parsedTextConfig.intermediateSize = try textConfigObject.decodeOptionalUInt32AllowingAbsent(fieldName: "intermediate_size") ?? 0;
        parsedTextConfig.mtpNumHiddenLayers = try textConfigObject.decodeUInt32(fieldName: "mtp_num_hidden_layers");
        parsedTextConfig.mambaSsmDtype = try textConfigObject.decodeOptionalStringAllowingAbsent(fieldName: "mamba_ssm_dtype");
        return parsedTextConfig;
    }
}

extension Qwen3_5TextConfig: QuantizationConfigSource {

    public func layerCount() -> UInt32 {
        return self.numHiddenLayers;
    }

    public func decoderLayerIsFullAttention(decoderLayerIndex: Int) -> Bool {
        guard decoderLayerIndex < self.layerTypes.count else {
            return false;
        }
        return self.layerTypes[decoderLayerIndex] == "full_attention";
    }
}

extension Qwen3_5TextConfig {

    func validate(feedForwardArchitecture: Qwen3_5FeedForwardArchitecture) throws -> Void {
        let expectedTextModelType: String;
        switch feedForwardArchitecture {
        case .dense:
            expectedTextModelType = EXPECTED_DENSE_TEXT_MODEL_TYPE;
        case .mixtureOfExperts:
            expectedTextModelType = EXPECTED_MOE_TEXT_MODEL_TYPE;
        }
        try Qwen3_5ConfigValidation.validateExactValue(
            fieldName: "text_config.model_type", actualValue: self.modelType,
            expectedValue: expectedTextModelType);
        try Qwen3_5ConfigValidation.validateExactValue(
            fieldName: "text_config.hidden_act", actualValue: self.hiddenAct,
            expectedValue: EXPECTED_HIDDEN_ACTIVATION);
        if let ropeParameters: Qwen3_5RopeParameters = self.ropeParameters {
            try ropeParameters.validate();
        }
        try Qwen3_5ConfigValidation.validateExactBoolean(
            fieldName: "text_config.attention_bias", actualValue: self.attentionBias, expectedValue: false);
        try Qwen3_5ConfigValidation.validateExactBoolean(
            fieldName: "text_config.mlp_bias", actualValue: self.mlpBias, expectedValue: false);
        try Qwen3_5ConfigValidation.validateExactBoolean(
            fieldName: "text_config.norm_topk_prob", actualValue: self.normTopkProb, expectedValue: true);
        if let mambaSsmDtype: String = self.mambaSsmDtype,
            mambaSsmDtype != "bfloat16" && mambaSsmDtype != "float32" {
            throw Qwen3_5ConfigError.invalidConfigValueDynamic(
                description: "text_config.mamba_ssm_dtype contains unsupported dtype '\(mambaSsmDtype)'");
        }
        if self.hiddenSize == 0
            || self.numHiddenLayers == 0
            || self.numAttentionHeads == 0
            || self.numKeyValueHeads == 0
            || self.headDim == 0
            || self.vocabSize == 0
            || self.maxPositionEmbeddings == 0
            || self.linearConvKernelDim == 0
            || self.linearNumKeyHeads == 0
            || self.linearNumValueHeads == 0
            || self.linearKeyHeadDim == 0
            || self.linearValueHeadDim == 0 {
            throw Qwen3_5ConfigError.invalidConfigValue(description: "text_config numeric fields must be positive");
        }
        switch feedForwardArchitecture {
        case .dense where self.intermediateSize == 0:
            throw Qwen3_5ConfigError.invalidConfigValue(description: "text_config numeric fields must be positive");
        case .mixtureOfExperts:
            if self.numExperts == 0 || self.numExpertsPerTok == 0
                || self.moeIntermediateSize == 0 || self.sharedExpertIntermediateSize == 0 {
                throw Qwen3_5ConfigError.invalidConfigValue(description: "text_config numeric fields must be positive");
            }
        default:
            break;
        }
        if self.numKeyValueHeads != 0 && self.numAttentionHeads % self.numKeyValueHeads != 0 {
            throw Qwen3_5ConfigError.invalidConfigValue(
                description: "text_config.num_attention_heads must divide evenly by num_key_value_heads");
        }
        if feedForwardArchitecture == .mixtureOfExperts && self.numExpertsPerTok > self.numExperts {
            throw Qwen3_5ConfigError.invalidConfigValue(
                description: "text_config.num_experts_per_tok must not exceed num_experts");
        }
        if feedForwardArchitecture == .dense
            && (self.numExperts != 0 || self.numExpertsPerTok != 0
                || self.moeIntermediateSize != 0 || self.sharedExpertIntermediateSize != 0) {
            throw Qwen3_5ConfigError.invalidConfigValue(
                description: "dense Qwen3.5 text config must not declare sparse-expert dimensions");
        }
        if self.linearConvKernelDim > MAXIMUM_MLX_SHAPE_DIMENSION
            || self.linearNumValueHeads > MAXIMUM_MLX_SHAPE_DIMENSION
            || self.linearValueHeadDim > MAXIMUM_MLX_SHAPE_DIMENSION
            || self.linearKeyHeadDim > MAXIMUM_MLX_SHAPE_DIMENSION
            || self.linearConvolutionStateDimension() > MAXIMUM_MLX_SHAPE_DIMENSION {
            throw Qwen3_5ConfigError.invalidConfigValue(
                description: "linear-attention dimensions must fit the MLX signed 32-bit shape range");
        }
        if self.layerTypes.count != Int(self.numHiddenLayers) {
            throw Qwen3_5ConfigError.layerTypeCountMismatch(
                actualLayerTypeCount: self.layerTypes.count,
                expectedLayerTypeCount: Int(self.numHiddenLayers));
        }
        for layerType: String in self.layerTypes {
            if layerType != "full_attention" && layerType != "linear_attention" {
                throw Qwen3_5ConfigError.invalidConfigValueDynamic(
                    description: "text_config.layer_types contains unsupported attention type '\(layerType)'");
            }
        }
    }

    func linearConvolutionStateDimension() -> UInt64 {
        let keyDimensionProduct: UInt64 = saturatingProduct(UInt64(self.linearNumKeyHeads), UInt64(self.linearKeyHeadDim));
        let valueDimensionProduct: UInt64 = saturatingProduct(UInt64(self.linearNumValueHeads), UInt64(self.linearValueHeadDim));
        return saturatingSum(saturatingProduct(keyDimensionProduct, 2), valueDimensionProduct);
    }

    private func saturatingProduct(_ leftValue: UInt64, _ rightValue: UInt64) -> UInt64 {
        let (productValue, didOverflow) = leftValue.multipliedReportingOverflow(by: rightValue);
        return didOverflow ? UInt64.max : productValue;
    }

    private func saturatingSum(_ leftValue: UInt64, _ rightValue: UInt64) -> UInt64 {
        let (sumValue, didOverflow) = leftValue.addingReportingOverflow(rightValue);
        return didOverflow ? UInt64.max : sumValue;
    }
}

/// The multimodal rotary parameters pinned by the Qwen3.5 text configuration.
struct Qwen3_5RopeParameters: Equatable {
    var mropeInterleaved: Bool = false;
    var mropeSection: Array<UInt32> = Array();
    /// Exact IEEE-754 float32 bits of `partial_rotary_factor`.
    var partialRotaryFactor: UInt32 = 0;
    /// Exact IEEE-754 float32 bits of `rope_theta`.
    var ropeThetaBits: UInt32 = 0;
    var ropeType: String = "";

    static func decoded(wireValue: JsonWireValue) throws -> Qwen3_5RopeParameters {
        let ropeObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var parsedRopeParameters: Qwen3_5RopeParameters = Qwen3_5RopeParameters();
        parsedRopeParameters.mropeInterleaved = try ropeObject.decodeBool(fieldName: "mrope_interleaved");
        parsedRopeParameters.mropeSection = try ropeObject.decodeArray(
            fieldName: "mrope_section",
            mappedElement: { (sectionWireValue: JsonWireValue) throws -> UInt32 in
                return try sectionWireValue.uint32TokenId();
            });
        parsedRopeParameters.partialRotaryFactor = try Qwen3_5ConfigValidation.float32Bits(
            fromNumber: try ropeObject.requireObjectValue(fieldName: "partial_rotary_factor"));
        parsedRopeParameters.ropeThetaBits = try Qwen3_5ConfigValidation.float32Bits(
            fromNumber: try ropeObject.requireObjectValue(fieldName: "rope_theta"));
        // serde alias: `type` preferred, `rope_type` accepted when absent.
        parsedRopeParameters.ropeType = try ropeObject.decodeOptionalStringAllowingAbsent(fieldName: "type")
            ?? ropeObject.decodeOptionalStringAllowingAbsent(fieldName: "rope_type")
            ?? { throw JsonWireProblem.missingField(fieldName: "type") }();
        return parsedRopeParameters;
    }

    func validate() throws -> Void {
        try Qwen3_5ConfigValidation.validateExactBoolean(
            fieldName: "text_config.rope_parameters.mrope_interleaved",
            actualValue: self.mropeInterleaved, expectedValue: true);
        if self.mropeSection != [11, 11, 10] {
            throw Qwen3_5ConfigError.mropeSectionMismatch(actualSection: self.mropeSection);
        }
        try Qwen3_5ConfigValidation.validateExactValue(
            fieldName: "text_config.rope_parameters.type", actualValue: self.ropeType,
            expectedValue: "default");
    }
}
