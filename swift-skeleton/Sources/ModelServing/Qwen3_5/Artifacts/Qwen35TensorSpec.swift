import Foundation;

/// Qwen3.5 language-tensor metadata generation, port of
/// crates/model-serving/src/qwen3_5/artifacts/tensor_spec.rs.

private let PACKING_WORD_BITS: Int = 32;

/// Qwen3.5 language tensor-spec helpers shared by the dense and MoE engines.
public enum Qwen3_5TensorSpec {

    /// Generates the executable language-tensor metadata for a Qwen3.5 artifact.
    public static func qwen3_5LanguageTensorProfiles(qwen3_5Config: Qwen3_5Config) -> Array<TensorProfile> {
        let hiddenSize: Int = Int(qwen3_5Config.hiddenSize());
        let layerCount: Int = Int(qwen3_5Config.layerCount());
        let vocabularySize: Int = Int(qwen3_5Config.vocabularySize());
        var tensorProfiles: Array<TensorProfile> = Array();
        appendQwen3_5QuantizedAffineTensorProfiles(
            tensorProfiles: &tensorProfiles,
            tensorPrefix: "language_model.model.embed_tokens",
            leadingDimensions: [vocabularySize],
            inputDimension: hiddenSize,
            quantizationProfile: qwen3_5Config.quantizationProfile(forModule: "language_model.model.embed_tokens"));
        for decoderLayerIndex: Int in 0..<layerCount {
            appendQwen3_5DecoderLayerTensorProfiles(
                tensorProfiles: &tensorProfiles,
                decoderLayerIndex: decoderLayerIndex,
                hiddenSize: hiddenSize,
                qwen3_5Config: qwen3_5Config);
        }
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "language_model.model.norm.weight",
            tensorDtype: .modelFloat,
            tensorShape: [hiddenSize]));
        if qwen3_5Config.hasTiedEmbeddings() == false {
            appendQwen3_5QuantizedAffineTensorProfiles(
                tensorProfiles: &tensorProfiles,
                tensorPrefix: "language_model.lm_head",
                leadingDimensions: [vocabularySize],
                inputDimension: hiddenSize,
                quantizationProfile: qwen3_5Config.quantizationProfile(forModule: "language_model.lm_head"));
        }
        return tensorProfiles;
    }

    /// Returns the language tensors that remain resident after sparse expert paging.
    public static func qwen3_5ResidentLanguageTensorProfiles(qwen3_5Config: Qwen3_5Config) -> Array<TensorProfile> {
        let completeLanguageTensorProfiles: Array<TensorProfile> = qwen3_5LanguageTensorProfiles(qwen3_5Config: qwen3_5Config);
        switch qwen3_5Config.feedForwardArchitecture() {
        case .dense:
            return completeLanguageTensorProfiles;
        case .mixtureOfExperts:
            return completeLanguageTensorProfiles.filter({ (tensorProfile: TensorProfile) -> Bool in
                return Qwen3_5MoeTensorSpec.isSparseSelectedExpertTensorName(tensorName: tensorProfile.name) == false;
            });
        }
    }

    static func appendQwen3_5DecoderLayerTensorProfiles(
        tensorProfiles: inout Array<TensorProfile>,
        decoderLayerIndex: Int,
        hiddenSize: Int,
        qwen3_5Config: Qwen3_5Config) {
        let queryHeadCount: Int = Int(qwen3_5Config.queryHeadCount());
        let keyValueHeadCount: Int = Int(qwen3_5Config.keyValueHeadCount());
        let headDimension: Int = Int(qwen3_5Config.headDimension());
        let layerPrefix: String = "language_model.model.layers.\(decoderLayerIndex)";
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(layerPrefix).input_layernorm.weight",
            tensorDtype: .modelFloat,
            tensorShape: [hiddenSize]));
        if qwen3_5Config.decoderLayerIsFullAttention(decoderLayerIndex: decoderLayerIndex) {
            appendQwen3_5FullAttentionTensorProfiles(
                tensorProfiles: &tensorProfiles,
                layerPrefix: layerPrefix,
                hiddenSize: hiddenSize,
                queryHeadCount: queryHeadCount,
                keyValueHeadCount: keyValueHeadCount,
                headDimension: headDimension,
                qwen3_5Config: qwen3_5Config);
        } else {
            appendQwen3_5LinearAttentionTensorProfiles(
                tensorProfiles: &tensorProfiles,
                layerPrefix: layerPrefix,
                hiddenSize: hiddenSize,
                qwen3_5Config: qwen3_5Config);
        }
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(layerPrefix).post_attention_layernorm.weight",
            tensorDtype: .modelFloat,
            tensorShape: [hiddenSize]));
        switch qwen3_5Config.feedForwardArchitecture() {
        case .dense:
            Qwen3_5DenseTensorSpec.appendQwen3_5DenseMlpTensorProfiles(
                tensorProfiles: &tensorProfiles,
                decoderLayerPrefix: layerPrefix,
                hiddenSize: hiddenSize,
                qwen3_5Config: qwen3_5Config);
        case .mixtureOfExperts:
            Qwen3_5MoeTensorSpec.appendQwen3_5MoeFeedForwardTensorProfiles(
                tensorProfiles: &tensorProfiles,
                decoderLayerPrefix: layerPrefix,
                hiddenSize: hiddenSize,
                qwen3_5Config: qwen3_5Config);
        }
    }

    private static func appendQwen3_5FullAttentionTensorProfiles(
        tensorProfiles: inout Array<TensorProfile>,
        layerPrefix: String,
        hiddenSize: Int,
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        qwen3_5Config: Qwen3_5Config) {
        let queryProjectionOutputDimension: Int = queryHeadCount * headDimension * 2;
        let keyValueProjectionOutputDimension: Int = keyValueHeadCount * headDimension;
        let attentionOutputInputDimension: Int = queryHeadCount * headDimension;
        let attentionProjectionShapes: Array<(projectionName: String, outputDimension: Int, inputDimension: Int)> = [
            ("q_proj", queryProjectionOutputDimension, hiddenSize),
            ("k_proj", keyValueProjectionOutputDimension, hiddenSize),
            ("v_proj", keyValueProjectionOutputDimension, hiddenSize),
            ("o_proj", hiddenSize, attentionOutputInputDimension),
        ];
        for attentionProjection: (projectionName: String, outputDimension: Int, inputDimension: Int) in attentionProjectionShapes {
            let projectionModuleName: String = "\(layerPrefix).self_attn.\(attentionProjection.projectionName)";
            appendQwen3_5QuantizedAffineTensorProfiles(
                tensorProfiles: &tensorProfiles,
                tensorPrefix: projectionModuleName,
                leadingDimensions: [attentionProjection.outputDimension],
                inputDimension: attentionProjection.inputDimension,
                quantizationProfile: qwen3_5Config.quantizationProfile(forModule: projectionModuleName));
        }
        for normalizationName in ["q_norm", "k_norm"] {
            tensorProfiles.append(qwen3_5TensorProfile(
                tensorName: "\(layerPrefix).self_attn.\(normalizationName).weight",
                tensorDtype: .modelFloat,
                tensorShape: [headDimension]));
        }
    }

    private static func appendQwen3_5LinearAttentionTensorProfiles(
        tensorProfiles: inout Array<TensorProfile>,
        layerPrefix: String,
        hiddenSize: Int,
        qwen3_5Config: Qwen3_5Config) {
        let linearKeyHeadCount: Int = Int(qwen3_5Config.linearKeyHeadCount());
        let linearValueHeadCount: Int = Int(qwen3_5Config.linearValueHeadCount());
        let linearKeyHeadDimension: Int = Int(qwen3_5Config.linearKeyHeadDimension());
        let linearValueHeadDimension: Int = Int(qwen3_5Config.linearValueHeadDimension());
        let linearKeyDimension: Int = linearKeyHeadCount * linearKeyHeadDimension;
        let linearValueDimension: Int = linearValueHeadCount * linearValueHeadDimension;
        let convolutionDimension: Int = linearKeyDimension * 2 + linearValueDimension;
        let linearAttentionPrefix: String = "\(layerPrefix).linear_attn";
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(linearAttentionPrefix).conv1d.weight",
            tensorDtype: .modelFloat,
            tensorShape: [
                convolutionDimension,
                Int(qwen3_5Config.linearConvolutionKernelDimension()),
                1,
            ]));
        let linearProjectionShapes: Array<(projectionName: String, outputDimension: Int, inputDimension: Int)> = [
            ("in_proj_qkv", linearKeyDimension * 2 + linearValueDimension, hiddenSize),
            ("in_proj_z", linearValueDimension, hiddenSize),
            ("in_proj_b", linearValueHeadCount, hiddenSize),
            ("in_proj_a", linearValueHeadCount, hiddenSize),
            ("out_proj", hiddenSize, linearValueDimension),
        ];
        for linearProjection: (projectionName: String, outputDimension: Int, inputDimension: Int) in linearProjectionShapes {
            let projectionModuleName: String = "\(linearAttentionPrefix).\(linearProjection.projectionName)";
            appendQwen3_5QuantizedAffineTensorProfiles(
                tensorProfiles: &tensorProfiles,
                tensorPrefix: projectionModuleName,
                leadingDimensions: [linearProjection.outputDimension],
                inputDimension: linearProjection.inputDimension,
                quantizationProfile: qwen3_5Config.quantizationProfile(forModule: projectionModuleName));
        }
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(linearAttentionPrefix).dt_bias",
            tensorDtype: .modelFloat,
            tensorShape: [linearValueHeadCount]));
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(linearAttentionPrefix).A_log",
            tensorDtype: .modelFloat,
            tensorShape: [linearValueHeadCount]));
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(linearAttentionPrefix).norm.weight",
            tensorDtype: .modelFloat,
            tensorShape: [linearValueHeadDimension]));
    }

    static func appendQwen3_5QuantizedAffineTensorProfiles(
        tensorProfiles: inout Array<TensorProfile>,
        tensorPrefix: String,
        leadingDimensions: Array<Int>,
        inputDimension: Int,
        quantizationProfile: OptiQQuantizationProfile) {
        if quantizationProfile.isUnquantized() {
            var nativeBfloat16WeightShape: Array<Int> = leadingDimensions;
            nativeBfloat16WeightShape.append(inputDimension);
            tensorProfiles.append(qwen3_5TensorProfile(
                tensorName: "\(tensorPrefix).weight",
                tensorDtype: .modelFloat,
                tensorShape: nativeBfloat16WeightShape));
            return;
        }
        let quantizationBits: Int = Int(quantizationProfile.bits);
        let quantizationGroupSize: Int = Int(quantizationProfile.groupSize);
        var packedWeightShape: Array<Int> = leadingDimensions;
        packedWeightShape.append(inputDimension * quantizationBits / PACKING_WORD_BITS);
        var scaleShape: Array<Int> = leadingDimensions;
        scaleShape.append(inputDimension / quantizationGroupSize);
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(tensorPrefix).weight",
            tensorDtype: .uint32,
            tensorShape: packedWeightShape));
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(tensorPrefix).scales",
            tensorDtype: .affineQuantizationFloat,
            tensorShape: scaleShape));
        tensorProfiles.append(qwen3_5TensorProfile(
            tensorName: "\(tensorPrefix).biases",
            tensorDtype: .affineQuantizationFloat,
            tensorShape: scaleShape));
    }

    static func qwen3_5TensorProfile(
        tensorName: String, tensorDtype: TensorDtype, tensorShape: Array<Int>) -> TensorProfile {
        return TensorProfile(
            name: tensorName, dtype: tensorDtype, shape: tensorShape,
            equivalentPublishedShapes: Array());
    }
}
