import Foundation;

/**
 * Geometry wire documents for reviewed QwenImage21 package components,
 * ported from the geometry half of crates/config/src/model_discovery/
 * qwen_image_21_documents.rs. Each struct mirrors one component config.json
 * consumed by verify_model_directory validation.
 */

internal struct QuantizationGeometry: Equatable, Sendable {
    internal let bits: UInt8;
    internal let groupSize: UInt16;
    internal let mode: String;

    internal init(bits: UInt8, groupSize: UInt16, mode: String) {
        self.bits = bits;
        self.groupSize = groupSize;
        self.mode = mode;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> QuantizationGeometry {
        return QuantizationGeometry(
            bits: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "bits"),
            groupSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "group_size"),
            mode: try StrictJson.requiredString(object: jsonObject, fieldName: "mode")
        );
    }
}

internal struct QwenImage21TransformerGeometry: Equatable, Sendable {
    internal let className: String;
    internal let attentionHeadDim: UInt32;
    internal let axesDimsRope: Array<UInt32>;
    internal let causalCondition: Bool;
    internal let contextInDim: UInt32;
    internal let eps: Double;
    internal let inChannels: UInt32;
    internal let mlxFormat: Bool;
    internal let mlpRatio: UInt32;
    internal let numAttentionHeads: UInt32;
    internal let numLayers: UInt32;
    internal let outChannels: UInt32;
    internal let patchSize: UInt32;
    internal let quantization: QuantizationGeometry;

    internal init(
        className: String,
        attentionHeadDim: UInt32,
        axesDimsRope: Array<UInt32>,
        causalCondition: Bool,
        contextInDim: UInt32,
        eps: Double,
        inChannels: UInt32,
        mlxFormat: Bool,
        mlpRatio: UInt32,
        numAttentionHeads: UInt32,
        numLayers: UInt32,
        outChannels: UInt32,
        patchSize: UInt32,
        quantization: QuantizationGeometry
    ) {
        self.className = className;
        self.attentionHeadDim = attentionHeadDim;
        self.axesDimsRope = axesDimsRope;
        self.causalCondition = causalCondition;
        self.contextInDim = contextInDim;
        self.eps = eps;
        self.inChannels = inChannels;
        self.mlxFormat = mlxFormat;
        self.mlpRatio = mlpRatio;
        self.numAttentionHeads = numAttentionHeads;
        self.numLayers = numLayers;
        self.outChannels = outChannels;
        self.patchSize = patchSize;
        self.quantization = quantization;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> QwenImage21TransformerGeometry {
        let quantizationObject: Dictionary<String, Any> = try StrictJson.objectValue(
            object: jsonObject,
            fieldName: "quantization"
        );
        return QwenImage21TransformerGeometry(
            className: try StrictJson.requiredString(object: jsonObject, fieldName: "_class_name"),
            attentionHeadDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "attention_head_dim"),
            axesDimsRope: try QwenImage21WireJson.requiredFixedUnsignedIntegerArray(
                object: jsonObject,
                fieldName: "axes_dims_rope",
                exactElementCount: 3
            ),
            causalCondition: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "causal_condition"),
            contextInDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "context_in_dim"),
            eps: try QwenImage21WireJson.requiredDouble(object: jsonObject, fieldName: "eps"),
            inChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "in_channels"),
            mlxFormat: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "mlx_format"),
            mlpRatio: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "mlp_ratio"),
            numAttentionHeads: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_attention_heads"),
            numLayers: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_layers"),
            outChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "out_channels"),
            patchSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "patch_size"),
            quantization: try QuantizationGeometry.fromJsonObject(quantizationObject)
        );
    }
}

internal struct QwenImage21TextEncoderGeometry: Equatable, Sendable {
    internal let architectures: Array<String>;
    internal let dtype: String;
    internal let mlxFormat: Bool;
    internal let modelType: String;
    internal let quantization: QuantizationGeometry;
    internal let textConfig: Qwen3VlTextGeometry;
    internal let tieWordEmbeddings: Bool;

    internal init(
        architectures: Array<String>,
        dtype: String,
        mlxFormat: Bool,
        modelType: String,
        quantization: QuantizationGeometry,
        textConfig: Qwen3VlTextGeometry,
        tieWordEmbeddings: Bool
    ) {
        self.architectures = architectures;
        self.dtype = dtype;
        self.mlxFormat = mlxFormat;
        self.modelType = modelType;
        self.quantization = quantization;
        self.textConfig = textConfig;
        self.tieWordEmbeddings = tieWordEmbeddings;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> QwenImage21TextEncoderGeometry {
        let textConfigObject: Dictionary<String, Any> = try StrictJson.objectValue(
            object: jsonObject,
            fieldName: "text_config"
        );
        let quantizationObject: Dictionary<String, Any> = try StrictJson.objectValue(
            object: jsonObject,
            fieldName: "quantization"
        );
        return QwenImage21TextEncoderGeometry(
            architectures: try QwenImage21WireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "architectures",
                exactElementCount: 1
            ),
            dtype: try StrictJson.requiredString(object: jsonObject, fieldName: "dtype"),
            mlxFormat: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "mlx_format"),
            modelType: try StrictJson.requiredString(object: jsonObject, fieldName: "model_type"),
            quantization: try QuantizationGeometry.fromJsonObject(quantizationObject),
            textConfig: try Qwen3VlTextGeometry.fromJsonObject(textConfigObject),
            tieWordEmbeddings: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "tie_word_embeddings")
        );
    }
}

internal struct Qwen3VlTextGeometry: Equatable, Sendable {
    internal let attentionBias: Bool;
    internal let attentionDropout: Double;
    internal let dtype: String;
    internal let headDim: UInt32;
    internal let hiddenAct: String;
    internal let hiddenSize: UInt32;
    internal let intermediateSize: UInt32;
    internal let maxPositionEmbeddings: UInt32;
    internal let modelType: String;
    internal let numAttentionHeads: UInt32;
    internal let numHiddenLayers: UInt32;
    internal let numKeyValueHeads: UInt32;
    internal let rmsNormEps: Double;
    internal let ropeScaling: RopeScalingGeometry;
    internal let ropeTheta: UInt64;
    internal let useCache: Bool;
    internal let vocabSize: UInt32;

    internal init(
        attentionBias: Bool,
        attentionDropout: Double,
        dtype: String,
        headDim: UInt32,
        hiddenAct: String,
        hiddenSize: UInt32,
        intermediateSize: UInt32,
        maxPositionEmbeddings: UInt32,
        modelType: String,
        numAttentionHeads: UInt32,
        numHiddenLayers: UInt32,
        numKeyValueHeads: UInt32,
        rmsNormEps: Double,
        ropeScaling: RopeScalingGeometry,
        ropeTheta: UInt64,
        useCache: Bool,
        vocabSize: UInt32
    ) {
        self.attentionBias = attentionBias;
        self.attentionDropout = attentionDropout;
        self.dtype = dtype;
        self.headDim = headDim;
        self.hiddenAct = hiddenAct;
        self.hiddenSize = hiddenSize;
        self.intermediateSize = intermediateSize;
        self.maxPositionEmbeddings = maxPositionEmbeddings;
        self.modelType = modelType;
        self.numAttentionHeads = numAttentionHeads;
        self.numHiddenLayers = numHiddenLayers;
        self.numKeyValueHeads = numKeyValueHeads;
        self.rmsNormEps = rmsNormEps;
        self.ropeScaling = ropeScaling;
        self.ropeTheta = ropeTheta;
        self.useCache = useCache;
        self.vocabSize = vocabSize;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> Qwen3VlTextGeometry {
        let ropeScalingObject: Dictionary<String, Any> = try StrictJson.objectValue(
            object: jsonObject,
            fieldName: "rope_scaling"
        );
        return Qwen3VlTextGeometry(
            attentionBias: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "attention_bias"),
            attentionDropout: try QwenImage21WireJson.requiredDouble(object: jsonObject, fieldName: "attention_dropout"),
            dtype: try StrictJson.requiredString(object: jsonObject, fieldName: "dtype"),
            headDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "head_dim"),
            hiddenAct: try StrictJson.requiredString(object: jsonObject, fieldName: "hidden_act"),
            hiddenSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "hidden_size"),
            intermediateSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "intermediate_size"),
            maxPositionEmbeddings: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "max_position_embeddings"),
            modelType: try StrictJson.requiredString(object: jsonObject, fieldName: "model_type"),
            numAttentionHeads: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_attention_heads"),
            numHiddenLayers: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_hidden_layers"),
            numKeyValueHeads: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_key_value_heads"),
            rmsNormEps: try QwenImage21WireJson.requiredDouble(object: jsonObject, fieldName: "rms_norm_eps"),
            ropeScaling: try RopeScalingGeometry.fromJsonObject(ropeScalingObject),
            ropeTheta: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "rope_theta"),
            useCache: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_cache"),
            vocabSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "vocab_size")
        );
    }
}

internal struct RopeScalingGeometry: Equatable, Sendable {
    internal let mropeInterleaved: Bool;
    internal let mropeSection: Array<UInt32>;
    internal let ropeType: String;

    internal init(mropeInterleaved: Bool, mropeSection: Array<UInt32>, ropeType: String) {
        self.mropeInterleaved = mropeInterleaved;
        self.mropeSection = mropeSection;
        self.ropeType = ropeType;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> RopeScalingGeometry {
        return RopeScalingGeometry(
            mropeInterleaved: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "mrope_interleaved"),
            mropeSection: try QwenImage21WireJson.requiredFixedUnsignedIntegerArray(
                object: jsonObject,
                fieldName: "mrope_section",
                exactElementCount: 3
            ),
            ropeType: try StrictJson.requiredString(object: jsonObject, fieldName: "rope_type")
        );
    }
}

internal struct QwenImage21VaeGeometry: Equatable, Sendable {
    internal let className: String;
    internal let attnScales: Array<UInt32>;
    internal let baseDim: UInt32;
    internal let decoderBaseDim: UInt32;
    internal let dimMult: Array<UInt32>;
    internal let dropout: Double;
    internal let inChannels: UInt32;
    internal let isResidual: Bool;
    internal let latentsMean: Array<Double>;
    internal let latentsStd: Array<Double>;
    internal let mlxFormat: Bool;
    internal let numResBlocks: UInt32;
    internal let outChannels: UInt32;
    internal let patchSize: UInt32?;
    internal let scaleFactorSpatial: UInt32;
    internal let scaleFactorTemporal: UInt32;
    internal let temporalDownsample: Array<Bool>;
    internal let zDim: UInt32;

    internal init(
        className: String,
        attnScales: Array<UInt32>,
        baseDim: UInt32,
        decoderBaseDim: UInt32,
        dimMult: Array<UInt32>,
        dropout: Double,
        inChannels: UInt32,
        isResidual: Bool,
        latentsMean: Array<Double>,
        latentsStd: Array<Double>,
        mlxFormat: Bool,
        numResBlocks: UInt32,
        outChannels: UInt32,
        patchSize: UInt32?,
        scaleFactorSpatial: UInt32,
        scaleFactorTemporal: UInt32,
        temporalDownsample: Array<Bool>,
        zDim: UInt32
    ) {
        self.className = className;
        self.attnScales = attnScales;
        self.baseDim = baseDim;
        self.decoderBaseDim = decoderBaseDim;
        self.dimMult = dimMult;
        self.dropout = dropout;
        self.inChannels = inChannels;
        self.isResidual = isResidual;
        self.latentsMean = latentsMean;
        self.latentsStd = latentsStd;
        self.mlxFormat = mlxFormat;
        self.numResBlocks = numResBlocks;
        self.outChannels = outChannels;
        self.patchSize = patchSize;
        self.scaleFactorSpatial = scaleFactorSpatial;
        self.scaleFactorTemporal = scaleFactorTemporal;
        self.temporalDownsample = temporalDownsample;
        self.zDim = zDim;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> QwenImage21VaeGeometry {
        return QwenImage21VaeGeometry(
            className: try StrictJson.requiredString(object: jsonObject, fieldName: "_class_name"),
            attnScales: try QwenImage21WireJson.requiredUnsignedIntegerArray(object: jsonObject, fieldName: "attn_scales"),
            baseDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "base_dim"),
            decoderBaseDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "decoder_base_dim"),
            dimMult: try QwenImage21WireJson.requiredFixedUnsignedIntegerArray(
                object: jsonObject,
                fieldName: "dim_mult",
                exactElementCount: 5
            ),
            dropout: try QwenImage21WireJson.requiredDouble(object: jsonObject, fieldName: "dropout"),
            inChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "in_channels"),
            isResidual: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "is_residual"),
            latentsMean: try QwenImage21WireJson.requiredDoubleArray(object: jsonObject, fieldName: "latents_mean"),
            latentsStd: try QwenImage21WireJson.requiredDoubleArray(object: jsonObject, fieldName: "latents_std"),
            mlxFormat: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "mlx_format"),
            numResBlocks: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_res_blocks"),
            outChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "out_channels"),
            patchSize: try StrictJson.optionalUnsignedInteger(object: jsonObject, fieldName: "patch_size"),
            scaleFactorSpatial: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "scale_factor_spatial"),
            scaleFactorTemporal: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "scale_factor_temporal"),
            temporalDownsample: try QwenImage21WireJson.requiredFixedBooleanArray(
                object: jsonObject,
                fieldName: "temperal_downsample",
                exactElementCount: 4
            ),
            zDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "z_dim")
        );
    }
}

internal struct QwenImage21SchedulerGeometry: Equatable, Sendable {
    internal let className: String;
    internal let baseImageSeqLen: UInt32;
    internal let baseShift: Double;
    internal let invertSigmas: Bool;
    internal let maxImageSeqLen: UInt32;
    internal let maxShift: Double;
    internal let numTrainTimesteps: UInt32;
    internal let shift: Double;
    internal let shiftTerminal: Double?;
    internal let stochasticSampling: Bool;
    internal let timeShiftType: String;
    internal let useBetaSigmas: Bool;
    internal let useDynamicShifting: Bool;
    internal let useExponentialSigmas: Bool;
    internal let useKarrasSigmas: Bool;

    internal init(
        className: String,
        baseImageSeqLen: UInt32,
        baseShift: Double,
        invertSigmas: Bool,
        maxImageSeqLen: UInt32,
        maxShift: Double,
        numTrainTimesteps: UInt32,
        shift: Double,
        shiftTerminal: Double?,
        stochasticSampling: Bool,
        timeShiftType: String,
        useBetaSigmas: Bool,
        useDynamicShifting: Bool,
        useExponentialSigmas: Bool,
        useKarrasSigmas: Bool
    ) {
        self.className = className;
        self.baseImageSeqLen = baseImageSeqLen;
        self.baseShift = baseShift;
        self.invertSigmas = invertSigmas;
        self.maxImageSeqLen = maxImageSeqLen;
        self.maxShift = maxShift;
        self.numTrainTimesteps = numTrainTimesteps;
        self.shift = shift;
        self.shiftTerminal = shiftTerminal;
        self.stochasticSampling = stochasticSampling;
        self.timeShiftType = timeShiftType;
        self.useBetaSigmas = useBetaSigmas;
        self.useDynamicShifting = useDynamicShifting;
        self.useExponentialSigmas = useExponentialSigmas;
        self.useKarrasSigmas = useKarrasSigmas;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> QwenImage21SchedulerGeometry {
        return QwenImage21SchedulerGeometry(
            className: try StrictJson.requiredString(object: jsonObject, fieldName: "_class_name"),
            baseImageSeqLen: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "base_image_seq_len"),
            baseShift: try QwenImage21WireJson.requiredDouble(object: jsonObject, fieldName: "base_shift"),
            invertSigmas: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "invert_sigmas"),
            maxImageSeqLen: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "max_image_seq_len"),
            maxShift: try QwenImage21WireJson.requiredDouble(object: jsonObject, fieldName: "max_shift"),
            numTrainTimesteps: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_train_timesteps"),
            shift: try QwenImage21WireJson.requiredDouble(object: jsonObject, fieldName: "shift"),
            shiftTerminal: try QwenImage21WireJson.optionalDouble(object: jsonObject, fieldName: "shift_terminal"),
            stochasticSampling: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "stochastic_sampling"),
            timeShiftType: try StrictJson.requiredString(object: jsonObject, fieldName: "time_shift_type"),
            useBetaSigmas: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_beta_sigmas"),
            useDynamicShifting: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_dynamic_shifting"),
            useExponentialSigmas: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_exponential_sigmas"),
            useKarrasSigmas: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_karras_sigmas")
        );
    }
}
