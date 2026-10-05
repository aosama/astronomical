import Foundation;

/**
 * Wire shapes of the FLUX.2-klein component config documents (transformer,
 * text encoder, VAE, scheduler). These mirror the Rust serde structs
 * one-for-one, including exact field sets (deny_unknown_fields upstream).
 */
internal struct TransformerGeometry: Equatable, Sendable {
    internal let className: String;
    internal let attentionHeadDim: UInt32;
    internal let axesDimsRope: Array<UInt32>;
    internal let eps: Double;
    internal let guidanceEmbeds: Bool;
    internal let inChannels: UInt32;
    internal let jointAttentionDim: UInt32;
    internal let mlpRatio: Double;
    internal let numAttentionHeads: UInt32;
    internal let numLayers: UInt32;
    internal let numSingleLayers: UInt32;
    internal let outChannels: UInt32?;
    internal let patchSize: UInt32;
    internal let ropeTheta: Double;
    internal let timestepGuidanceChannels: UInt32;

    internal init(
        className: String,
        attentionHeadDim: UInt32,
        axesDimsRope: Array<UInt32>,
        eps: Double,
        guidanceEmbeds: Bool,
        inChannels: UInt32,
        jointAttentionDim: UInt32,
        mlpRatio: Double,
        numAttentionHeads: UInt32,
        numLayers: UInt32,
        numSingleLayers: UInt32,
        outChannels: UInt32?,
        patchSize: UInt32,
        ropeTheta: Double,
        timestepGuidanceChannels: UInt32
    ) {
        self.className = className;
        self.attentionHeadDim = attentionHeadDim;
        self.axesDimsRope = axesDimsRope;
        self.eps = eps;
        self.guidanceEmbeds = guidanceEmbeds;
        self.inChannels = inChannels;
        self.jointAttentionDim = jointAttentionDim;
        self.mlpRatio = mlpRatio;
        self.numAttentionHeads = numAttentionHeads;
        self.numLayers = numLayers;
        self.numSingleLayers = numSingleLayers;
        self.outChannels = outChannels;
        self.patchSize = patchSize;
        self.ropeTheta = ropeTheta;
        self.timestepGuidanceChannels = timestepGuidanceChannels;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> TransformerGeometry {
        return TransformerGeometry(
            className: try StrictJson.requiredString(object: jsonObject, fieldName: "_class_name"),
            attentionHeadDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "attention_head_dim"),
            axesDimsRope: try Flux2KleinWireJson.requiredFixedUnsignedIntegerArray(
                object: jsonObject,
                fieldName: "axes_dims_rope",
                exactElementCount: 4
            ),
            eps: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "eps"),
            guidanceEmbeds: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "guidance_embeds"),
            inChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "in_channels"),
            jointAttentionDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "joint_attention_dim"),
            mlpRatio: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "mlp_ratio"),
            numAttentionHeads: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_attention_heads"),
            numLayers: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_layers"),
            numSingleLayers: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_single_layers"),
            outChannels: try StrictJson.optionalUnsignedInteger(object: jsonObject, fieldName: "out_channels"),
            patchSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "patch_size"),
            ropeTheta: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "rope_theta"),
            timestepGuidanceChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "timestep_guidance_channels")
        );
    }
}

internal struct TextEncoderGeometry: Equatable, Sendable {
    internal let architectures: Array<String>;
    internal let attentionBias: Bool;
    internal let attentionDropout: Double;
    internal let dtype: String;
    internal let headDim: UInt32;
    internal let hiddenAct: String;
    internal let hiddenSize: UInt32;
    internal let intermediateSize: UInt32;
    internal let layerTypes: Array<String>;
    internal let maxPositionEmbeddings: UInt32;
    internal let maxWindowLayers: UInt32;
    internal let modelType: String;
    internal let numAttentionHeads: UInt32;
    internal let numHiddenLayers: UInt32;
    internal let numKeyValueHeads: UInt32;
    internal let rmsNormEps: Double;
    internal let ropeScaling: ValueMarker?;
    internal let ropeTheta: Double;
    internal let slidingWindow: UInt32?;
    internal let tieWordEmbeddings: Bool;
    internal let useCache: Bool;
    internal let useSlidingWindow: Bool;
    internal let vocabSize: UInt32;

    internal init(
        architectures: Array<String>,
        attentionBias: Bool,
        attentionDropout: Double,
        dtype: String,
        headDim: UInt32,
        hiddenAct: String,
        hiddenSize: UInt32,
        intermediateSize: UInt32,
        layerTypes: Array<String>,
        maxPositionEmbeddings: UInt32,
        maxWindowLayers: UInt32,
        modelType: String,
        numAttentionHeads: UInt32,
        numHiddenLayers: UInt32,
        numKeyValueHeads: UInt32,
        rmsNormEps: Double,
        ropeScaling: ValueMarker?,
        ropeTheta: Double,
        slidingWindow: UInt32?,
        tieWordEmbeddings: Bool,
        useCache: Bool,
        useSlidingWindow: Bool,
        vocabSize: UInt32
    ) {
        self.architectures = architectures;
        self.attentionBias = attentionBias;
        self.attentionDropout = attentionDropout;
        self.dtype = dtype;
        self.headDim = headDim;
        self.hiddenAct = hiddenAct;
        self.hiddenSize = hiddenSize;
        self.intermediateSize = intermediateSize;
        self.layerTypes = layerTypes;
        self.maxPositionEmbeddings = maxPositionEmbeddings;
        self.maxWindowLayers = maxWindowLayers;
        self.modelType = modelType;
        self.numAttentionHeads = numAttentionHeads;
        self.numHiddenLayers = numHiddenLayers;
        self.numKeyValueHeads = numKeyValueHeads;
        self.rmsNormEps = rmsNormEps;
        self.ropeScaling = ropeScaling;
        self.ropeTheta = ropeTheta;
        self.slidingWindow = slidingWindow;
        self.tieWordEmbeddings = tieWordEmbeddings;
        self.useCache = useCache;
        self.useSlidingWindow = useSlidingWindow;
        self.vocabSize = vocabSize;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> TextEncoderGeometry {
        return TextEncoderGeometry(
            architectures: try Flux2KleinWireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "architectures",
                exactElementCount: 1
            ),
            attentionBias: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "attention_bias"),
            attentionDropout: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "attention_dropout"),
            dtype: try StrictJson.requiredString(object: jsonObject, fieldName: "dtype"),
            headDim: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "head_dim"),
            hiddenAct: try StrictJson.requiredString(object: jsonObject, fieldName: "hidden_act"),
            hiddenSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "hidden_size"),
            intermediateSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "intermediate_size"),
            layerTypes: try StrictJson.requiredStringArray(object: jsonObject, fieldName: "layer_types"),
            maxPositionEmbeddings: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "max_position_embeddings"),
            maxWindowLayers: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "max_window_layers"),
            modelType: try StrictJson.requiredString(object: jsonObject, fieldName: "model_type"),
            numAttentionHeads: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_attention_heads"),
            numHiddenLayers: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_hidden_layers"),
            numKeyValueHeads: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_key_value_heads"),
            rmsNormEps: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "rms_norm_eps"),
            ropeScaling: try Flux2KleinWireJson.optionalValueMarker(object: jsonObject, fieldName: "rope_scaling"),
            ropeTheta: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "rope_theta"),
            slidingWindow: try StrictJson.optionalUnsignedInteger(object: jsonObject, fieldName: "sliding_window"),
            tieWordEmbeddings: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "tie_word_embeddings"),
            useCache: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_cache"),
            useSlidingWindow: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_sliding_window"),
            vocabSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "vocab_size")
        );
    }
}

internal struct VaeGeometry: Equatable, Sendable {
    internal let className: String;
    internal let actFn: String;
    internal let batchNormEps: Double;
    internal let batchNormMomentum: Double;
    internal let blockOutChannels: Array<UInt32>;
    internal let downBlockTypes: Array<String>;
    internal let forceUpcast: Bool;
    internal let inChannels: UInt32;
    internal let latentChannels: UInt32;
    internal let layersPerBlock: UInt32;
    internal let midBlockAddAttention: Bool;
    internal let normNumGroups: UInt32;
    internal let outChannels: UInt32;
    internal let patchSize: Array<UInt32>;
    internal let sampleSize: UInt32;
    internal let upBlockTypes: Array<String>;
    internal let usePostQuantConv: Bool;
    internal let useQuantConv: Bool;

    internal init(
        className: String,
        actFn: String,
        batchNormEps: Double,
        batchNormMomentum: Double,
        blockOutChannels: Array<UInt32>,
        downBlockTypes: Array<String>,
        forceUpcast: Bool,
        inChannels: UInt32,
        latentChannels: UInt32,
        layersPerBlock: UInt32,
        midBlockAddAttention: Bool,
        normNumGroups: UInt32,
        outChannels: UInt32,
        patchSize: Array<UInt32>,
        sampleSize: UInt32,
        upBlockTypes: Array<String>,
        usePostQuantConv: Bool,
        useQuantConv: Bool
    ) {
        self.className = className;
        self.actFn = actFn;
        self.batchNormEps = batchNormEps;
        self.batchNormMomentum = batchNormMomentum;
        self.blockOutChannels = blockOutChannels;
        self.downBlockTypes = downBlockTypes;
        self.forceUpcast = forceUpcast;
        self.inChannels = inChannels;
        self.latentChannels = latentChannels;
        self.layersPerBlock = layersPerBlock;
        self.midBlockAddAttention = midBlockAddAttention;
        self.normNumGroups = normNumGroups;
        self.outChannels = outChannels;
        self.patchSize = patchSize;
        self.sampleSize = sampleSize;
        self.upBlockTypes = upBlockTypes;
        self.usePostQuantConv = usePostQuantConv;
        self.useQuantConv = useQuantConv;
    }

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> VaeGeometry {
        return VaeGeometry(
            className: try StrictJson.requiredString(object: jsonObject, fieldName: "_class_name"),
            actFn: try StrictJson.requiredString(object: jsonObject, fieldName: "act_fn"),
            batchNormEps: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "batch_norm_eps"),
            batchNormMomentum: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "batch_norm_momentum"),
            blockOutChannels: try Flux2KleinWireJson.requiredFixedUnsignedIntegerArray(
                object: jsonObject,
                fieldName: "block_out_channels",
                exactElementCount: 4
            ),
            downBlockTypes: try Flux2KleinWireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "down_block_types",
                exactElementCount: 4
            ),
            forceUpcast: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "force_upcast"),
            inChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "in_channels"),
            latentChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "latent_channels"),
            layersPerBlock: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "layers_per_block"),
            midBlockAddAttention: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "mid_block_add_attention"),
            normNumGroups: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "norm_num_groups"),
            outChannels: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "out_channels"),
            patchSize: try Flux2KleinWireJson.requiredFixedUnsignedIntegerArray(
                object: jsonObject,
                fieldName: "patch_size",
                exactElementCount: 2
            ),
            sampleSize: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "sample_size"),
            upBlockTypes: try Flux2KleinWireJson.requiredFixedStringArray(
                object: jsonObject,
                fieldName: "up_block_types",
                exactElementCount: 4
            ),
            usePostQuantConv: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_post_quant_conv"),
            useQuantConv: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_quant_conv")
        );
    }
}

internal struct SchedulerGeometry: Equatable, Sendable {
    internal let className: String;
    internal let baseImageSeqLength: UInt32;
    internal let baseShift: Double;
    internal let invertSigmas: Bool;
    internal let maxImageSeqLength: UInt32;
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
        baseImageSeqLength: UInt32,
        baseShift: Double,
        invertSigmas: Bool,
        maxImageSeqLength: UInt32,
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
        self.baseImageSeqLength = baseImageSeqLength;
        self.baseShift = baseShift;
        self.invertSigmas = invertSigmas;
        self.maxImageSeqLength = maxImageSeqLength;
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

    internal static func fromJsonObject(_ jsonObject: Dictionary<String, Any>) throws -> SchedulerGeometry {
        return SchedulerGeometry(
            className: try StrictJson.requiredString(object: jsonObject, fieldName: "_class_name"),
            baseImageSeqLength: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "base_image_seq_len"),
            baseShift: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "base_shift"),
            invertSigmas: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "invert_sigmas"),
            maxImageSeqLength: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "max_image_seq_len"),
            maxShift: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "max_shift"),
            numTrainTimesteps: try StrictJson.requiredUnsignedInteger(object: jsonObject, fieldName: "num_train_timesteps"),
            shift: try Flux2KleinWireJson.requiredDouble(object: jsonObject, fieldName: "shift"),
            shiftTerminal: try Flux2KleinWireJson.optionalDouble(object: jsonObject, fieldName: "shift_terminal"),
            stochasticSampling: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "stochastic_sampling"),
            timeShiftType: try StrictJson.requiredString(object: jsonObject, fieldName: "time_shift_type"),
            useBetaSigmas: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_beta_sigmas"),
            useDynamicShifting: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_dynamic_shifting"),
            useExponentialSigmas: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_exponential_sigmas"),
            useKarrasSigmas: try StrictJson.requiredBoolean(object: jsonObject, fieldName: "use_karras_sigmas")
        );
    }
}
