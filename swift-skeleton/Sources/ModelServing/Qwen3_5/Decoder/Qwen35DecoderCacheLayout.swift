import Foundation;


/// Exact live execution dtypes for one Qwen decoder layer's persistent
/// state, port of the Rust `Qwen3_5DecoderLayerCacheDtypes`.
public enum Qwen35DecoderLayerCacheDtypes: Equatable, Sendable {

    case linearAttention(convolution: DecoderCacheTensorDtype);

    case fullAttention(keys: DecoderCacheTensorDtype, values: DecoderCacheTensorDtype);
}

/// The persisted tensor-role vocabulary of the Qwen3.5 decoder cache.
enum Qwen35DecoderCacheTensorRole {

    static let CONVOLUTION: String = "linear.convolution";
    static let RECURRENCE: String = "linear.gated_delta_recurrent";
    static let ATTENTION_KEYS: String = "attention.keys";
    static let ATTENTION_VALUES: String = "attention.values";
}

/// Combines model geometry with dtypes derived from the bound execution
/// graph, port of the Rust `qwen3_5_decoder_cache_layout`. Geometry comes
/// from validated configuration, while scalar types come from actual weight
/// propagation. Keeping those inputs separate prevents a nominal activation
/// dtype from silently narrowing persistent state.
public enum Qwen35DecoderCacheLayoutBuilder {

    public static func buildDecoderCacheLayout(
        qwen35Config: Qwen3_5Config,
        fullAttentionKeyValueGrowthTokens: Int,
        decoderLayerCacheDtypes: [Qwen35DecoderLayerCacheDtypes]
    ) throws -> DecoderCacheLayout {
        let decoderLayerCount: Int = Int(qwen35Config.layerCount());
        if decoderLayerCacheDtypes.count != decoderLayerCount {
            throw DecoderCacheLayoutError.executionDtypeLayerCountMismatch(
                expectedLayerCount: decoderLayerCount,
                actualLayerCount: decoderLayerCacheDtypes.count);
        }
        let linearConvolutionStateDimension: Int = Int(
            qwen35Config.linearConvolutionStateDimension());
        let fullAttentionKeyValueDimensions: [Int] = [
            1,
            Int(qwen35Config.keyValueHeadCount()),
            0,
            Int(qwen35Config.headDimension()),
        ];
        let linearConvolutionDimensions: [Int] = [
            1,
            max(Int(qwen35Config.linearConvolutionKernelDimension()) - 1, 0),
            linearConvolutionStateDimension,
        ];
        let linearRecurrentDimensions: [Int] = [
            1,
            Int(qwen35Config.linearValueHeadCount()),
            Int(qwen35Config.linearValueHeadDimension()),
            Int(qwen35Config.linearKeyHeadDimension()),
        ];
        var decoderLayerLayouts: [DecoderCacheLayerLayout] = [];
        decoderLayerLayouts.reserveCapacity(decoderLayerCount);
        for (decoderLayerIndex, decoderLayerCacheDtypes): (Int, Qwen35DecoderLayerCacheDtypes)
            in decoderLayerCacheDtypes.enumerated() {
            let layerIsFullAttention: Bool = qwen35Config
                .decoderLayerIsFullAttention(decoderLayerIndex: decoderLayerIndex);
            switch (layerIsFullAttention, decoderLayerCacheDtypes) {
            case (true, .fullAttention(let keys, let values)):
                decoderLayerLayouts.append(.appendOnlyAttention(
                    keys: .sequence(
                        tensorRoleName: Qwen35DecoderCacheTensorRole.ATTENTION_KEYS,
                        dtype: keys,
                        dimensions: fullAttentionKeyValueDimensions,
                        sequenceAxis: 2),
                    values: .sequence(
                        tensorRoleName: Qwen35DecoderCacheTensorRole.ATTENTION_VALUES,
                        dtype: values,
                        dimensions: fullAttentionKeyValueDimensions,
                        sequenceAxis: 2),
                    capacityGrowthTokens: fullAttentionKeyValueGrowthTokens));
            case (false, .linearAttention(let convolution)):
                decoderLayerLayouts.append(.composite(components: [
                    .recurrentTensor(tensor: .fixed(
                        tensorRoleName: Qwen35DecoderCacheTensorRole.CONVOLUTION,
                        dtype: convolution,
                        dimensions: linearConvolutionDimensions)),
                    // The gated-delta recurrent accumulator is Float32 in
                    // live execution; preserve it independently from the
                    // activation state above.
                    .recurrentTensor(tensor: .fixed(
                        tensorRoleName: Qwen35DecoderCacheTensorRole.RECURRENCE,
                        dtype: .float32,
                        dimensions: linearRecurrentDimensions)),
                ]));
            default:
                throw DecoderCacheLayoutError.executionDtypeLayerFamilyMismatch(
                    layerIndex: decoderLayerIndex);
            }
        }
        return try DecoderCacheLayout(layers: decoderLayerLayouts);
    }
}
