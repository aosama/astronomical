import Foundation;

/// Typed accessors over the validated Qwen3.5 configuration, port of
/// crates/model-serving/src/qwen3_5/configuration/config_accessors.rs and
/// config_memory.rs.
extension Qwen3_5Config {

    /// Returns the activation dtype declared by the model config.
    public func activationDtype() -> String {
        return self.activationDtypeName;
    }

    /// Returns the feed-forward architecture declared by this checkpoint.
    public func feedForwardArchitecture() -> Qwen3_5FeedForwardArchitecture {
        return self.feedForwardArchitectureValue;
    }

    /// Returns the intermediate width for a dense Qwen3.5 SwiGLU MLP.
    public func denseIntermediateSize() -> UInt32 {
        return self.textConfigValue.intermediateSize;
    }

    /// Returns the validated OptiQ quantization profile for every executable quantized module.
    public func quantizedModuleProfiles() -> SortedModuleProfiles {
        return self.quantizedModuleProfileMap;
    }

    /// Returns the artifact-wide executable weight representation.
    public func modelWeightStorage() -> ModelWeightStorage {
        return self.modelWeightStorageValue;
    }

    /// Returns the quantization profile for a specific module, falling back to the
    /// `mtplx_mtp_quantization` global default for MTP modules, then the model-wide
    /// default profile if the module is not found in any override map.
    public func quantizationProfile(forModule moduleName: String) -> OptiQQuantizationProfile {
        if let mtpProfile: OptiQQuantizationProfile = self.mtpQuantizedModuleProfileMap.profile(forKey: moduleName) {
            return mtpProfile;
        }
        if let moduleProfile: OptiQQuantizationProfile = self.quantizedModuleProfileMap.profile(forKey: moduleName) {
            return moduleProfile;
        }
        // MTP modules without a per-module override get the global
        // mtplx_mtp_quantization fallback when declared.
        if moduleName.hasPrefix("language_model.mtp."),
            let fallback: MtplxMtpQuantizationFallback = self.mtxplxMtpQuantizationFallback {
            return OptiQQuantizationProfile(bits: fallback.bits, groupSize: fallback.groupSize);
        }
        return OptiQQuantizationProfile(
            bits: self.defaultQuantizationBitsValue,
            groupSize: self.defaultQuantizationGroupSizeValue);
    }

    /// Returns the default quantization bit width for modules not in the override map.
    public func defaultQuantizationBits() -> UInt32 {
        return self.defaultQuantizationBitsValue;
    }

    /// Resolves modules stored as native floating-point when both affine companion
    /// tensors are absent from the safetensors index. This handles mixed storage
    /// artifacts whose default quantization profile does not describe every module,
    /// including optional MTP modules.
    ///
    /// The shardTensorNames parameter should contain all tensor names from the
    /// safetensors index (the weight_map keys).
    public mutating func resolveUnquantizedModulesFromShardIndex(shardTensorNames: Set<String>) {
        var nativeModuleNames: Array<String> = Array();
        for tensorName: String in shardTensorNames {
            guard tensorName.hasSuffix(".weight") else {
                continue;
            }
            let moduleName: String = String(tensorName.dropLast(".weight".count));
            if shardTensorNames.contains("\(moduleName).scales")
                || shardTensorNames.contains("\(moduleName).biases") {
                continue;
            }
            nativeModuleNames.append(moduleName);
        }
        for nativeModuleName: String in nativeModuleNames {
            let nativeQuantizationProfile: OptiQQuantizationProfile = .unquantized();
            if nativeModuleName.hasPrefix("language_model.mtp.") {
                self.mtpQuantizedModuleProfileMap.insert(profile: nativeQuantizationProfile, forKey: nativeModuleName);
            } else {
                self.quantizedModuleProfileMap.insert(profile: nativeQuantizationProfile, forKey: nativeModuleName);
            }
        }
    }

    /// Returns the default quantization group size for modules not in the override map.
    public func defaultQuantizationGroupSize() -> UInt32 {
        return self.defaultQuantizationGroupSizeValue;
    }

    /// Returns the activation dtype declared by the model config.
    public func torchDtype() -> String {
        return self.activationDtype();
    }

    /// Returns the declared MLP activation.
    public func hiddenActivation() -> String {
        return self.textConfigValue.hiddenAct;
    }

    /// Returns the declared text hidden dimension.
    public func hiddenSize() -> UInt32 {
        return self.textConfigValue.hiddenSize;
    }

    /// Returns the declared text decoder-layer count.
    public func layerCount() -> UInt32 {
        return self.textConfigValue.numHiddenLayers;
    }

    /// Returns the declared tokenizer vocabulary size.
    public func vocabularySize() -> UInt32 {
        return self.textConfigValue.vocabSize;
    }

    /// Returns the native combined prompt and generation position count.
    public func maximumPositionCount() -> UInt32 {
        return self.textConfigValue.maxPositionEmbeddings;
    }

    /// Returns the exact RMSNorm epsilon bits.
    public func rmsNormEpsilonBits() -> UInt32 {
        return self.textConfigValue.rmsNormEpsilonBits;
    }

    /// Returns the exact RoPE base bits.
    public func ropeThetaBits() -> UInt32 {
        return self.textConfigValue.ropeParameters?.ropeThetaBits
            ?? self.textConfigValue.legacyRopeThetaBits
            ?? 0;
    }

    /// Returns the exact partial rotary factor bits.
    /// Falls back to rope_parameters.partial_rotary_factor when the top-level
    /// text_config.partial_rotary_factor is absent, matching the Qwen3.5
    /// configuration compatibility rule above.
    public func partialRotaryFactorBits() -> UInt32 {
        return self.textConfigValue.partialRotaryFactorBits
            ?? self.textConfigValue.ropeParameters?.partialRotaryFactor
            ?? 0;
    }

    /// Returns every stop-token ID declared by the model config.
    public func endOfSequenceTokenIds() -> Array<UInt32> {
        return self.eosTokenIds;
    }

    /// Returns whether attention projections use bias terms.
    public func hasAttentionBias() -> Bool {
        return self.textConfigValue.attentionBias;
    }

    /// Returns whether MLP projections use bias terms.
    public func hasMlpBias() -> Bool {
        return self.textConfigValue.mlpBias;
    }

    /// Returns whether input and output embeddings are tied.
    public func hasTiedEmbeddings() -> Bool {
        return self.hasTiedEmbeddingsFlag;
    }

    /// Returns whether selected MoE router probabilities are normalized.
    public func normalizesTopKProbabilities() -> Bool {
        return self.textConfigValue.normTopkProb;
    }

    /// Returns the decoder-layer indexes that use full attention.
    public func fullAttentionDecoderLayerIndexes() -> Array<Int> {
        var fullAttentionIndexes: Array<Int> = Array();
        for (decoderLayerIndex, decoderLayerType): (Int, String) in self.textConfigValue.layerTypes.enumerated() {
            if decoderLayerType == "full_attention" {
                fullAttentionIndexes.append(decoderLayerIndex);
            }
        }
        return fullAttentionIndexes;
    }

    /// Returns the decoder-layer indexes that use linear attention.
    public func linearAttentionDecoderLayerIndexes() -> Array<Int> {
        var linearAttentionIndexes: Array<Int> = Array();
        for (decoderLayerIndex, decoderLayerType): (Int, String) in self.textConfigValue.layerTypes.enumerated() {
            if decoderLayerType == "linear_attention" {
                linearAttentionIndexes.append(decoderLayerIndex);
            }
        }
        return linearAttentionIndexes;
    }

    /// Returns whether the validated decoder layer uses full attention.
    public func decoderLayerIsFullAttention(decoderLayerIndex: Int) -> Bool {
        return self.textConfigValue.decoderLayerIsFullAttention(decoderLayerIndex: decoderLayerIndex);
    }

    /// Returns the full-attention key/value head count.
    public func keyValueHeadCount() -> UInt32 {
        return self.textConfigValue.numKeyValueHeads;
    }

    /// Returns the full-attention key/value head dimension.
    public func headDimension() -> UInt32 {
        return self.textConfigValue.headDim;
    }

    /// Returns the linear-attention convolution kernel dimension.
    public func linearConvolutionKernelDimension() -> UInt32 {
        return self.textConfigValue.linearConvKernelDim;
    }

    /// Returns the linear-attention key head count.
    public func linearKeyHeadCount() -> UInt32 {
        return self.textConfigValue.linearNumKeyHeads;
    }

    /// Returns the linear-attention value head count.
    public func linearValueHeadCount() -> UInt32 {
        return self.textConfigValue.linearNumValueHeads;
    }

    /// Returns the linear-attention key head dimension.
    public func linearKeyHeadDimension() -> UInt32 {
        return self.textConfigValue.linearKeyHeadDim;
    }

    /// Returns the linear-attention value head dimension.
    public func linearValueHeadDimension() -> UInt32 {
        return self.textConfigValue.linearValueHeadDim;
    }

    /// Returns the validated convolution-state width used by linear attention.
    public func linearConvolutionStateDimension() -> Int32 {
        return Int32(truncatingIfNeeded: self.textConfigValue.linearConvolutionStateDimension());
    }

    /// Returns the full-attention query head count.
    public func queryHeadCount() -> UInt32 {
        return self.textConfigValue.numAttentionHeads;
    }

    /// Returns the total number of sparse experts.
    public func expertCount() -> UInt32 {
        return self.textConfigValue.numExperts;
    }

    /// Returns the number of experts selected per token.
    public func expertsPerToken() -> UInt32 {
        return self.textConfigValue.numExpertsPerTok;
    }

    /// Returns the per-expert feed-forward intermediate size.
    public func expertIntermediateSize() -> UInt32 {
        return self.textConfigValue.moeIntermediateSize;
    }

    /// Returns the shared expert feed-forward intermediate size.
    public func sharedExpertIntermediateSize() -> UInt32 {
        return self.textConfigValue.sharedExpertIntermediateSize;
    }

    /// Returns the linear-attention key dimension (key heads * key head dim).
    public func linearKeyDimension() -> UInt32 {
        return self.textConfigValue.linearNumKeyHeads
            .multipliedReportingOverflow(by: self.textConfigValue.linearKeyHeadDim).partialValue;
    }

    /// Returns the linear-attention value dimension (value heads * value head dim).
    public func linearValueDimension() -> UInt32 {
        return self.textConfigValue.linearNumValueHeads
            .multipliedReportingOverflow(by: self.textConfigValue.linearValueHeadDim).partialValue;
    }

    /// Returns the linear-attention convolution dimension
    /// (2 * key dimension + value dimension). The Rust original saturates.
    public func linearConvolutionDimension() -> UInt32 {
        let (doubledKeyDimension, keyOverflow) = self.linearKeyDimension()
            .multipliedReportingOverflow(by: 2);
        if keyOverflow {
            return UInt32.max;
        }
        let (convolutionDimension, valueOverflow) = doubledKeyDimension
            .addingReportingOverflow(self.linearValueDimension());
        return valueOverflow ? UInt32.max : convolutionDimension;
    }

    /// Returns the RoPE rotary dimension (head_dim * partial_rotary_factor).
    public func rotaryDimension() -> UInt32 {
        let partialRotaryFactor: Float = Float(bitPattern: self.partialRotaryFactorBits());
        return UInt32(Float(self.textConfigValue.headDim) * partialRotaryFactor);
    }

    /// Returns the artifact-declared MTP layer count.
    public func mtpLayerCount() -> UInt32 {
        return self.textConfigValue.mtpNumHiddenLayers;
    }

    /// Returns the MTP sidecar file path declared in `mlx_lm_extra_tensors.mtp_file`,
    /// or `nil` when MTP weights are embedded in the shard index or absent.
    public func sidecarMtpFile() -> String? {
        return self.sidecarMtpFileValue;
    }

    /// Reserves context-growing full-attention key/value state for each context token.
    /// Returns nil on any overflow, mirroring the Rust checked arithmetic.
    public func contextMemoryReservationBytes(contextTokenCount: Int) -> Int? {
        let bfloat16ElementSizeBytes: Int = 2;
        let fullAttentionKeyValueStateTensorCount: Int = 2;
        let fullAttentionLayerCount: Int = self.fullAttentionDecoderLayerIndexes().count;
        let (layerTokenCount, layerOverflow) = contextTokenCount
            .multipliedReportingOverflow(by: fullAttentionLayerCount);
        if layerOverflow {
            return nil;
        }
        let (perLayerTokenCount, tensorOverflow) = layerTokenCount
            .multipliedReportingOverflow(by: fullAttentionKeyValueStateTensorCount);
        if tensorOverflow {
            return nil;
        }
        let (perHeadTokenCount, headOverflow) = perLayerTokenCount
            .multipliedReportingOverflow(by: Int(self.keyValueHeadCount()));
        if headOverflow {
            return nil;
        }
        let (perDimensionTokenCount, dimensionOverflow) = perHeadTokenCount
            .multipliedReportingOverflow(by: Int(self.headDimension()));
        if dimensionOverflow {
            return nil;
        }
        let (totalReservationBytes, byteOverflow) = perDimensionTokenCount
            .multipliedReportingOverflow(by: bfloat16ElementSizeBytes);
        return byteOverflow ? nil : totalReservationBytes;
    }

    /// Returns one full-attention layer's exact key/value state bytes per token.
    /// Returns nil on any overflow, mirroring the Rust checked arithmetic.
    public func fullAttentionKeyValueStateBytesPerLayerToken() -> Int? {
        let bfloat16ElementSizeBytes: Int = 2;
        let fullAttentionKeyValueStateTensorCount: Int = 2;
        let (perHeadBytes, headOverflow) = fullAttentionKeyValueStateTensorCount
            .multipliedReportingOverflow(by: Int(self.keyValueHeadCount()));
        if headOverflow {
            return nil;
        }
        let (perDimensionBytes, dimensionOverflow) = perHeadBytes
            .multipliedReportingOverflow(by: Int(self.headDimension()));
        if dimensionOverflow {
            return nil;
        }
        let (totalBytes, byteOverflow) = perDimensionBytes
            .multipliedReportingOverflow(by: bfloat16ElementSizeBytes);
        return byteOverflow ? nil : totalBytes;
    }
}
