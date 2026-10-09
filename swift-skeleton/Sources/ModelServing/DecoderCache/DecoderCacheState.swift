import Foundation;

/// One architecture-neutral in-memory decoder-cache state family, port of
/// the Rust `DecoderCacheState`. Hybrid models mix both families across
/// layers; the state-bridge and the prompt-cache contracts dispatch on the
/// family, never on a concrete model type.
public enum DecoderCacheState {

    /// Append-only attention keys and values.
    case appendOnlyAttention(FullAttentionKeyValueState);

    /// A hybrid layer combining rolling convolution and recurrent state.
    case composite(convolution: ConvolutionState, recurrent: GatedDeltaRecurrentState);
}
