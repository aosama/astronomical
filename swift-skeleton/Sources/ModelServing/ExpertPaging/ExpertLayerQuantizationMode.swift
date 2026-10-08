import Foundation

/// The storage encoding of one expert projection or one whole MoE layer.
/// Port of the Rust `QuantizationMode` domain enum: affine layers carry a
/// packed U32 weight with float scales and biases, native layers store a
/// plain bfloat16 weight. Deliberately distinct from MLX's own
/// `QuantizationMode`, which has no unquantized member.
public enum ExpertLayerQuantizationMode: Equatable, Sendable {
    /// Packed affine storage: U32 weight plus scales and biases.
    case affine
    /// Plain bfloat16 weight storage.
    case nativeBfloat16
}
