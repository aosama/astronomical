import Foundation;

/// One Astronomical custom Metal kernel whose dispatch can fall back to a
/// public MLX API, port of the Rust `CustomMetalKernelFamily`.
///
/// Kernel sources are fixed constants, so a verdict depends only on the GPU
/// and operating system, never on the loaded model.
public enum CustomMetalKernelFamily: Equatable, Hashable, Sendable {

    /// Sorted mixture-of-experts weighted reduction, shared by Qwen families.
    case sortedExpertWeightedSum;

    /// Fused single-token quantized expert decode for K2 Horizon MoVA.
    case fusedQuantizedExpertDecode;

    /// Fused Qwen3.5 gated-delta sequence recurrence.
    case gatedDeltaSequence;

    /// Boundary-checkpoint variant of the fused gated-delta recurrence.
    case gatedDeltaBoundaryCheckpoint;

    /// Fused gated-delta decode prework: convolution window, conv1d, SiLU,
    /// q/k/v split, RMS norms, and scalar scales in one launch.
    case gdnDecodePrework;
}
