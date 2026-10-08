import Foundation

/// Scalar type required by one persisted decoder-cache tensor, port of
/// the Rust `DecoderCacheTensorDtype`. The Rust `mlx_dtype` mapping has
/// no port: Swift execution resolves scalars through MLX-Swift's own
/// dtype vocabulary at the engine seam.
public enum DecoderCacheTensorDtype: Equatable, Hashable, Sendable {

    case float16

    case bfloat16

    case float32

    case int32

    /// Returns the scalar payload width used by this decoder-cache tensor.
    public var scalarByteCount: Int {
        switch self {
        case .float16:
            return 2
        case .bfloat16:
            return 2
        case .float32:
            return 4
        case .int32:
            return 4
        }
    }

    /// Returns the scalar label written by MLX safetensors serialization.
    public var safetensorsDtypeName: String {
        switch self {
        case .float16:
            return "F16"
        case .bfloat16:
            return "BF16"
        case .float32:
            return "F32"
        case .int32:
            return "I32"
        }
    }
}
