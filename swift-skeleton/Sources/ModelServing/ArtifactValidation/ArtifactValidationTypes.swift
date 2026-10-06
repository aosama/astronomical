import Foundation;

/// Core artifact-validation wire types, port of
/// crates/model-serving/src/artifact_validation/types.rs. The remaining
/// artifact-validation surface lands with its own owner files.

/// Structural identity for one required file in the model directory.
public struct RequiredFileProfile: Equatable {
    /// File name relative to the model artifact directory.
    public let fileName: String;
    /// Exact file size in bytes.
    public let sizeBytes: UInt64;

    public init(fileName: String, sizeBytes: UInt64) {
        self.fileName = fileName;
        self.sizeBytes = sizeBytes;
    }
}

/// Supported safetensors dtype names for expected tensor metadata.
public enum TensorDtype: Equatable, Sendable {
    /// Floating-point storage accepted by MLX affine scales and biases.
    case affineQuantizationFloat;
    /// Floating-point storage for model parameters retained without conversion.
    case modelFloat;
    /// Brain floating point with 16-bit storage.
    case bfloat16;
    /// 32-bit IEEE floating point.
    case float32;
    /// Unsigned 32-bit integers used by MLX packed quantized weights.
    case uint32;
}

/// Expected metadata for one expected tensor in the model weight file.
public struct TensorProfile: Equatable {
    /// Full safetensors tensor name.
    public let name: String;
    /// Expected tensor dtype.
    public let dtype: TensorDtype;
    /// Expected tensor shape in the engine's execution layout.
    public let shape: Array<Int>;
    /// Additional shapes the tensor may publish in when publishers store a
    /// pure axis permutation of the execution layout. Validation accepts
    /// them, and the loader normalizes the stored tensor into `shape` before
    /// execution; a permutation reorders the same values, so numerics are
    /// unchanged. Empty for tensors with one published form.
    public let equivalentPublishedShapes: Array<Array<Int>>;

    public init(
        name: String, dtype: TensorDtype, shape: Array<Int>,
        equivalentPublishedShapes: Array<Array<Int>> = Array()) {
        self.name = name;
        self.dtype = dtype;
        self.shape = shape;
        self.equivalentPublishedShapes = equivalentPublishedShapes;
    }
}
