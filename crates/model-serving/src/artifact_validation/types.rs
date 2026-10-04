/// Structural identity for one required file in the model directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequiredFileProfile {
    /// File name relative to the model artifact directory.
    pub file_name: String,
    /// Exact file size in bytes.
    pub size_bytes: u64,
}

/// Supported safetensors dtype names for expected tensor metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDtype {
    /// Floating-point storage accepted by MLX affine scales and biases.
    AffineQuantizationFloat,
    /// Floating-point storage for model parameters retained without conversion.
    ModelFloat,
    /// Brain floating point with 16-bit storage.
    BFloat16,
    /// 32-bit IEEE floating point.
    Float32,
    /// Unsigned 32-bit integers used by MLX packed quantized weights.
    UInt32,
}

/// Expected metadata for one expected tensor in the model weight file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorProfile {
    /// Full safetensors tensor name.
    pub name: String,
    /// Expected tensor dtype.
    pub dtype: TensorDtype,
    /// Expected tensor shape in the engine's execution layout.
    pub shape: Vec<usize>,
    /// Additional shapes the tensor may publish in when publishers store a
    /// pure axis permutation of the execution layout. Validation accepts
    /// them, and the loader normalizes the stored tensor into `shape` before
    /// execution; a permutation reorders the same values, so numerics are
    /// unchanged. Empty for tensors with one published form.
    pub equivalent_published_shapes: Vec<Vec<usize>>,
}
