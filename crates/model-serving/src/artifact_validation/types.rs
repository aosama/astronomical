/// Structural identity for one required file in the model directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequiredFileProfile {
    /// File name relative to the model artifact directory.
    pub file_name: String,
    /// Exact file size in bytes.
    pub size_bytes: u64,
}

/// Supported safetensors dtype names for certified tensor metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDtype {
    /// Brain floating point with 16-bit storage.
    BFloat16,
    /// Either BF16 storage or F32 storage; runtime computation promotes to F32.
    BFloat16OrFloat32,
    /// 32-bit IEEE floating point.
    Float32,
    /// Unsigned 32-bit integers used by MLX packed quantized weights.
    UInt32,
}

/// Certified metadata for one expected tensor in the model weight file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorProfile {
    /// Full safetensors tensor name.
    pub name: String,
    /// Expected tensor dtype.
    pub dtype: TensorDtype,
    /// Expected tensor shape.
    pub shape: Vec<usize>,
}
