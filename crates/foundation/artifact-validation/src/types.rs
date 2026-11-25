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
    /// 16-bit IEEE floating point, used by jangtq2 codebook row scales.
    Float16,
    /// 32-bit IEEE floating point.
    Float32,
    /// Unsigned 32-bit integers used by MLX packed quantized weights.
    UInt32,
    /// Signed 64-bit integers, used by hash tables a model ships with its
    /// weights (the n-gram head sizes and multipliers of Qwen3.8-Flash-Next).
    Int64,
}

impl TensorDtype {
    /// Whether a SafeTensors header's dtype spelling satisfies this profile.
    #[must_use]
    pub fn accepts_safetensors_dtype(self, safetensors_dtype: &str) -> bool {
        match self {
            Self::AffineQuantizationFloat | Self::ModelFloat => {
                matches!(safetensors_dtype, "F16" | "BF16" | "F32")
            }
            Self::BFloat16 => safetensors_dtype == "BF16",
            Self::Float16 => safetensors_dtype == "F16",
            Self::Float32 => safetensors_dtype == "F32",
            Self::UInt32 => safetensors_dtype == "U32",
            Self::Int64 => safetensors_dtype == "I64",
        }
    }
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
    ///
    /// Every entry must have a paired layout normalizer at weight binding —
    /// accepting a published shape without one would silently bind a
    /// layout-permuted tensor. Current pairs: `linear_attn.conv1d.weight`
    /// (gated delta convolution) and `vision_tower.patch_embed.proj.weight`.
    pub equivalent_published_shapes: Vec<Vec<usize>>,
}
