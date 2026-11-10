//! The summary the requantizer returns, recording what was written rather
//! than what was planned.

/// What the requantizer produced, for the pipeline's summary and metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequantizeOutcome {
    pub shard_count: usize,
    pub total_output_bytes: u64,
    pub quantized_tensor_count: usize,
    pub passed_through_tensor_count: usize,
    /// Per-module `(bits, group_size)` as actually written, which can differ
    /// from the plan: a module whose last dimension does not divide its
    /// planned group size falls back to a smaller one or is left at bfloat16.
    /// The artifact metadata must describe what was written, not the plan.
    pub written_quantization_by_module: std::collections::BTreeMap<String, (u32, u32)>,
    pub peak_mlx_memory_bytes: u64,
    /// Tower tensors written to the `optiq_vision.safetensors` sidecar; zero
    /// when the source declares no tower.
    pub vision_sidecar_tensor_count: usize,
    /// The multi-token-prediction head, when the source had one and it was
    /// written to `optiq/mtp.safetensors`.
    pub mtp_sidecar: Option<super::mtp_sidecar::MtpSidecarSummary>,
    /// Logical weights in the model shards, a packed weight counted at its
    /// unpacked size and its scales and biases counted too: mlx-lm's
    /// `total_parameters` figure for the shard index.
    pub total_parameter_count: u64,
}
