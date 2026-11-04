//! Whether Neural Engine MLP programs fit under the user's memory ceiling.
//!
//! Their weights live outside MLX, so MLX's own accounting never sees them, yet
//! the ceiling is the user's promise about the whole model. The programs are
//! therefore admitted only when they fit beside everything MLX can grow to.

/// Measured while serving a real model with 9 GiB of raw fp16 MLP weights (issue
/// #161): reclaimable memory fell 12 to 16 GiB against the same model without
/// programs, mostly wired by the Neural Engine. This takes the top of that range,
/// because admitting too much costs the Mac far more than refusing a little early.
const NEURAL_ENGINE_BYTES_PER_TEN_FP16_WEIGHT_BYTES: u64 = 18;
/// Room for one chunk's activations and merged results between programs.
const PREFILL_TRANSIENT_HEADROOM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Every memory claim the admission weighs, in exact bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NeuralEnginePrefillMemoryRequirements {
    pub resident_model_payload_bytes: u64,
    /// Decoder state at the model's largest context, which MLX can grow to.
    pub maximum_context_state_bytes: u64,
    pub fp16_mlp_weight_bytes: u64,
    pub active_memory_ceiling_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeuralEnginePrefillAdmission {
    Admitted {
        estimated_neural_engine_bytes: u64,
    },
    Refused {
        estimated_neural_engine_bytes: u64,
        required_bytes: u64,
        active_memory_ceiling_bytes: u64,
    },
}

impl NeuralEnginePrefillMemoryRequirements {
    /// Saturating arithmetic refuses absurd inputs instead of wrapping into an
    /// admission.
    #[must_use]
    pub const fn decide(self) -> NeuralEnginePrefillAdmission {
        let estimated_neural_engine_bytes = self
            .fp16_mlp_weight_bytes
            .saturating_mul(NEURAL_ENGINE_BYTES_PER_TEN_FP16_WEIGHT_BYTES)
            / 10;
        let required_bytes = self
            .resident_model_payload_bytes
            .saturating_add(self.maximum_context_state_bytes)
            .saturating_add(PREFILL_TRANSIENT_HEADROOM_BYTES)
            .saturating_add(estimated_neural_engine_bytes);
        if required_bytes <= self.active_memory_ceiling_bytes {
            NeuralEnginePrefillAdmission::Admitted {
                estimated_neural_engine_bytes,
            }
        } else {
            NeuralEnginePrefillAdmission::Refused {
                estimated_neural_engine_bytes,
                required_bytes,
                active_memory_ceiling_bytes: self.active_memory_ceiling_bytes,
            }
        }
    }
}

#[cfg(test)]
#[path = "neural_engine_admission_tests.rs"]
mod tests;
