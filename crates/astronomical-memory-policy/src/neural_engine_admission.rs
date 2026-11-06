//! Whether Neural Engine programs fit under the user's memory ceiling.
//!
//! Their weights live outside MLX, so MLX's own accounting never sees them, yet the
//! ceiling is the user's promise about the whole model. Their cost is measured per
//! model when they first run, because it depends on the model, the weight form, the
//! Mac and macOS: 3.3 times the weights on a 0.8B model, 2.0 on a 2B model and 1.5 on
//! a 9B model (issue #161). No fixed ratio is right for more than one of them.
//!
//! Two questions follow, and neither assumes the model's largest context. The driver
//! holds this memory only during prefill, so what matters is the prompt really running.

/// Every memory claim weighed when deciding whether programs can ever run, in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NeuralEnginePrefillMemoryRequirements {
    pub resident_model_payload_bytes: u64,
    /// Decoder state after one chunk, the least any prompt using the programs holds.
    pub one_chunk_state_bytes: u64,
    /// The least the programs can cost before they run, or what they cost once measured.
    pub program_bytes: u64,
    pub active_memory_ceiling_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeuralEnginePrefillAdmission {
    Admitted,
    Refused { required_bytes: u64 },
}

impl NeuralEnginePrefillMemoryRequirements {
    /// Saturating arithmetic refuses absurd inputs instead of wrapping into an
    /// admission.
    #[must_use]
    pub const fn decide(self) -> NeuralEnginePrefillAdmission {
        let required_bytes = self
            .resident_model_payload_bytes
            .saturating_add(self.one_chunk_state_bytes)
            .saturating_add(self.program_bytes);
        if required_bytes <= self.active_memory_ceiling_bytes {
            NeuralEnginePrefillAdmission::Admitted
        } else {
            NeuralEnginePrefillAdmission::Refused { required_bytes }
        }
    }
}

/// Whether this chunk may use the programs, from the peak MLX is projected to reach
/// during it. A chunk that does not fit runs on the GPU alone, which costs speed and
/// never the request.
#[must_use]
pub const fn chunk_fits_beside_programs(
    projected_chunk_peak_bytes: u64,
    program_bytes: u64,
    active_memory_ceiling_bytes: u64,
) -> bool {
    projected_chunk_peak_bytes.saturating_add(program_bytes) <= active_memory_ceiling_bytes
}

#[cfg(test)]
#[path = "neural_engine_admission_tests.rs"]
mod tests;
