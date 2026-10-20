//! A small runtime for the contracts and a deterministic weight generator.

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

/// Two gigabytes is far more than any contract tensor needs and far below
/// any Apple Silicon laptop's ceiling.
const CONTRACT_ACTIVE_MEMORY_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;

pub fn contract_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            CONTRACT_ACTIVE_MEMORY_LIMIT_BYTES,
            CONTRACT_ACTIVE_MEMORY_LIMIT_BYTES,
        )
        .expect("the contract memory limits are valid"),
    )
    .expect("the MLX runtime should initialize on an Apple Silicon host")
}

/// Laplace-like weights with a few outliers, the regime where the range
/// search matters, from a tiny deterministic generator so the contracts
/// never depend on a random crate.
pub fn outlier_heavy_weights(rows: usize, columns: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut weights = Vec::with_capacity(rows * columns);
    for index in 0..rows * columns {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let uniform = (state >> 11) as f64 / (1_u64 << 53) as f64;
        let laplace = if uniform < 0.5 {
            (2.0 * uniform).ln()
        } else {
            -(2.0 * (1.0 - uniform)).ln()
        };
        let mut value = (0.02 * laplace) as f32;
        if index % 97 == 5 {
            value = 0.9;
        }
        weights.push(value);
    }
    weights
}
