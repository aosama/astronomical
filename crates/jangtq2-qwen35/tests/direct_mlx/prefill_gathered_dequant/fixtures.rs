//! Deterministic fixtures for the tier-0 prefill gathered-dequant journey:
//! reproducible packed words, float16-exact scales and activations, the
//! chunk's routed assignments in expert-sorted compressed sparse row order,
//! and the shared measurement helpers.

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

pub(super) fn prototype_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(20 * 1024 * 1024 * 1024, 20 * 1024 * 1024 * 1024)
            .expect("the prefill gathered dequant memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Deterministic splitmix64 packed words, matching the decoder throughput
/// fixtures so the weight bit patterns are reproducible run over run.
pub(super) fn deterministic_packed_words(seed: u64, word_count: usize) -> Vec<u32> {
    let mut state = seed;
    (0..word_count)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (mixed ^ (mixed >> 31)) as u32
        })
        .collect()
}

/// Row scales exactly representable in float16 so fixtures never depend on
/// rounding of a wider dtype.
pub(super) fn deterministic_row_scales(seed: u64, row_count: usize) -> Vec<f32> {
    (0..row_count)
        .map(|row_index| {
            let step = seed.wrapping_add(row_index as u64) % 16;
            ((step + 4) as f32) / 8.0
        })
        .collect()
}

/// Token activation fixture: float16-exact halves keep the fixture
/// reproducible and the exact-parity denominator well-conditioned.
pub(super) fn deterministic_token_activations(seed: u64, value_count: usize) -> Vec<f32> {
    deterministic_row_scales(seed, value_count)
}

pub(super) fn median_duration_millis(durations_ms: &[f64]) -> f64 {
    let mut sorted_durations = durations_ms.to_vec();
    sorted_durations.sort_by(|left, right| left.total_cmp(right));
    sorted_durations[sorted_durations.len() / 2]
}

/// One token's routed experts: `top_k` distinct experts drawn from a
/// deterministic splitmix64 stream by rejection, so every expert of the
/// layer is covered across the chunk and the fixture is reproducible.
pub(super) fn deterministic_token_experts(
    seed: u64,
    expert_count: usize,
    top_k: usize,
) -> Vec<usize> {
    let mut state = seed;
    let mut selected_experts = Vec::with_capacity(top_k);
    while selected_experts.len() < top_k {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let candidate_expert = (mixed ^ (mixed >> 31)) as usize % expert_count;
        if !selected_experts.contains(&candidate_expert) {
            selected_experts.push(candidate_expert);
        }
    }
    selected_experts
}

/// Expert-sorted assignments with compressed sparse row per-expert ranges: the
/// layout the prototype kernel iterates and the layout `gather_mm` receives
/// when the serving path sorts assignments by expert.
pub(super) struct SortedAssignments {
    pub(super) assignment_expert_ids: Vec<u32>,
    pub(super) assignment_token_indices: Vec<usize>,
    pub(super) expert_assignment_starts: Vec<u32>,
}

pub(super) fn expert_sorted_assignments(
    token_count: usize,
    expert_count: usize,
    top_k: usize,
) -> SortedAssignments {
    let mut token_expert_pairs = Vec::with_capacity(token_count * top_k);
    for token_index in 0..token_count {
        for expert_index in deterministic_token_experts(
            0x5EED_0000_0000_0001u64.wrapping_add((token_index as u64) * 0x9E37),
            expert_count,
            top_k,
        ) {
            token_expert_pairs.push((expert_index, token_index));
        }
    }
    token_expert_pairs.sort_unstable();
    let assignment_expert_ids = token_expert_pairs
        .iter()
        .map(|&(expert_index, _)| expert_index as u32)
        .collect();
    let assignment_token_indices = token_expert_pairs
        .iter()
        .map(|&(_, token_index)| token_index)
        .collect();
    let mut expert_assignment_starts = vec![0u32; expert_count + 1];
    for (position, &(expert_index, _)) in token_expert_pairs.iter().enumerate() {
        expert_assignment_starts[expert_index + 1] = (position + 1) as u32;
    }
    // Experts with no assignment keep the previous boundary (a running
    // maximum repairs draws that skipped an expert entirely).
    for boundary_index in 1..expert_assignment_starts.len() {
        expert_assignment_starts[boundary_index] = expert_assignment_starts[boundary_index]
            .max(expert_assignment_starts[boundary_index - 1]);
    }
    SortedAssignments {
        assignment_expert_ids,
        assignment_token_indices,
        expert_assignment_starts,
    }
}
