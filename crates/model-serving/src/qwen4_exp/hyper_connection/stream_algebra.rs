//! Stream algebra for `qwen4_exp` hyper-connections.
//!
//! Pure float arithmetic with no tensor library and no I/O: streams in,
//! mixed streams out. This owner exists so the mixing rule is pinned,
//! auditable, and testable before any graph, kernel, or cache work builds on
//! it — a wrong mixing rule produces fluent wrong text that no structural
//! check can catch.
//!
//! The algebra follows the gated-residual scheme (arXiv 2409.19606) as the
//! public reference implements it for this architecture, with per-branch
//! normalization: a grouped RMSNorm over each stream, a low-rank SiLU-scaled
//! gate reduced by sigmoid, a stream-mean block input, and a learned
//! per-stream injection weight of `2 · sigmoid(...)` on the way out. The
//! average-pooling variant stays expressible because published variants may
//! select it; the weight inventory decides, never an assumption.
//!
//! Arithmetic runs in `f32` here. Production instantiates it with the model's
//! activation dtype; the contract tests pin operation order and values at
//! `f32` precision against independently derived expectations.

/// Geometry and normalization constant for one hyper-connection site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamMixingPlan {
    pub stream_count: u32,
    pub stream_width: u32,
    pub low_rank: u32,
    pub rms_norm_epsilon: f32,
}

/// Why the stream algebra cannot run with the provided plan or weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamAlgebraError {
    ZeroStreamCount,
    ZeroStreamWidth,
    NormWeightLengthMismatch { expected: usize, provided: usize },
    DownWeightLengthMismatch { expected: usize, provided: usize },
    UpWeightLengthMismatch { expected: usize, provided: usize },
    InjectWeightLengthMismatch { expected: usize, provided: usize },
    HyperInputLengthMismatch { expected: usize, provided: usize },
    BlockOutputLengthMismatch { expected: usize, provided: usize },
}

impl std::fmt::Display for StreamAlgebraError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroStreamCount => write!(formatter, "stream count must be positive"),
            Self::ZeroStreamWidth => write!(formatter, "stream width must be positive"),
            Self::NormWeightLengthMismatch { expected, provided } => write!(
                formatter,
                "norm weights must hold {expected} elements, got {provided}"
            ),
            Self::DownWeightLengthMismatch { expected, provided } => write!(
                formatter,
                "down weights must hold {expected} elements, got {provided}"
            ),
            Self::UpWeightLengthMismatch { expected, provided } => write!(
                formatter,
                "up weights must hold {expected} elements, got {provided}"
            ),
            Self::InjectWeightLengthMismatch { expected, provided } => write!(
                formatter,
                "injection weights must hold {expected} elements, got {provided}"
            ),
            Self::HyperInputLengthMismatch { expected, provided } => write!(
                formatter,
                "hyper input must hold {expected} elements, got {provided}"
            ),
            Self::BlockOutputLengthMismatch { expected, provided } => write!(
                formatter,
                "block output must hold {expected} elements, got {provided}"
            ),
        }
    }
}

impl std::error::Error for StreamAlgebraError {}

/// Row-major weights for the gated-residual variant, laid out the way the
/// checkpoint names them: norm over all streams, down `[low_rank, hyper]`,
/// up `[hyper, low_rank]`, injection `[stream_count, hyper]`.
#[derive(Clone, Copy, Debug)]
pub struct GatedResidualWeights<'a> {
    pub norm: &'a [f32],
    pub down: &'a [f32],
    pub up: &'a [f32],
    pub inject: &'a [f32],
}

/// The two results every gated mix must hand to its paired combine.
#[derive(Clone, Debug)]
pub struct GatedMixOutput {
    /// Block input: one width-wide vector averaged over the streams.
    pub mixed_input: Vec<f32>,
    /// The normalized streams, so combine reuses the same normalization.
    pub normalized: Vec<f32>,
}

fn hyper_width(plan: &StreamMixingPlan) -> usize {
    plan.stream_count as usize * plan.stream_width as usize
}

fn validate_plan(plan: &StreamMixingPlan) -> Result<(), StreamAlgebraError> {
    if plan.stream_count == 0 {
        return Err(StreamAlgebraError::ZeroStreamCount);
    }
    if plan.stream_width == 0 {
        return Err(StreamAlgebraError::ZeroStreamWidth);
    }
    Ok(())
}

fn validate_input_lengths(
    plan: &StreamMixingPlan,
    hyper_input: &[f32],
) -> Result<(), StreamAlgebraError> {
    let expected = hyper_width(plan);
    if hyper_input.len() != expected {
        return Err(StreamAlgebraError::HyperInputLengthMismatch {
            expected,
            provided: hyper_input.len(),
        });
    }
    Ok(())
}

fn validate_block_output_length(
    plan: &StreamMixingPlan,
    block_output: &[f32],
) -> Result<(), StreamAlgebraError> {
    let expected = plan.stream_width as usize;
    if block_output.len() != expected {
        return Err(StreamAlgebraError::BlockOutputLengthMismatch {
            expected,
            provided: block_output.len(),
        });
    }
    Ok(())
}

fn dot_row(row: &[f32], input: &[f32]) -> f32 {
    row.iter()
        .zip(input.iter())
        .map(|(weight, value)| weight * value)
        .sum()
}

/// Grouped RMSNorm over each stream, computed in `f32`, with the per-element
/// affine of `1 + weight` the scheme uses.
///
/// The variance is taken over each `stream_width` group, which is why the
/// published norm weight spans all `stream_count * stream_width` elements.
///
/// # Errors
/// When the plan is degenerate or a slice length disagrees with the plan.
pub fn grouped_rms_norm(
    plan: &StreamMixingPlan,
    norm_weights: &[f32],
    hyper_input: &[f32],
) -> Result<Vec<f32>, StreamAlgebraError> {
    validate_plan(plan)?;
    let expected = hyper_width(plan);
    if norm_weights.len() != expected {
        return Err(StreamAlgebraError::NormWeightLengthMismatch {
            expected,
            provided: norm_weights.len(),
        });
    }
    validate_input_lengths(plan, hyper_input)?;
    let width = plan.stream_width as usize;
    let mut normalized = Vec::with_capacity(hyper_input.len());
    for (stream_index, stream) in hyper_input.chunks_exact(width).enumerate() {
        let variance = stream.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let scale = 1.0 / (variance + plan.rms_norm_epsilon).sqrt();
        let weight_base = stream_index * width;
        for (offset, value) in stream.iter().enumerate() {
            let weight = norm_weights[weight_base + offset];
            normalized.push(value * scale * (1.0 + weight));
        }
    }
    Ok(normalized)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Gated mix: normalize, project down, SiLU scaled by one over the stream
/// count, project up, sigmoid, multiply into the normalized streams, and
/// average over streams.
///
/// # Errors
/// When the plan is degenerate or a slice length disagrees with the plan.
pub fn gated_mix(
    plan: &StreamMixingPlan,
    weights: &GatedResidualWeights<'_>,
    hyper_input: &[f32],
) -> Result<GatedMixOutput, StreamAlgebraError> {
    validate_plan(plan)?;
    let hyper = hyper_width(plan);
    let low_rank = plan.low_rank as usize;
    if weights.norm.len() != hyper {
        return Err(StreamAlgebraError::NormWeightLengthMismatch {
            expected: hyper,
            provided: weights.norm.len(),
        });
    }
    if weights.down.len() != low_rank * hyper {
        return Err(StreamAlgebraError::DownWeightLengthMismatch {
            expected: low_rank * hyper,
            provided: weights.down.len(),
        });
    }
    if weights.up.len() != hyper * low_rank {
        return Err(StreamAlgebraError::UpWeightLengthMismatch {
            expected: hyper * low_rank,
            provided: weights.up.len(),
        });
    }
    validate_input_lengths(plan, hyper_input)?;
    let normalized = grouped_rms_norm(plan, weights.norm, hyper_input)?;
    let stream_count = plan.stream_count as f32;
    let mut hidden = Vec::with_capacity(low_rank);
    for rank in 0..low_rank {
        let row = &weights.down[rank * hyper..(rank + 1) * hyper];
        hidden.push(sigmoid_for_silu(dot_row(row, &normalized) / stream_count));
    }
    let mut gate = Vec::with_capacity(hyper);
    for stream in 0..hyper {
        let mut product = 0.0_f32;
        for rank in 0..low_rank {
            product += weights.up[stream * low_rank + rank] * hidden[rank];
        }
        gate.push(sigmoid(product));
    }
    let width = plan.stream_width as usize;
    let mut mixed_input = vec![0.0_f32; width];
    for stream in 0..plan.stream_count as usize {
        for offset in 0..width {
            let index = stream * width + offset;
            mixed_input[offset] += gate[index] * normalized[index];
        }
    }
    for value in &mut mixed_input {
        *value /= stream_count;
    }
    Ok(GatedMixOutput {
        mixed_input,
        normalized,
    })
}

/// SiLU, written out so the divide-then-activate order stays visible.
fn sigmoid_for_silu(value: f32) -> f32 {
    value * sigmoid(value)
}

/// Gated combine: a per-stream injection weight of
/// `2 · sigmoid(project(normalized) / stream_count)` scales the block output
/// before it is added to the raw residual. The normalized streams participate
/// only in the injection weight, exactly as the reference orders it.
///
/// # Errors
/// When the plan is degenerate or a slice length disagrees with the plan.
pub fn gated_combine(
    plan: &StreamMixingPlan,
    weights: &GatedResidualWeights<'_>,
    block_output: &[f32],
    hyper_input: &[f32],
    normalized: &[f32],
) -> Result<Vec<f32>, StreamAlgebraError> {
    validate_plan(plan)?;
    let hyper = hyper_width(plan);
    if weights.inject.len() != plan.stream_count as usize * hyper {
        return Err(StreamAlgebraError::InjectWeightLengthMismatch {
            expected: plan.stream_count as usize * hyper,
            provided: weights.inject.len(),
        });
    }
    validate_input_lengths(plan, hyper_input)?;
    if normalized.len() != hyper {
        return Err(StreamAlgebraError::HyperInputLengthMismatch {
            expected: hyper,
            provided: normalized.len(),
        });
    }
    validate_block_output_length(plan, block_output)?;
    let stream_count = plan.stream_count as f32;
    let mut output = vec![0.0_f32; hyper];
    for stream in 0..plan.stream_count as usize {
        let row = &weights.inject[stream * hyper..(stream + 1) * hyper];
        let injection = 2.0 * sigmoid(dot_row(row, normalized) / stream_count);
        for (offset, value) in block_output.iter().enumerate() {
            let index = stream * plan.stream_width as usize + offset;
            output[index] = hyper_input[index] + value * injection;
        }
    }
    Ok(output)
}

/// Average-pooling mix: the plain mean over streams.
///
/// # Errors
/// When the plan is degenerate or the input length disagrees with the plan.
pub fn average_mix(
    plan: &StreamMixingPlan,
    hyper_input: &[f32],
) -> Result<Vec<f32>, StreamAlgebraError> {
    validate_plan(plan)?;
    validate_input_lengths(plan, hyper_input)?;
    let width = plan.stream_width as usize;
    let mut mixed = vec![0.0_f32; width];
    for stream in 0..plan.stream_count as usize {
        for offset in 0..width {
            mixed[offset] += hyper_input[stream * width + offset];
        }
    }
    let stream_count = plan.stream_count as f32;
    for value in &mut mixed {
        *value /= stream_count;
    }
    Ok(mixed)
}

/// Average-pooling combine: the block output added to every stream.
///
/// # Errors
/// When the plan is degenerate or a slice length disagrees with the plan.
pub fn average_combine(
    plan: &StreamMixingPlan,
    block_output: &[f32],
    hyper_input: &[f32],
) -> Result<Vec<f32>, StreamAlgebraError> {
    validate_plan(plan)?;
    validate_input_lengths(plan, hyper_input)?;
    validate_block_output_length(plan, block_output)?;
    let width = plan.stream_width as usize;
    let mut output = vec![0.0_f32; hyper_input.len()];
    for stream in 0..plan.stream_count as usize {
        for offset in 0..width {
            let index = stream * width + offset;
            output[index] = hyper_input[index] + block_output[offset];
        }
    }
    Ok(output)
}
