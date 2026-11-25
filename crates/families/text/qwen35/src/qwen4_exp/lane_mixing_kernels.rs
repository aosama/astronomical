//! Fused Metal kernels for the element-wise steps of Flash-Next's lane
//! mixers.
//!
//! A Flash-Next reply token evaluates about 9,700 GPU primitives, and its
//! lane mixers are 98 of the about 191 a layer: each of the two lane reads
//! has three weight multiplies and 38 small steps around them, and each of
//! the two write-backs is 8 more. A small primitive costs 0.3 to 1.9
//! microseconds whatever its tensor holds
//! (`cargo measure-primitive-dispatch-cost`), so the small steps are a
//! fixed cost every token pays. These kernels do three of those chains in
//! one launch each:
//!
//! - the mix gate, `silu(x / lanes)`: 5 primitives;
//! - the write-back weights, `2 * sigmoid(x / lanes)`: 7 primitives;
//! - the write-back itself, `lanes + block * weight`: 8 primitives.
//!
//! Together they take a token from 9,678 primitives to about 8,135 and a
//! reply at a 34 GB limit from 21.8 to 22.0 tokens a second, over three
//! alternated pairs of runs (21.77, 21.80, 21.79 against 21.97, 22.08,
//! 21.99): 1%. Small primitives overlap on the GPU, so removing one saves
//! about 0.3 microseconds, not the price of one run alone.
//!
//! Each rounds to bfloat16 exactly where the composed steps do: bfloat16
//! has no native Metal arithmetic, so every operator rounds through
//! float32 as MLX's own kernels do, and the sigmoid lines are copied from
//! MLX's `Sigmoid::operator()` (metal/kernels/unary_ops.h), because any
//! other form of it rounds differently. The kernels are used only after a
//! model has compared them with the composed steps on this GPU and found
//! every bit equal (`enable_fused_lane_mixing`).

use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

/// Threads a threadgroup: one SIMD group's worth, or the row when shorter.
const THREADS_A_GROUP: i32 = 32;

/// What a gate kernel computes from `x / lanes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateKind {
    /// `silu`: the lane read's mix.
    Silu = 0,
    /// Twice the sigmoid: how much of a block's output a lane receives.
    TwiceSigmoid = 1,
}

/// The fused kernels of one model's lane mixers.
#[derive(Debug)]
pub struct LaneMixingKernels {
    gate: MlxMetalKernel,
    write_back: MlxMetalKernel,
}

impl LaneMixingKernels {
    /// Declares the kernels for a model with `lane_count` residual lanes,
    /// or `None` when one over the lane count is not a bfloat16 value the
    /// source can state exactly.
    pub fn declare(lane_count: i32) -> Result<Option<Self>, MlxRuntimeError> {
        if lane_count <= 0 || lane_count.count_ones() != 1 {
            return Ok(None);
        }
        let inverse_lane_count = format!("{:.9e}", 1.0 / lane_count as f32);
        // `scaled` is `x * (1 / lanes)` as the composed scalar multiply
        // leaves it; the next two lines are MLX's sigmoid, verbatim.
        let gate_source = format!(
            r#"
    uint element = thread_position_in_grid.y * uint(W) + thread_position_in_grid.x;
    const T scaled = gate_input[element] * T({inverse_lane_count}f);
    auto y = 1 / (1 + metal::precise::exp(metal::abs(scaled)));
    const T sigmoid_value = (scaled < 0) ? y : 1 - y;
    if (KIND == 0) {{
        gate_output[element] = scaled * sigmoid_value;
    }} else {{
        gate_output[element] = sigmoid_value * T(2.0f);
    }}
"#
        );
        let write_back_source = r#"
    uint feature = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint lane = feature / uint(H);
    uint hidden = feature - lane * uint(H);
    const T injected = block_output[row * uint(H) + hidden]
        * lane_weights[row * uint(L) + lane];
    uint element = row * uint(L) * uint(H) + feature;
    written_lanes[element] = lanes[element] + injected;
"#;
        Ok(Some(Self {
            gate: MlxMetalKernel::new(
                "astronomical_lane_mixing_gate",
                &["gate_input"],
                &["gate_output"],
                &gate_source,
            )?,
            write_back: MlxMetalKernel::new(
                "astronomical_lane_mixing_write_back",
                &["lanes", "block_output", "lane_weights"],
                &["written_lanes"],
                write_back_source,
            )?,
        }))
    }

    /// `silu(projected_down / lanes)`, `[1, rows, width]` bfloat16.
    pub fn mix_gate(
        &self,
        runtime: &MlxRuntime,
        projected_down: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.gate_of(runtime, projected_down, GateKind::Silu)
    }

    /// `2 * sigmoid(scores / lanes)`, `[1, rows, lanes]` bfloat16.
    pub fn write_back_weights(
        &self,
        runtime: &MlxRuntime,
        scores: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.gate_of(runtime, scores, GateKind::TwiceSigmoid)
    }

    fn gate_of(
        &self,
        runtime: &MlxRuntime,
        gate_input: &MlxArray,
        gate_kind: GateKind,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let input_shape = gate_input.shape();
        let [_, row_count, width] = rank_three(&input_shape)?;
        let outputs = runtime.apply_metal_kernel(
            &self.gate,
            &[gate_input],
            &[MlxMetalKernelOutput::new(
                input_shape.clone(),
                MlxDtype::BFloat16,
            )],
            [width, row_count, 1],
            [width.min(THREADS_A_GROUP), 1, 1],
            &[
                bfloat16_template_argument(),
                integer_template_argument("W", width),
                integer_template_argument("KIND", gate_kind as i32),
            ],
        )?;
        only_output(outputs)
    }

    /// `lanes + block_output * lane_weights`, each lane taking its own
    /// weight: `lanes` is `[1, rows, lanes * hidden]`, `block_output`
    /// `[1, rows, hidden]` and `lane_weights` `[1, rows, lanes]`.
    pub fn write_back(
        &self,
        runtime: &MlxRuntime,
        lanes: &MlxArray,
        block_output: &MlxArray,
        lane_weights: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let lanes_shape = lanes.shape();
        let [_, row_count, lanes_width] = rank_three(&lanes_shape)?;
        let [_, _, hidden_width] = rank_three(&block_output.shape())?;
        let [_, _, lane_count] = rank_three(&lane_weights.shape())?;
        if lane_count * hidden_width != lanes_width {
            return Err(lane_mixing_error(
                "the lanes must be as wide as the lane count times the block output",
            ));
        }
        let outputs = runtime.apply_metal_kernel(
            &self.write_back,
            &[lanes, block_output, lane_weights],
            &[MlxMetalKernelOutput::new(
                lanes_shape.clone(),
                MlxDtype::BFloat16,
            )],
            [lanes_width, row_count, 1],
            [lanes_width.min(THREADS_A_GROUP), 1, 1],
            &[
                bfloat16_template_argument(),
                integer_template_argument("H", hidden_width),
                integer_template_argument("L", lane_count),
            ],
        )?;
        only_output(outputs)
    }
}

/// Whether every operand is the one dtype the kernels are proven for.
#[must_use]
pub fn are_all_bfloat16(operands: &[&MlxArray]) -> bool {
    let mut are_bfloat16 = true;
    for operand in operands {
        are_bfloat16 &= operand.dtype() == MlxDtype::BFloat16;
    }
    are_bfloat16
}

fn rank_three(shape: &[i32]) -> Result<[i32; 3], MlxRuntimeError> {
    match shape {
        [1, row_count, width] if *row_count > 0 && *width > 0 => Ok([1, *row_count, *width]),
        _ => Err(lane_mixing_error(
            "a lane-mixing kernel takes [1, rows, width] operands",
        )),
    }
}

fn only_output(mut outputs: Vec<MlxArray>) -> Result<MlxArray, MlxRuntimeError> {
    outputs
        .pop()
        .ok_or_else(|| lane_mixing_error("a lane-mixing kernel returned no output"))
}

fn bfloat16_template_argument() -> MlxMetalKernelTemplateArgument {
    MlxMetalKernelTemplateArgument::Dtype {
        name: "T",
        dtype: MlxDtype::BFloat16,
    }
}

fn integer_template_argument(name: &'static str, value: i32) -> MlxMetalKernelTemplateArgument {
    MlxMetalKernelTemplateArgument::Integer {
        name,
        integer_template_argument: value,
    }
}

fn lane_mixing_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "apply a fused lane-mixing kernel",
        description: description.to_owned(),
    }
}
