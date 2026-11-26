//! The operands and the composed reference of the vision tower's rotate-half
//! RoPE, shared by the bit-exactness contract and the timing probe so both
//! compare the compiled graph against the very same inputs and op sequence.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;

// Qwen3.5 vision tower geometry: hidden 1152 over 16 heads is head dimension 72.
pub(super) const HEAD_COUNT: i32 = 16;
pub(super) const HEAD_DIMENSION: i32 = 72;

pub(super) struct VisionRopeOperands {
    pub(super) attention_states: MlxArray,
    pub(super) rotary_cosines: MlxArray,
    pub(super) rotary_sines: MlxArray,
    pub(super) first_half: MlxArray,
    pub(super) second_half: MlxArray,
}

impl VisionRopeOperands {
    /// Patterned operands for `patch_count` patches; the head halves are the
    /// two slices the rotate-half concat consumes.
    pub(super) fn patterned(runtime: &MlxRuntime, patch_count: i32) -> Self {
        let attention_states = bf16_array(
            runtime,
            &patterned_values(
                (patch_count * HEAD_COUNT * HEAD_DIMENSION) as usize,
                0.01,
                -0.5,
            ),
            &[patch_count, HEAD_COUNT, HEAD_DIMENSION],
        );
        let rotary_cosines = f32_array(
            runtime,
            &patterned_values((patch_count * HEAD_DIMENSION) as usize, 0.02, -1.0),
            &[patch_count, 1, HEAD_DIMENSION],
        );
        let rotary_sines = f32_array(
            runtime,
            &patterned_values((patch_count * HEAD_DIMENSION) as usize, 0.03, 0.25),
            &[patch_count, 1, HEAD_DIMENSION],
        );
        let half_head_dimension = HEAD_DIMENSION / 2;
        let first_half = runtime
            .slice(
                &attention_states,
                &[0, 0, 0],
                &[patch_count, HEAD_COUNT, half_head_dimension],
                &[1, 1, 1],
            )
            .expect("the first head half should slice");
        let second_half = runtime
            .slice(
                &attention_states,
                &[0, 0, half_head_dimension],
                &[patch_count, HEAD_COUNT, HEAD_DIMENSION],
                &[1, 1, 1],
            )
            .expect("the second head half should slice");
        Self {
            attention_states,
            rotary_cosines,
            rotary_sines,
            first_half,
            second_half,
        }
    }

    pub(super) fn composed_rope(&self, runtime: &MlxRuntime) -> MlxArray {
        composed_vision_rope(
            runtime,
            &self.attention_states,
            &self.rotary_cosines,
            &self.rotary_sines,
            &self.first_half,
            &self.second_half,
        )
    }
}

// Mirrors the production composed RoPE op sequence (the path the compiled
// graph replaces) so callers measure and compare against the real reference.
fn composed_vision_rope(
    runtime: &MlxRuntime,
    attention_states: &MlxArray,
    rotary_cosines: &MlxArray,
    rotary_sines: &MlxArray,
    first_half: &MlxArray,
    second_half: &MlxArray,
) -> MlxArray {
    let negative_second_half = runtime
        .negative(second_half)
        .expect("the second half should negate");
    let rotated_states = runtime
        .concatenate_axis(&[&negative_second_half, first_half], 2)
        .expect("the rotated halves should concatenate");
    let cosine_component = runtime
        .multiply(attention_states, rotary_cosines)
        .expect("the cosine component should build");
    let sine_component = runtime
        .multiply(&rotated_states, rotary_sines)
        .expect("the sine component should build");
    let summed_components = runtime
        .add(&cosine_component, &sine_component)
        .expect("the components should add");
    runtime
        .astype(&summed_components, attention_states.dtype())
        .expect("the composed rope should restore the input dtype")
}

fn bf16_array(runtime: &MlxRuntime, values: &[f32], shape: &[i32]) -> MlxArray {
    runtime
        .astype(
            &runtime
                .array_from_f32(values, shape)
                .expect("the bfloat16 source should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the array should cast to bfloat16")
}

fn f32_array(runtime: &MlxRuntime, values: &[f32], shape: &[i32]) -> MlxArray {
    runtime
        .array_from_f32(values, shape)
        .expect("the float32 array should be valid")
}

fn patterned_values(element_count: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..element_count)
        .map(|value_index| ((value_index % 23) as f32 - 11.0) * scale + offset)
        .collect()
}
