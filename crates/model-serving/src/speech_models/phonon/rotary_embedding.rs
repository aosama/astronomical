//! Rotary position embedding that rotates interleaved pairs (`x0, x1`),
//! (`x2, x3`), ... as Pocket TTS's `rope_i` does. MLX's fused rope rotates
//! halves instead, which would scramble a checkpoint trained on pairs, so the
//! rotation is written out here.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::speech_model_error::SpeechModelError;

/// `[heads, length, head width]`, rotated for absolute positions starting at
/// `first_position`.
pub(crate) fn rotate_interleaved_pairs(
    runtime: &MlxRuntime,
    heads: &MlxArray,
    first_position: usize,
    max_period: f32,
) -> Result<MlxArray, SpeechModelError> {
    let [head_count, length, head_width] = heads.shape()[..] else {
        return Err(SpeechModelError::Input(
            "rotary embedding expects [heads, length, width]".to_owned(),
        ));
    };
    let pair_count = head_width / 2;
    let rotary_table = rotary_table(
        first_position,
        length as usize,
        pair_count as usize,
        max_period,
    );
    let cosine = runtime.array_from_f32(&rotary_table.cosines, &[1, length, pair_count])?;
    let sine = runtime.array_from_f32(&rotary_table.sines, &[1, length, pair_count])?;
    let pairs = runtime.reshape(heads, &[head_count, length, pair_count, 2])?;
    let pair_shape = [head_count, length, pair_count];
    let pair_real = pair_member(runtime, &pairs, pair_shape, 0)?;
    let pair_imaginary = pair_member(runtime, &pairs, pair_shape, 1)?;
    let rotated_real = runtime.subtract(
        &runtime.multiply(&pair_real, &cosine)?,
        &runtime.multiply(&pair_imaginary, &sine)?,
    )?;
    let rotated_imaginary = runtime.add(
        &runtime.multiply(&pair_real, &sine)?,
        &runtime.multiply(&pair_imaginary, &cosine)?,
    )?;
    let rotated_pairs = runtime.stack_axis(&[&rotated_real, &rotated_imaginary], 3)?;
    Ok(runtime.reshape(&rotated_pairs, &[head_count, length, head_width])?)
}

/// The first or second member of every pair, `[heads, length, pairs]`.
fn pair_member(
    runtime: &MlxRuntime,
    pairs: &MlxArray,
    [head_count, length, pair_count]: [i32; 3],
    member_index: i32,
) -> Result<MlxArray, SpeechModelError> {
    let member = runtime.slice(
        pairs,
        &[0, 0, 0, member_index],
        &[head_count, length, pair_count, member_index + 1],
        &[1, 1, 1, 1],
    )?;
    Ok(runtime.squeeze_axis(&member, 3)?)
}

/// Cosines and sines of `position / max_period^(pair / pairs)`, row-major
/// `[length, pairs]`.
pub(crate) struct RotaryTable {
    pub(crate) cosines: Vec<f32>,
    pub(crate) sines: Vec<f32>,
}

pub(crate) fn rotary_table(
    first_position: usize,
    length: usize,
    pair_count: usize,
    max_period: f32,
) -> RotaryTable {
    let inverse_frequencies: Vec<f32> = (0..pair_count)
        .map(|pair_index| 1.0 / max_period.powf(pair_index as f32 / pair_count as f32))
        .collect();
    let mut rotary_table = RotaryTable {
        cosines: Vec::with_capacity(length * pair_count),
        sines: Vec::with_capacity(length * pair_count),
    };
    for position in first_position..first_position + length {
        for inverse_frequency in &inverse_frequencies {
            let angle = position as f32 * inverse_frequency;
            rotary_table.cosines.push(angle.cos());
            rotary_table.sines.push(angle.sin());
        }
    }
    rotary_table
}

#[cfg(test)]
#[path = "rotary_embedding_tests.rs"]
mod tests;
