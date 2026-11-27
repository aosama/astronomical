//! Whether a training's gradient is the loss's own slope, array by array,
//! on the real model and the real backend.
//!
//! The parity check proves a trainable copy scores as serving does; it says
//! nothing about the backward pass. A backend whose backward kernel is wrong
//! (a missing term, a wrong transpose, a lost scale) still passes parity
//! and then trains slowly or not at all, which looks exactly like a model
//! that is hard to train. This check tells the two apart: for each trainable
//! array it steps the parameters a little along a pseudo-random direction
//! confined to that array, both ways, and compares the loss's measured
//! change with the change the gradient predicts. Written for the CUDA
//! backend (issue #2699), it holds on any backend and family.

use mlx_bridge::MlxArray;

use super::gpu_math::{GpuMath, GpuResult, GpuTrainingError};
use super::gpu_training::{GpuTrainingSet, batch_objective};
use super::question_batch::QuestionBatch;
use super::row_objective::{RowTargets, mean_cross_entropy};

/// How far each array is stepped along its direction, both ways, as a share
/// of the direction's size: small enough that the loss is close to linear,
/// large enough that float32 rounding of a loss near 0.7 (about 1e-7) stays
/// far below the change.
const DIRECTION_STEP: f32 = 0.1;
/// The root-mean-square size of a direction for an array that starts at
/// zero (an add-on's `up`), where the array's own size gives no scale.
const ZERO_ARRAY_DIRECTION_SIZE: f32 = 0.01;

/// The smallest loss change per unit step the check judges. A loss near 0.7
/// rounds near 1e-7 in float32, so a central difference over steps of
/// [`DIRECTION_STEP`] resolves changes of about 1e-6; an array that moves
/// the loss less (a bias that shifts every option alike, say) is reported
/// but not judged.
pub const MINIMUM_MEASURABLE_CHANGE: f64 = 1e-5;

/// One array's check: the loss change per unit step along the direction the
/// gradient predicts, and the one a central difference measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrayGradientCheck {
    pub parameter_index: usize,
    pub predicted_change: f64,
    pub measured_change: f64,
}

impl ArrayGradientCheck {
    /// The larger of the predicted and the measured change: how much the
    /// array moves the loss by either account.
    #[must_use]
    pub fn change_size(&self) -> f64 {
        self.predicted_change.abs().max(self.measured_change.abs())
    }

    /// Whether the array is judged: it moves the loss measurably by either
    /// account, or either account is not a number (a NaN or infinite
    /// gradient is always judged, and always fails).
    #[must_use]
    pub fn is_measurable(&self) -> bool {
        !self.predicted_change.is_finite()
            || !self.measured_change.is_finite()
            || self.change_size() >= MINIMUM_MEASURABLE_CHANGE
    }

    /// How far the prediction is from the measurement, relative to the
    /// larger of the two, so a gradient the loss ignores reads as wholly
    /// wrong (1) rather than infinitely so, and a doubled one reads 0.5.
    #[must_use]
    pub fn relative_gap(&self) -> f64 {
        self.signed_relative_gap().abs()
    }

    /// [`Self::relative_gap`] with its sign: positive where the gradient
    /// predicts more change than the loss shows. A gradient scaled wrong
    /// leans one way across arrays; rounding does not.
    #[must_use]
    pub fn signed_relative_gap(&self) -> f64 {
        if !self.predicted_change.is_finite() || !self.measured_change.is_finite() {
            return f64::NAN;
        }
        let change_size = self.change_size();
        if change_size == 0.0 {
            return 0.0;
        }
        // Signed along the direction the loss really moves (or, where it
        // does not move, the one predicted), so "more" means more change.
        let reference_direction = if self.measured_change == 0.0 {
            self.predicted_change.signum()
        } else {
            self.measured_change.signum()
        };
        (self.predicted_change - self.measured_change) * reference_direction / change_size
    }

    /// Whether the gap is below `allowed_gap`; false for a gap that is not a
    /// number, which a plain comparison would let through.
    #[must_use]
    pub fn is_within(&self, allowed_gap: f64) -> bool {
        self.relative_gap() < allowed_gap
    }
}

/// What a gradient check is doing when it reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientCheckStage {
    /// Reading the job's examples through the model, as training does.
    Reading,
    /// Checking the arrays, two forward passes each.
    Checking,
}

/// Told how far each stage is; a whole model reads a few hundred examples
/// and has a few hundred arrays.
pub trait GradientCheckProgress {
    fn report(&mut self, stage: GradientCheckStage, completed: usize, total: usize);
}

/// Checks every parameter array at `parameters` over the training rows at
/// `batch_question_rows`, weighted equally, in parameter order.
///
/// # Errors
///
/// Refuses an empty batch, whose mean loss is not a number, and a scorer
/// that returns a gradient count other than one per array; returns the GPU
/// failure of any forward or backward pass.
pub fn directional_gradient_checks(
    math: GpuMath<'_>,
    training_set: &GpuTrainingSet,
    parameters: &[MlxArray],
    batch_question_rows: &[usize],
    progress: &mut dyn GradientCheckProgress,
) -> GpuResult<Vec<ArrayGradientCheck>> {
    if batch_question_rows.is_empty() {
        return Err(GpuTrainingError(
            "a gradient check needs at least one training row".to_owned(),
        ));
    }
    let row_weights = vec![1.0 / batch_question_rows.len() as f64; batch_question_rows.len()];
    let (question_batch, row_targets) =
        batch_objective(math, training_set, batch_question_rows, &row_weights)?;
    let gradients =
        training_set
            .scorer
            .parameter_gradients(math, parameters, &question_batch, &row_targets)?;
    if gradients.len() != parameters.len() {
        return Err(GpuTrainingError(format!(
            "the scorer returned {} gradients for {} arrays",
            gradients.len(),
            parameters.len()
        )));
    }
    let batch_loss = BatchLoss {
        training_set,
        question_batch: &question_batch,
        row_targets: &row_targets,
    };
    let mut array_checks = Vec::with_capacity(parameters.len());
    for (parameter_index, gradient) in gradients.iter().enumerate() {
        array_checks.push(check_one_array(
            math,
            &batch_loss,
            parameters,
            parameter_index,
            gradient,
        )?);
        progress.report(
            GradientCheckStage::Checking,
            parameter_index + 1,
            gradients.len(),
        );
    }
    Ok(array_checks)
}

/// The batch's mean loss, which equals the trained objective when every row
/// weighs the same.
struct BatchLoss<'batch> {
    training_set: &'batch GpuTrainingSet,
    question_batch: &'batch QuestionBatch,
    row_targets: &'batch RowTargets,
}

impl BatchLoss<'_> {
    fn at(&self, math: GpuMath<'_>, parameters: &[MlxArray]) -> GpuResult<f64> {
        let option_scores =
            self.training_set
                .scorer
                .option_scores(math, parameters, self.question_batch)?;
        mean_cross_entropy(math, &option_scores, self.question_batch, self.row_targets)
    }
}

fn check_one_array(
    math: GpuMath<'_>,
    batch_loss: &BatchLoss<'_>,
    parameters: &[MlxArray],
    parameter_index: usize,
    gradient: &MlxArray,
) -> GpuResult<ArrayGradientCheck> {
    let parameter_values = math.download(&parameters[parameter_index])?;
    let direction_values = pseudo_random_direction(&parameter_values, parameter_index);
    let gradient_values = math.download(gradient)?;
    let mut predicted_change = 0.0_f64;
    for (gradient_value, direction_value) in gradient_values.iter().zip(&direction_values) {
        predicted_change += f64::from(*gradient_value) * f64::from(*direction_value);
    }
    // Every array is measured, even one whose gradient is exactly zero: a
    // backward pass that loses a layer's term reports zeros while the loss
    // still moves, and only the measurement shows it.
    let parameter_shape = array_shape(&parameters[parameter_index])?;
    let mut stepped_losses = [0.0_f64; 2];
    for (stepped_loss, signed_step) in stepped_losses
        .iter_mut()
        .zip([DIRECTION_STEP, -DIRECTION_STEP])
    {
        let mut stepped_values = parameter_values.clone();
        for (stepped_value, direction_value) in stepped_values.iter_mut().zip(&direction_values) {
            *stepped_value += signed_step * direction_value;
        }
        let mut stepped_parameters = Vec::with_capacity(parameters.len());
        for (other_index, parameter) in parameters.iter().enumerate() {
            stepped_parameters.push(if other_index == parameter_index {
                math.upload(&stepped_values, &parameter_shape)?
            } else {
                parameter.retain()?
            });
        }
        *stepped_loss = batch_loss.at(math, &stepped_parameters)?;
    }
    Ok(ArrayGradientCheck {
        parameter_index,
        predicted_change,
        measured_change: (stepped_losses[0] - stepped_losses[1])
            / (2.0 * f64::from(DIRECTION_STEP)),
    })
}

fn array_shape(parameter: &MlxArray) -> GpuResult<Vec<usize>> {
    let mut dimensions = Vec::new();
    for dimension in parameter.shape() {
        match usize::try_from(dimension) {
            Ok(dimension) => dimensions.push(dimension),
            Err(_) => {
                return Err(GpuTrainingError(format!(
                    "a parameter reports the negative dimension {dimension}"
                )));
            }
        }
    }
    Ok(dimensions)
}

/// A direction the size of the array (its root-mean-square, or
/// [`ZERO_ARRAY_DIRECTION_SIZE`] for an array of zeros), from a linear
/// congruential sequence seeded by the array's position, so every run checks
/// the same directions.
fn pseudo_random_direction(parameter_values: &[f32], parameter_index: usize) -> Vec<f32> {
    let mut squared_sum = 0.0_f64;
    for parameter_value in parameter_values {
        squared_sum += f64::from(*parameter_value) * f64::from(*parameter_value);
    }
    let root_mean_square = (squared_sum / parameter_values.len().max(1) as f64).sqrt() as f32;
    let direction_size = if root_mean_square > 0.0 {
        root_mean_square
    } else {
        ZERO_ARRAY_DIRECTION_SIZE
    };
    // The seed spreads neighbouring array positions apart with the 64-bit
    // golden-ratio multiplier, and the step below is Knuth's MMIX linear
    // congruential generator; its top 24 bits fill an f32 mantissa evenly.
    // Any fixed sequence would do: what matters is that every run and every
    // backend checks the same directions, which no seeded library promises.
    let mut sequence_state = (parameter_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut direction_values = Vec::with_capacity(parameter_values.len());
    for _ in parameter_values {
        sequence_state = sequence_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Uniform on [-1, 1), whose root-mean-square is 1/sqrt(3).
        let unit_value = ((sequence_state >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
        direction_values.push(unit_value * direction_size * 3.0_f32.sqrt());
    }
    direction_values
}

#[cfg(test)]
#[path = "directional_gradient_check_tests.rs"]
mod tests;
