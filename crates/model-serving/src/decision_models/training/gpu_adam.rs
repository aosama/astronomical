//! Anchored Adam on the GPU: Adam whose weight decay pulls toward the
//! shipped weights instead of toward zero, with gradient clipping.
//!
//! The rules are `astronomical-decision-training`'s `anchored_adam`, which
//! explains them and is the oracle these steps are tested against. Every
//! element-wise update stays on the GPU; the one value read back per step is
//! the gradient's length, because a step whose gradient holds a non-finite
//! number must be refused before any optimizer state changes, and that
//! decision is the host's.

use astronomical_decision_training::anchored_adam::AdamSettings;
use mlx_bridge::MlxArray;

use super::gpu_math::{GpuMath, GpuResult};

/// What one step did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StepOutcome {
    Taken,
    /// The gradient held a non-finite number; nothing changed.
    RefusedNonFinite,
}

/// Adam's bias corrections for the current step count.
#[derive(Clone, Copy)]
struct BiasCorrections {
    first: f32,
    second: f32,
}

/// One parameter's two running averages after this step.
#[derive(Clone, Copy)]
struct MomentPair<'moments> {
    first: &'moments MlxArray,
    second: &'moments MlxArray,
}

/// The optimizer's running state, one array per parameter.
pub(crate) struct GpuAdam {
    settings: AdamSettings,
    first_moments: Vec<MlxArray>,
    second_moments: Vec<MlxArray>,
    step_count: i32,
}

impl GpuAdam {
    pub(crate) fn new(
        math: GpuMath<'_>,
        settings: AdamSettings,
        parameters: &[MlxArray],
    ) -> GpuResult<Self> {
        let mut first_moments = Vec::with_capacity(parameters.len());
        let mut second_moments = Vec::with_capacity(parameters.len());
        for parameter in parameters {
            first_moments.push(math.runtime.zeros_like(parameter)?);
            second_moments.push(math.runtime.zeros_like(parameter)?);
        }
        Ok(Self {
            settings,
            first_moments,
            second_moments,
            step_count: 0,
        })
    }

    /// Moves `parameters` one step down `gradients` and toward `anchors`.
    pub(crate) fn step(
        &mut self,
        math: GpuMath<'_>,
        parameters: &mut [MlxArray],
        gradients: &[MlxArray],
        anchors: &[MlxArray],
    ) -> GpuResult<StepOutcome> {
        let gradient_length = gradient_length(math, gradients)?;
        if !gradient_length.is_finite() {
            return Ok(StepOutcome::RefusedNonFinite);
        }
        let shrink = if gradient_length > self.settings.maximum_gradient_norm {
            self.settings.maximum_gradient_norm / gradient_length
        } else {
            1.0
        };
        self.step_count += 1;
        let settings = self.settings;
        let corrections = BiasCorrections {
            first: 1.0 - settings.first_moment_decay.powi(self.step_count),
            second: 1.0 - settings.second_moment_decay.powi(self.step_count),
        };
        for parameter_index in 0..parameters.len() {
            let gradient = math
                .runtime
                .multiply_scalar(&gradients[parameter_index], shrink)?;
            let (first_moment, second_moment) =
                self.next_moments(math, parameter_index, &gradient)?;
            let update = self.parameter_update(
                math,
                MomentPair {
                    first: &first_moment,
                    second: &second_moment,
                },
                corrections,
                &parameters[parameter_index],
                &anchors[parameter_index],
            )?;
            parameters[parameter_index] = math
                .runtime
                .subtract(&parameters[parameter_index], &update)?;
            self.first_moments[parameter_index] = first_moment;
            self.second_moments[parameter_index] = second_moment;
        }
        self.evaluate(math, parameters)?;
        Ok(StepOutcome::Taken)
    }

    /// The running averages of the gradient and of its square, moved on by
    /// this step's (clipped) gradient.
    fn next_moments(
        &self,
        math: GpuMath<'_>,
        parameter_index: usize,
        gradient: &MlxArray,
    ) -> GpuResult<(MlxArray, MlxArray)> {
        let runtime = math.runtime;
        let settings = self.settings;
        let first_moment = runtime.add(
            &runtime.multiply_scalar(
                &self.first_moments[parameter_index],
                settings.first_moment_decay,
            )?,
            &runtime.multiply_scalar(gradient, 1.0 - settings.first_moment_decay)?,
        )?;
        let second_moment = runtime.add(
            &runtime.multiply_scalar(
                &self.second_moments[parameter_index],
                settings.second_moment_decay,
            )?,
            &runtime.multiply_scalar(
                &runtime.square(gradient)?,
                1.0 - settings.second_moment_decay,
            )?,
        )?;
        Ok((first_moment, second_moment))
    }

    /// How far one parameter moves: the bias-corrected adaptive step plus
    /// the pull toward its shipped value, scaled by the learning rate.
    fn parameter_update(
        &self,
        math: GpuMath<'_>,
        moments: MomentPair<'_>,
        corrections: BiasCorrections,
        parameter: &MlxArray,
        anchor: &MlxArray,
    ) -> GpuResult<MlxArray> {
        let runtime = math.runtime;
        let settings = self.settings;
        let adaptive_step = runtime.divide(
            &runtime.multiply_scalar(moments.first, 1.0 / corrections.first)?,
            &runtime.add(
                &runtime
                    .sqrt(&runtime.multiply_scalar(moments.second, 1.0 / corrections.second)?)?,
                &math.scalar(settings.epsilon)?,
            )?,
        )?;
        let anchor_pull = runtime
            .multiply_scalar(&runtime.subtract(parameter, anchor)?, settings.anchor_decay)?;
        Ok(runtime.multiply_scalar(
            &runtime.add(&adaptive_step, &anchor_pull)?,
            settings.learning_rate,
        )?)
    }

    /// Runs the step's graph now, so each step's arrays are concrete and the
    /// next step's graph does not grow from them.
    fn evaluate(&self, math: GpuMath<'_>, parameters: &[MlxArray]) -> GpuResult<()> {
        let mut arrays = Vec::with_capacity(parameters.len() * 3);
        for parameter_index in 0..parameters.len() {
            arrays.push(&parameters[parameter_index]);
            arrays.push(&self.first_moments[parameter_index]);
            arrays.push(&self.second_moments[parameter_index]);
        }
        Ok(math.runtime.evaluate_arrays(&arrays)?)
    }
}

/// The length of all gradients together, read back to the host.
fn gradient_length(math: GpuMath<'_>, gradients: &[MlxArray]) -> GpuResult<f32> {
    let runtime = math.runtime;
    let mut squared_total = math.scalar(0.0)?;
    for gradient in gradients {
        squared_total = runtime.add(
            &squared_total,
            &runtime.sum(&runtime.square(gradient)?, false)?,
        )?;
    }
    let length = math.download(&runtime.sqrt(&squared_total)?)?;
    Ok(length.first().copied().unwrap_or(f32::NAN))
}

#[cfg(test)]
#[path = "gpu_adam_tests.rs"]
mod tests;
