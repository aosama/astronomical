//! Adam with its weight decay pulling toward the shipped weights instead of
//! toward zero.
//!
//! Ordinary weight decay shrinks every weight toward zero, which for a head
//! that starts out trained means steadily forgetting what it knew. Pulling
//! toward the shipped weights instead (L2-SP, Li et al. 2018) lets a few
//! hundred examples move the head only as far as they keep paying for. The
//! pull is decoupled from the gradient, as AdamW decouples its decay, so it
//! does not depend on how large the gradients happen to be.
//!
//! A step whose gradient holds a non-finite number is refused before any of
//! the optimizer's state changes: an infinity folded into the running
//! averages would poison every later step, a known way runs quietly destroy
//! weights.

/// The settings one training run uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamSettings {
    pub learning_rate: f32,
    pub first_moment_decay: f32,
    pub second_moment_decay: f32,
    pub epsilon: f32,
    /// How strongly each step pulls the weights back toward the shipped ones.
    pub anchor_decay: f32,
    /// The longest a gradient may be before it is scaled down to this length.
    pub maximum_gradient_norm: f32,
}

/// What the stage 2 runs use. A head learning rate of 1e-4 and a gradient
/// clip of 1.0 are Laya's own training defaults; the anchor pull is a
/// conservative choice for a few hundred examples.
pub const HEAD_TRAINING_ADAM_SETTINGS: AdamSettings = AdamSettings {
    learning_rate: 1e-4,
    first_moment_decay: 0.9,
    second_moment_decay: 0.999,
    epsilon: 1e-8,
    anchor_decay: 0.01,
    maximum_gradient_norm: 1.0,
};

/// Why a step was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepRefusal {
    NonFiniteGradient,
}

/// The optimizer's running state for one parameter vector.
pub struct AnchoredAdam {
    settings: AdamSettings,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
    step_count: i32,
}

impl AnchoredAdam {
    #[must_use]
    pub fn new(settings: AdamSettings, parameter_count: usize) -> Self {
        Self {
            settings,
            first_moment: vec![0.0; parameter_count],
            second_moment: vec![0.0; parameter_count],
            step_count: 0,
        }
    }

    /// Moves `parameters` one step down `gradient` and toward `anchor`.
    /// The gradient is scaled in place when it is longer than allowed.
    pub fn step(
        &mut self,
        parameters: &mut [f32],
        gradient: &mut [f32],
        anchor: &[f32],
    ) -> Result<(), StepRefusal> {
        let mut squared_norm = 0.0_f64;
        for gradient_value in gradient.iter() {
            if !gradient_value.is_finite() {
                return Err(StepRefusal::NonFiniteGradient);
            }
            squared_norm += f64::from(*gradient_value) * f64::from(*gradient_value);
        }
        let gradient_norm = squared_norm.sqrt() as f32;
        if gradient_norm > self.settings.maximum_gradient_norm {
            let shrink = self.settings.maximum_gradient_norm / gradient_norm;
            for gradient_value in gradient.iter_mut() {
                *gradient_value *= shrink;
            }
        }
        self.step_count += 1;
        let settings = self.settings;
        let first_correction = 1.0 - settings.first_moment_decay.powi(self.step_count);
        let second_correction = 1.0 - settings.second_moment_decay.powi(self.step_count);
        for ((((parameter, gradient_value), first_moment), second_moment), anchor_value) in
            parameters
                .iter_mut()
                .zip(gradient.iter())
                .zip(self.first_moment.iter_mut())
                .zip(self.second_moment.iter_mut())
                .zip(anchor)
        {
            *first_moment = settings.first_moment_decay * *first_moment
                + (1.0 - settings.first_moment_decay) * gradient_value;
            *second_moment = settings.second_moment_decay * *second_moment
                + (1.0 - settings.second_moment_decay) * gradient_value * gradient_value;
            let adaptive_step = (*first_moment / first_correction)
                / ((*second_moment / second_correction).sqrt() + settings.epsilon);
            let anchor_pull = settings.anchor_decay * (*parameter - anchor_value);
            *parameter -= settings.learning_rate * (adaptive_step + anchor_pull);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "anchored_adam_tests.rs"]
mod tests;
