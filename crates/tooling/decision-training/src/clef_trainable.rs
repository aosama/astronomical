//! Clef's residual scorer and final mix as a trainable part.
//!
//! Each option's logit is `prior · lexical + gate · (agreement · cosine + residual)`,
//! where `prior = exp(min(p, ln 100))`, `agreement = exp(min(j, ln 100))`,
//! `gate = sigmoid(g)` and `residual` is `Linear(1) ∘ GELU ∘ Linear` over
//! `[field, option, field · option, |field − option|]`. The field and option
//! vectors, the lexical prior and the cosine come from the frozen routing and
//! decoder layers; the residual scorer and the three scalars train, about
//! 4.2 million numbers.

use decision_contract::HeadInputs;

use crate::dense_math::{dot, linear_forward, linear_parameter_gradients};
use crate::gelu::{gelu, gelu_derivative};
use crate::head_parameters::{ParameterLayout, ParameterTensor, split_weight_and_bias};
use crate::trainable_head::{HeadInputsMismatch, TrainableHead, input_family_name};

/// The ceiling `clamp(max=log(100))` puts on both logit scales, as serving applies it.
const MAXIMUM_LOGIT_SCALE: f64 = 100.0;

/// Clef's trainable part for one field width and residual width.
pub struct ClefTrainableHead {
    layout: ParameterLayout,
    field_width: usize,
    residual_width: usize,
    residual_weight: ParameterTensor,
    residual_bias: ParameterTensor,
    score_weight: ParameterTensor,
    score_bias: ParameterTensor,
    prior_logit_scale: ParameterTensor,
    joint_logit_scale: ParameterTensor,
    residual_gate: ParameterTensor,
}

/// One option's inputs, borrowed from the head inputs.
struct ClefOption<'inputs> {
    field_vector: &'inputs [f32],
    option_vector: &'inputs [f32],
    lexical_prior: f32,
    cosine: f32,
}

/// One option's values on the way through the residual scorer.
struct ResidualPass {
    features: Vec<f32>,
    hidden_input: Vec<f32>,
    hidden_activation: Vec<f32>,
    residual: f32,
}

/// The three mixing numbers, as serving derives them from the scalars.
struct MixScales {
    prior: f64,
    agreement: f64,
    gate: f64,
}

impl ClefTrainableHead {
    #[must_use]
    pub fn new(field_width: usize, residual_width: usize) -> Self {
        let layout = ParameterLayout::new(&[
            (
                "residual_scorer.0.weight",
                vec![residual_width, 4 * field_width],
            ),
            ("residual_scorer.0.bias", vec![residual_width]),
            ("residual_scorer.3.weight", vec![1, residual_width]),
            ("residual_scorer.3.bias", vec![1]),
            ("prior_logit_scale", Vec::new()),
            ("joint_logit_scale", Vec::new()),
            ("residual_gate", Vec::new()),
        ]);
        Self {
            residual_weight: layout.tensor("residual_scorer.0.weight").clone(),
            residual_bias: layout.tensor("residual_scorer.0.bias").clone(),
            score_weight: layout.tensor("residual_scorer.3.weight").clone(),
            score_bias: layout.tensor("residual_scorer.3.bias").clone(),
            prior_logit_scale: layout.tensor("prior_logit_scale").clone(),
            joint_logit_scale: layout.tensor("joint_logit_scale").clone(),
            residual_gate: layout.tensor("residual_gate").clone(),
            layout,
            field_width,
            residual_width,
        }
    }

    fn mix_scales(&self, parameters: &[f32]) -> MixScales {
        let maximum_log_scale = MAXIMUM_LOGIT_SCALE.ln();
        let prior_log = f64::from(self.prior_logit_scale.values(parameters)[0]);
        let agreement_log = f64::from(self.joint_logit_scale.values(parameters)[0]);
        let gate_log = f64::from(self.residual_gate.values(parameters)[0]);
        MixScales {
            prior: prior_log.min(maximum_log_scale).exp(),
            agreement: agreement_log.min(maximum_log_scale).exp(),
            gate: 1.0 / (1.0 + (-gate_log).exp()),
        }
    }

    fn residual_pass(&self, parameters: &[f32], clef_option: &ClefOption<'_>) -> ResidualPass {
        let mut features = Vec::with_capacity(4 * self.field_width);
        features.extend_from_slice(clef_option.field_vector);
        features.extend_from_slice(clef_option.option_vector);
        for (field_value, option_value) in clef_option
            .field_vector
            .iter()
            .zip(clef_option.option_vector)
        {
            features.push(field_value * option_value);
        }
        for (field_value, option_value) in clef_option
            .field_vector
            .iter()
            .zip(clef_option.option_vector)
        {
            features.push((field_value - option_value).abs());
        }
        let mut hidden_input = Vec::with_capacity(self.residual_width);
        linear_forward(
            self.residual_weight.values(parameters),
            self.residual_bias.values(parameters),
            &features,
            &mut hidden_input,
        );
        let mut hidden_activation = Vec::with_capacity(self.residual_width);
        for hidden_value in &hidden_input {
            hidden_activation.push(gelu(*hidden_value));
        }
        let residual = dot(self.score_weight.values(parameters), &hidden_activation)
            + self.score_bias.values(parameters)[0];
        ResidualPass {
            features,
            hidden_input,
            hidden_activation,
            residual,
        }
    }

    /// `prior · lexical + gate · (agreement · cosine + residual)`, in 64-bit
    /// as serving combines it.
    fn option_logit(mix_scales: &MixScales, clef_option: &ClefOption<'_>, residual: f32) -> f32 {
        let joint = mix_scales.agreement * f64::from(clef_option.cosine) + f64::from(residual);
        (mix_scales.prior * f64::from(clef_option.lexical_prior) + mix_scales.gate * joint) as f32
    }

    fn add_scalar_gradients(
        &self,
        parameters: &[f32],
        mix_scales: &MixScales,
        clef_option: &ClefOption<'_>,
        residual: f32,
        logit_gradient: f64,
        gradient: &mut [f32],
    ) {
        let maximum_log_scale = MAXIMUM_LOGIT_SCALE.ln();
        // Past the ceiling the scale is constant, so its gradient is zero.
        if f64::from(self.prior_logit_scale.values(parameters)[0]) < maximum_log_scale {
            self.prior_logit_scale.values_mut(gradient)[0] +=
                (logit_gradient * mix_scales.prior * f64::from(clef_option.lexical_prior)) as f32;
        }
        if f64::from(self.joint_logit_scale.values(parameters)[0]) < maximum_log_scale {
            self.joint_logit_scale.values_mut(gradient)[0] += (logit_gradient
                * mix_scales.gate
                * mix_scales.agreement
                * f64::from(clef_option.cosine))
                as f32;
        }
        let joint = mix_scales.agreement * f64::from(clef_option.cosine) + f64::from(residual);
        self.residual_gate.values_mut(gradient)[0] +=
            (logit_gradient * mix_scales.gate * (1.0 - mix_scales.gate) * joint) as f32;
    }

    fn add_residual_gradients(
        &self,
        parameters: &[f32],
        residual_pass: &ResidualPass,
        residual_gradient: f32,
        gradient: &mut [f32],
    ) {
        let (score_weight_gradient, score_bias_gradient) =
            split_weight_and_bias(gradient, &self.score_weight, &self.score_bias);
        linear_parameter_gradients(
            &residual_pass.hidden_activation,
            &[residual_gradient],
            score_weight_gradient,
            score_bias_gradient,
        );
        let mut hidden_input_gradient = Vec::with_capacity(self.residual_width);
        for (score_weight_value, hidden_value) in self
            .score_weight
            .values(parameters)
            .iter()
            .zip(&residual_pass.hidden_input)
        {
            hidden_input_gradient
                .push(residual_gradient * score_weight_value * gelu_derivative(*hidden_value));
        }
        let (residual_weight_gradient, residual_bias_gradient) =
            split_weight_and_bias(gradient, &self.residual_weight, &self.residual_bias);
        linear_parameter_gradients(
            &residual_pass.features,
            &hidden_input_gradient,
            residual_weight_gradient,
            residual_bias_gradient,
        );
    }
}

fn clef_options(head_inputs: &HeadInputs) -> Option<Vec<ClefOption<'_>>> {
    let HeadInputs::Clef {
        field_vectors,
        option_vectors,
        lexical_priors,
        cosines,
    } = head_inputs
    else {
        return None;
    };
    head_inputs.option_count()?;
    let mut clef_options = Vec::with_capacity(option_vectors.len());
    for (((field_vector, option_vector), lexical_prior), cosine) in field_vectors
        .iter()
        .zip(option_vectors)
        .zip(lexical_priors)
        .zip(cosines)
    {
        clef_options.push(ClefOption {
            field_vector: &field_vector.0,
            option_vector: &option_vector.0,
            lexical_prior: *lexical_prior,
            cosine: *cosine,
        });
    }
    Some(clef_options)
}

impl TrainableHead for ClefTrainableHead {
    fn layout(&self) -> &ParameterLayout {
        &self.layout
    }

    fn check_inputs(&self, head_inputs: &HeadInputs) -> Result<usize, HeadInputsMismatch> {
        if !matches!(head_inputs, HeadInputs::Clef { .. }) {
            return Err(HeadInputsMismatch::WrongFamily {
                expected: "Clef",
                found: input_family_name(head_inputs),
            });
        }
        let Some(clef_options) = clef_options(head_inputs) else {
            return Err(HeadInputsMismatch::OptionCountDisagrees);
        };
        if clef_options.is_empty() {
            return Err(HeadInputsMismatch::NoOptions);
        }
        for clef_option in &clef_options {
            for vector in [clef_option.field_vector, clef_option.option_vector] {
                if vector.len() != self.field_width {
                    return Err(HeadInputsMismatch::WrongWidth {
                        expected: self.field_width,
                        found: vector.len(),
                    });
                }
            }
        }
        Ok(clef_options.len())
    }

    fn option_logits(&self, parameters: &[f32], head_inputs: &HeadInputs) -> Vec<f32> {
        let Some(clef_options) = clef_options(head_inputs) else {
            return Vec::new();
        };
        let mix_scales = self.mix_scales(parameters);
        let mut option_logits = Vec::with_capacity(clef_options.len());
        for clef_option in &clef_options {
            let residual = self.residual_pass(parameters, clef_option).residual;
            option_logits.push(Self::option_logit(&mix_scales, clef_option, residual));
        }
        option_logits
    }

    fn add_gradient(
        &self,
        parameters: &[f32],
        head_inputs: &HeadInputs,
        logit_gradients: &[f32],
        gradient: &mut [f32],
    ) {
        let Some(clef_options) = clef_options(head_inputs) else {
            return;
        };
        let mix_scales = self.mix_scales(parameters);
        for (clef_option, logit_gradient) in clef_options.iter().zip(logit_gradients) {
            if *logit_gradient == 0.0 {
                continue;
            }
            let residual_pass = self.residual_pass(parameters, clef_option);
            let wide_gradient = f64::from(*logit_gradient);
            self.add_scalar_gradients(
                parameters,
                &mix_scales,
                clef_option,
                residual_pass.residual,
                wide_gradient,
                gradient,
            );
            let residual_gradient = (wide_gradient * mix_scales.gate) as f32;
            self.add_residual_gradients(parameters, &residual_pass, residual_gradient, gradient);
        }
    }
}

#[cfg(test)]
#[path = "clef_trainable_tests.rs"]
mod tests;
