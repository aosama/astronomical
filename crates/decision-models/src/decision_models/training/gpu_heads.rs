//! Each family's trainable part on the GPU: its option scores for a batch,
//! and the gradient of a loss with respect to every trainable tensor.
//!
//! The formulas are the ones `decision-training` implements on
//! the CPU and pins with finite differences; that CPU code is the oracle
//! the GPU passes are tested against. Parameters are held in the order the
//! family's `ParameterLayout` lays them out, so they write back under the
//! shipped names.

use mlx_bridge::MlxArray;

use super::gpu_math::{GpuMath, GpuResult};
use super::question_batch::QuestionBatch;

/// The ceiling `clamp(max=log(100))` puts on Clef's two logit scales.
const CLEF_MAXIMUM_LOG_SCALE: f32 = 4.605_170_2;

/// What a family's trainable part reads, for every question of a training,
/// uploaded once.
pub enum GpuHeadInputs {
    /// `[questions, hidden]` decision states and `[options, hidden]` option states.
    Kev {
        decision_states: MlxArray,
        option_states: MlxArray,
    },
    /// `[options, hidden]` marker states already normalized: the scorer's
    /// normalization input never changes, so only its scale and shift train.
    Laya { normalized_markers: MlxArray },
    /// `[options, 4·width]` residual features `[field, option, field·option,
    /// |field − option|]`, and `[options]` lexical priors and cosines.
    Clef {
        residual_features: MlxArray,
        lexical_priors: MlxArray,
        cosines: MlxArray,
    },
}

/// One family's trainable part.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuHeadFamily {
    Kev { pointer_dimension: usize },
    Laya,
    Clef,
}

impl GpuHeadFamily {
    /// The batch's option scores, in option order.
    pub fn option_scores(
        self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        head_inputs: &GpuHeadInputs,
        question_batch: &QuestionBatch,
    ) -> GpuResult<MlxArray> {
        Ok(self
            .forward(math, parameters, head_inputs, question_batch)?
            .option_scores)
    }

    /// `d loss / d parameter` for every parameter, given `d loss / d score`.
    pub fn parameter_gradients(
        self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        head_inputs: &GpuHeadInputs,
        question_batch: &QuestionBatch,
        score_gradients: &MlxArray,
    ) -> GpuResult<Vec<MlxArray>> {
        let forward_pass = self.forward(math, parameters, head_inputs, question_batch)?;
        match forward_pass.saved {
            SavedValues::Kev(saved) => kev_gradients(math, self, &saved, score_gradients),
            SavedValues::Dense(saved) => dense_gradients(math, parameters, &saved, score_gradients),
            SavedValues::Clef(saved) => clef_gradients(math, parameters, &saved, score_gradients),
        }
    }

    fn forward(
        self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        head_inputs: &GpuHeadInputs,
        question_batch: &QuestionBatch,
    ) -> GpuResult<ForwardPass> {
        match (self, head_inputs) {
            (
                Self::Kev { .. },
                GpuHeadInputs::Kev {
                    decision_states,
                    option_states,
                },
            ) => kev_forward(
                math,
                self,
                parameters,
                (decision_states, option_states),
                question_batch,
            ),
            (Self::Laya, GpuHeadInputs::Laya { normalized_markers }) => {
                laya_forward(math, parameters, normalized_markers, question_batch)
            }
            (
                Self::Clef,
                GpuHeadInputs::Clef {
                    residual_features,
                    lexical_priors,
                    cosines,
                },
            ) => clef_forward(
                math,
                parameters,
                (residual_features, lexical_priors, cosines),
                question_batch,
            ),
            _ => Err(super::gpu_math::GpuTrainingError(
                "the head inputs are another family's".to_owned(),
            )),
        }
    }

    fn kev_scale(self) -> f32 {
        match self {
            Self::Kev { pointer_dimension } => 1.0 / (pointer_dimension as f32).sqrt(),
            _ => 1.0,
        }
    }
}

struct ForwardPass {
    option_scores: MlxArray,
    saved: SavedValues,
}

enum SavedValues {
    Kev(KevSaved),
    Dense(DenseSaved),
    Clef(ClefSaved),
}

/// Kev's values the backward pass reuses.
struct KevSaved {
    decision_states: MlxArray,
    option_states: MlxArray,
    option_assignment: MlxArray,
    query_per_option: MlxArray,
    keys: MlxArray,
}

/// `[options, in] → Linear → GELU → Linear(1)`, shared by Laya's scorer
/// (after its normalization's scale and shift) and Clef's residual scorer.
struct DenseSaved {
    /// Laya's normalized markers, for the scale's gradient; `None` for Clef.
    normalized: Option<MlxArray>,
    hidden_input_source: MlxArray,
    hidden_input: MlxArray,
    hidden_activation: MlxArray,
}

struct ClefSaved {
    dense: DenseSaved,
    lexical_priors: MlxArray,
    cosines: MlxArray,
    residuals: MlxArray,
}

fn kev_forward(
    math: GpuMath<'_>,
    family: GpuHeadFamily,
    parameters: &[MlxArray],
    (decision_states, option_states): (&MlxArray, &MlxArray),
    question_batch: &QuestionBatch,
) -> GpuResult<ForwardPass> {
    let runtime = math.runtime;
    let batch_decisions =
        runtime.take_axis(decision_states, &question_batch.dataset_question_rows, 0)?;
    let batch_options = runtime.take_axis(option_states, &question_batch.dataset_option_rows, 0)?;
    let queries = math.linear(&batch_decisions, &parameters[0], &parameters[1])?;
    let keys = math.linear(&batch_options, &parameters[2], &parameters[3])?;
    let query_per_option = math.precise_matmul(&question_batch.option_assignment, &queries)?;
    let option_scores = runtime.multiply_scalar(
        &runtime.sum_axis(&runtime.multiply(&keys, &query_per_option)?, 1, false)?,
        family.kev_scale(),
    )?;
    Ok(ForwardPass {
        option_scores,
        saved: SavedValues::Kev(KevSaved {
            decision_states: batch_decisions,
            option_states: batch_options,
            option_assignment: question_batch.option_assignment.retain()?,
            query_per_option,
            keys,
        }),
    })
}

fn kev_gradients(
    math: GpuMath<'_>,
    family: GpuHeadFamily,
    saved: &KevSaved,
    score_gradients: &MlxArray,
) -> GpuResult<Vec<MlxArray>> {
    let runtime = math.runtime;
    let option_count = saved.keys.shape()[0];
    let scaled = runtime.reshape(
        &runtime.multiply_scalar(score_gradients, family.kev_scale())?,
        &[option_count, 1],
    )?;
    let key_gradients = runtime.multiply(&scaled, &saved.query_per_option)?;
    let query_per_option_gradients = runtime.multiply(&scaled, &saved.keys)?;
    let query_gradients =
        math.transposed_product(&saved.option_assignment, &query_per_option_gradients)?;
    Ok(vec![
        math.transposed_product(&query_gradients, &saved.decision_states)?,
        math.column_sum(&query_gradients)?,
        math.transposed_product(&key_gradients, &saved.option_states)?,
        math.column_sum(&key_gradients)?,
    ])
}

fn laya_forward(
    math: GpuMath<'_>,
    parameters: &[MlxArray],
    normalized_markers: &MlxArray,
    question_batch: &QuestionBatch,
) -> GpuResult<ForwardPass> {
    let runtime = math.runtime;
    let normalized =
        runtime.take_axis(normalized_markers, &question_batch.dataset_option_rows, 0)?;
    let scaled = runtime.add(
        &runtime.multiply(&normalized, &parameters[0])?,
        &parameters[1],
    )?;
    let (option_scores, mut dense) = dense_scorer_forward(math, &scaled, &parameters[2..6])?;
    dense.normalized = Some(normalized);
    Ok(ForwardPass {
        option_scores,
        saved: SavedValues::Dense(dense),
    })
}

/// `Linear → GELU → Linear(1)` over `[options, in]`, returning `[options]`.
fn dense_scorer_forward(
    math: GpuMath<'_>,
    scorer_input: &MlxArray,
    dense_parameters: &[MlxArray],
) -> GpuResult<(MlxArray, DenseSaved)> {
    let runtime = math.runtime;
    let hidden_input = math.linear(scorer_input, &dense_parameters[0], &dense_parameters[1])?;
    let hidden_activation = runtime.gelu(&hidden_input)?;
    let single_scores = math.linear(
        &hidden_activation,
        &dense_parameters[2],
        &dense_parameters[3],
    )?;
    let option_count = single_scores.shape()[0];
    Ok((
        runtime.reshape(&single_scores, &[option_count])?,
        DenseSaved {
            normalized: None,
            hidden_input_source: scorer_input.retain()?,
            hidden_input,
            hidden_activation,
        },
    ))
}

/// The dense scorer's four gradients, then (for Laya) the normalization's scale and shift.
fn dense_gradients(
    math: GpuMath<'_>,
    parameters: &[MlxArray],
    saved: &DenseSaved,
    score_gradients: &MlxArray,
) -> GpuResult<Vec<MlxArray>> {
    let runtime = math.runtime;
    let dense_offset = if saved.normalized.is_some() { 2 } else { 0 };
    let dense_parameters = &parameters[dense_offset..dense_offset + 4];
    let (mut gradients, input_gradients) =
        dense_scorer_gradients(math, dense_parameters, saved, score_gradients)?;
    if let Some(normalized) = &saved.normalized {
        let scale_gradient = math.column_sum(&runtime.multiply(&input_gradients, normalized)?)?;
        let shift_gradient = math.column_sum(&input_gradients)?;
        gradients.insert(0, shift_gradient);
        gradients.insert(0, scale_gradient);
    }
    Ok(gradients)
}

/// The dense scorer's weight and bias gradients, and the gradient flowing
/// back into its input.
fn dense_scorer_gradients(
    math: GpuMath<'_>,
    dense_parameters: &[MlxArray],
    saved: &DenseSaved,
    score_gradients: &MlxArray,
) -> GpuResult<(Vec<MlxArray>, MlxArray)> {
    let runtime = math.runtime;
    let option_count = saved.hidden_input.shape()[0];
    let column_gradients = runtime.reshape(score_gradients, &[option_count, 1])?;
    let output_weight_gradient =
        math.transposed_product(&column_gradients, &saved.hidden_activation)?;
    let output_bias_gradient = math.column_sum(&column_gradients)?;
    let hidden_input_gradients = runtime.multiply(
        &runtime.multiply(&column_gradients, &dense_parameters[2])?,
        &math.gelu_derivative(&saved.hidden_input)?,
    )?;
    let hidden_weight_gradient =
        math.transposed_product(&hidden_input_gradients, &saved.hidden_input_source)?;
    let hidden_bias_gradient = math.column_sum(&hidden_input_gradients)?;
    let input_gradients = math.precise_matmul(&hidden_input_gradients, &dense_parameters[0])?;
    Ok((
        vec![
            hidden_weight_gradient,
            hidden_bias_gradient,
            output_weight_gradient,
            output_bias_gradient,
        ],
        input_gradients,
    ))
}

fn clef_forward(
    math: GpuMath<'_>,
    parameters: &[MlxArray],
    (residual_features, lexical_priors, cosines): (&MlxArray, &MlxArray, &MlxArray),
    question_batch: &QuestionBatch,
) -> GpuResult<ForwardPass> {
    let runtime = math.runtime;
    let rows = &question_batch.dataset_option_rows;
    let features = runtime.take_axis(residual_features, rows, 0)?;
    let batch_priors = runtime.take_axis(lexical_priors, rows, 0)?;
    let batch_cosines = runtime.take_axis(cosines, rows, 0)?;
    let (residuals, dense) = dense_scorer_forward(math, &features, &parameters[0..4])?;
    let mix = clef_mix(math, parameters)?;
    let option_scores = clef_mixed_scores(math, &mix, &residuals, &batch_priors, &batch_cosines)?;
    Ok(ForwardPass {
        option_scores,
        saved: SavedValues::Clef(ClefSaved {
            dense,
            lexical_priors: batch_priors,
            cosines: batch_cosines,
            residuals,
        }),
    })
}

/// `prior = exp(min(p, ln 100))`, `agreement = exp(min(j, ln 100))`,
/// `gate = sigmoid(g)`, and whether each scale is below its ceiling (1 or 0).
/// Clef's option scores from what its trainable final layers read, in
/// option order: `prior · lexical prior + gate · (agreement · cosine +
/// residual scorer(features))`. A whole-model training mixes the terms its
/// graph computed with this, so both trainings score with one formula.
pub fn clef_option_scores(
    math: GpuMath<'_>,
    parameters: &[MlxArray],
    (residual_features, lexical_priors, cosines): (&MlxArray, &MlxArray, &MlxArray),
) -> GpuResult<MlxArray> {
    let (residuals, _dense) = dense_scorer_forward(math, residual_features, &parameters[0..4])?;
    let mix = clef_mix(math, parameters)?;
    clef_mixed_scores(math, &mix, &residuals, lexical_priors, cosines)
}

fn clef_mixed_scores(
    math: GpuMath<'_>,
    mix: &ClefMix,
    residuals: &MlxArray,
    lexical_priors: &MlxArray,
    cosines: &MlxArray,
) -> GpuResult<MlxArray> {
    let runtime = math.runtime;
    let joint = runtime.add(&runtime.multiply(&mix.agreement, cosines)?, residuals)?;
    Ok(runtime.add(
        &runtime.multiply(&mix.prior, lexical_priors)?,
        &runtime.multiply(&mix.gate, &joint)?,
    )?)
}

struct ClefMix {
    prior: MlxArray,
    agreement: MlxArray,
    gate: MlxArray,
    prior_below_ceiling: MlxArray,
    agreement_below_ceiling: MlxArray,
}

fn clef_mix(math: GpuMath<'_>, parameters: &[MlxArray]) -> GpuResult<ClefMix> {
    let runtime = math.runtime;
    let ceiling = math.scalar(CLEF_MAXIMUM_LOG_SCALE)?;
    Ok(ClefMix {
        prior: runtime.exp(&runtime.minimum(&parameters[4], &ceiling)?)?,
        agreement: runtime.exp(&runtime.minimum(&parameters[5], &ceiling)?)?,
        gate: runtime.sigmoid(&parameters[6])?,
        prior_below_ceiling: below_ceiling(math, &parameters[4], &ceiling)?,
        agreement_below_ceiling: below_ceiling(math, &parameters[5], &ceiling)?,
    })
}

/// One where `log_scale` is below the ceiling, zero at or above it: past the
/// ceiling the scale is constant, so its gradient stops.
fn below_ceiling(
    math: GpuMath<'_>,
    log_scale: &MlxArray,
    ceiling: &MlxArray,
) -> GpuResult<MlxArray> {
    let runtime = math.runtime;
    Ok(runtime.astype(
        &runtime.less(log_scale, ceiling)?,
        mlx_bridge::MlxDtype::Float32,
    )?)
}

/// `Σ score_gradient · value` over the batch's options.
fn summed_product(
    math: GpuMath<'_>,
    score_gradients: &MlxArray,
    values: &MlxArray,
) -> GpuResult<MlxArray> {
    let runtime = math.runtime;
    Ok(runtime.sum(&runtime.multiply(score_gradients, values)?, false)?)
}

fn clef_gradients(
    math: GpuMath<'_>,
    parameters: &[MlxArray],
    saved: &ClefSaved,
    score_gradients: &MlxArray,
) -> GpuResult<Vec<MlxArray>> {
    let runtime = math.runtime;
    let mix = clef_mix(math, parameters)?;
    let residual_gradients = runtime.multiply(score_gradients, &mix.gate)?;
    let (mut gradients, _) =
        dense_scorer_gradients(math, &parameters[0..4], &saved.dense, &residual_gradients)?;
    gradients.extend(clef_scalar_gradients(math, &mix, saved, score_gradients)?);
    Ok(gradients)
}

/// The gradients of the three learned scalars: the prior's log scale, the
/// agreement's log scale and the residual gate. Each log scale stops
/// learning once it reaches the ceiling the forward pass clamps it to.
fn clef_scalar_gradients(
    math: GpuMath<'_>,
    mix: &ClefMix,
    saved: &ClefSaved,
    score_gradients: &MlxArray,
) -> GpuResult<[MlxArray; 3]> {
    let runtime = math.runtime;
    let prior_gradient = runtime.multiply(
        &runtime.multiply(
            &summed_product(math, score_gradients, &saved.lexical_priors)?,
            &mix.prior,
        )?,
        &mix.prior_below_ceiling,
    )?;
    let agreement_gradient = runtime.multiply(
        &runtime.multiply(
            &runtime.multiply(
                &summed_product(math, score_gradients, &saved.cosines)?,
                &mix.gate,
            )?,
            &mix.agreement,
        )?,
        &mix.agreement_below_ceiling,
    )?;
    let gate_gradient = clef_gate_gradient(math, mix, saved, score_gradients)?;
    Ok([prior_gradient, agreement_gradient, gate_gradient])
}

/// The gate scales the agreement and residual terms together, through a
/// sigmoid, so its slope is gate × (1 − gate).
fn clef_gate_gradient(
    math: GpuMath<'_>,
    mix: &ClefMix,
    saved: &ClefSaved,
    score_gradients: &MlxArray,
) -> GpuResult<MlxArray> {
    let runtime = math.runtime;
    let joint = runtime.add(
        &runtime.multiply(&mix.agreement, &saved.cosines)?,
        &saved.residuals,
    )?;
    let gate_slope =
        runtime.multiply(&mix.gate, &runtime.subtract(&math.scalar(1.0)?, &mix.gate)?)?;
    Ok(runtime.multiply(&summed_product(math, score_gradients, &joint)?, &gate_slope)?)
}

#[cfg(test)]
#[path = "gpu_heads_tests.rs"]
mod tests;
