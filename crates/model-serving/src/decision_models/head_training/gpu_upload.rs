//! Putting a training's inputs and starting weights on the GPU, once.
//!
//! The engines hand over what each question's final layers read; this
//! stacks it into one array per kind of input and computes, on the GPU, the
//! parts that never change while training: Laya's normalized marker states
//! and Clef's residual features.

use astronomical_decision_contract::HeadInputs;
use astronomical_decision_contract::head_tuning_file::TunableHeadFamily;
use astronomical_decision_training::head_parameters::ParameterLayout;
use mlx_bridge::MlxArray;

use super::gpu_heads::{GpuHeadFamily, GpuHeadInputs};
use super::gpu_math::{GpuMath, GpuResult, GpuTrainingError};
use super::question_batch::OptionLayout;

/// Laya's scorer normalization epsilon, PyTorch's default, as serving uses it.
const LAYA_SCORER_LAYER_NORM_EPSILON: f32 = 1e-5;

/// Stacks every question's head inputs into GPU arrays, with their layout.
pub(crate) fn upload_head_inputs(
    math: GpuMath<'_>,
    family: TunableHeadFamily,
    question_inputs: &[&HeadInputs],
) -> GpuResult<(GpuHeadInputs, OptionLayout)> {
    let mut option_counts = Vec::with_capacity(question_inputs.len());
    for head_inputs in question_inputs {
        match head_inputs.option_count() {
            Some(option_count) => option_counts.push(option_count),
            None => {
                return Err(GpuTrainingError(
                    "a question's inputs disagree about its options".to_owned(),
                ));
            }
        }
    }
    let gpu_inputs = match family {
        TunableHeadFamily::Kev => upload_kev(math, question_inputs)?,
        TunableHeadFamily::Laya => upload_laya(math, question_inputs)?,
        TunableHeadFamily::Clef => upload_clef(math, question_inputs)?,
    };
    Ok((gpu_inputs, OptionLayout::new(option_counts)))
}

/// The trainable part's parameters on the GPU, in layout order and shape.
pub(crate) fn upload_parameters(
    math: GpuMath<'_>,
    layout: &ParameterLayout,
    parameter_values: &[f32],
) -> GpuResult<Vec<MlxArray>> {
    let mut parameters = Vec::with_capacity(layout.tensors.len());
    for parameter_tensor in &layout.tensors {
        parameters.push(math.upload(
            parameter_tensor.values(parameter_values),
            &parameter_tensor.shape,
        )?);
    }
    Ok(parameters)
}

/// The parameters read back into one flat host vector in layout order.
pub(crate) fn download_parameters(
    math: GpuMath<'_>,
    parameters: &[MlxArray],
) -> GpuResult<Vec<f32>> {
    let mut parameter_values = Vec::new();
    for parameter in parameters {
        parameter_values.extend(math.download(parameter)?);
    }
    Ok(parameter_values)
}

/// The GPU form of `family`'s trainable part, sized from its layout.
pub(crate) fn gpu_head_family(
    family: TunableHeadFamily,
    layout: &ParameterLayout,
) -> GpuHeadFamily {
    match family {
        TunableHeadFamily::Kev => GpuHeadFamily::Kev {
            pointer_dimension: layout.tensor("q.bias").length,
        },
        TunableHeadFamily::Laya => GpuHeadFamily::Laya,
        TunableHeadFamily::Clef => GpuHeadFamily::Clef,
    }
}

fn upload_kev(math: GpuMath<'_>, question_inputs: &[&HeadInputs]) -> GpuResult<GpuHeadInputs> {
    let mut decision_values = Vec::new();
    let mut option_values = Vec::new();
    let mut option_total = 0;
    let mut hidden_size = 0;
    for head_inputs in question_inputs {
        let HeadInputs::Kev {
            decision_state,
            option_states,
        } = head_inputs
        else {
            return Err(wrong_family());
        };
        hidden_size = decision_state.0.len();
        decision_values.extend_from_slice(&decision_state.0);
        for option_state in option_states {
            option_values.extend_from_slice(&option_state.0);
            option_total += 1;
        }
    }
    Ok(GpuHeadInputs::Kev {
        decision_states: math.upload(&decision_values, &[question_inputs.len(), hidden_size])?,
        option_states: math.upload(&option_values, &[option_total, hidden_size])?,
    })
}

fn upload_laya(math: GpuMath<'_>, question_inputs: &[&HeadInputs]) -> GpuResult<GpuHeadInputs> {
    let mut marker_values = Vec::new();
    let mut option_total = 0;
    let mut marker_width = 0;
    for head_inputs in question_inputs {
        let HeadInputs::Laya { marker_states } = head_inputs else {
            return Err(wrong_family());
        };
        for marker_state in marker_states {
            marker_width = marker_state.0.len();
            marker_values.extend_from_slice(&marker_state.0);
            option_total += 1;
        }
    }
    let markers = math.upload(&marker_values, &[option_total, marker_width])?;
    Ok(GpuHeadInputs::Laya {
        normalized_markers: math
            .runtime
            .layer_norm_without_weight_and_bias(&markers, LAYA_SCORER_LAYER_NORM_EPSILON)?,
    })
}

fn upload_clef(math: GpuMath<'_>, question_inputs: &[&HeadInputs]) -> GpuResult<GpuHeadInputs> {
    let mut field_values = Vec::new();
    let mut option_values = Vec::new();
    let mut prior_values = Vec::new();
    let mut cosine_values = Vec::new();
    let mut width = 0;
    for head_inputs in question_inputs {
        let HeadInputs::Clef {
            field_vectors,
            option_vectors,
            lexical_priors,
            cosines,
        } = head_inputs
        else {
            return Err(wrong_family());
        };
        for (field_vector, option_vector) in field_vectors.iter().zip(option_vectors) {
            width = field_vector.0.len();
            field_values.extend_from_slice(&field_vector.0);
            option_values.extend_from_slice(&option_vector.0);
        }
        prior_values.extend_from_slice(lexical_priors);
        cosine_values.extend_from_slice(cosines);
    }
    let option_total = prior_values.len();
    let runtime = math.runtime;
    let fields = math.upload(&field_values, &[option_total, width])?;
    let options = math.upload(&option_values, &[option_total, width])?;
    let products = runtime.multiply(&fields, &options)?;
    let distances = runtime.abs(&runtime.subtract(&fields, &options)?)?;
    Ok(GpuHeadInputs::Clef {
        residual_features: runtime
            .concatenate_axis(&[&fields, &options, &products, &distances], 1)?,
        lexical_priors: math.upload(&prior_values, &[option_total])?,
        cosines: math.upload(&cosine_values, &[option_total])?,
    })
}

fn wrong_family() -> GpuTrainingError {
    GpuTrainingError("the head inputs are another family's".to_owned())
}
