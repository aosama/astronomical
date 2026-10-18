//! Request sampling policy and structured-token constraints for folded-ternary output.

use astronomical_mlx_c_rust::MlxArray;

use super::PrismHadamardQwen35ActiveGeneration;
use crate::InferenceEngineError;
use crate::Qwen3_5SamplingStrategy;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::structured_generation::StructuredTokenConstraint;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SamplingPlan {
    HighestLogit,
    TopKTopP {
        temperature_thousandths: u16,
        top_k: u16,
        top_p_thousandths: u16,
    },
}

pub(super) fn sample_token_id(
    model: &PrismHadamardTextModel,
    logits: &MlxArray,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
) -> Result<u32, InferenceEngineError> {
    let repetition_penalized_logits = crate::gpu_token_sampling::apply_repetition_penalty(
        &model.runtime,
        logits,
        &active_generation.penalized_token_ids,
        active_generation.request.repetition_penalty_thousandths(),
    )?;
    let logits_after_repetition_penalty = repetition_penalized_logits.as_ref().unwrap_or(logits);
    let structured_logit_bias_values = structured_logit_bias_values_if_visible(
        active_generation.structured_generation.as_mut(),
        active_generation.thinking_budget.is_inside_thinking(),
    );
    let structured_logits = structured_logit_bias_values
        .as_ref()
        .map(|logit_bias_values| {
            crate::gpu_token_sampling::add_token_logit_bias(
                &model.runtime,
                logits_after_repetition_penalty,
                logit_bias_values,
            )
        })
        .transpose()?;
    let sampling_logits = structured_logits
        .as_ref()
        .unwrap_or(logits_after_repetition_penalty);

    match sampling_plan(active_generation.request.sampling_strategy()) {
        // Greedy requests must bypass the categorical helper, whose positive
        // temperature precondition intentionally excludes temperature zero.
        SamplingPlan::HighestLogit => {
            model
                .argmax_last_token(sampling_logits)
                .map_err(|model_error| InferenceEngineError::Fatal {
                    reason: model_error.to_string(),
                })
        }
        SamplingPlan::TopKTopP {
            temperature_thousandths,
            top_k,
            top_p_thousandths,
        } => {
            let sampled_token = crate::gpu_token_sampling::build_sampled_token(
                &model.runtime,
                sampling_logits,
                temperature_thousandths,
                top_p_thousandths,
                Some(top_k),
                &mut active_generation.random_state,
            )?;
            sampled_token
                .item_u32()
                .map_err(|sampling_error| InferenceEngineError::Fatal {
                    reason: sampling_error.to_string(),
                })
        }
    }
}

pub(super) fn record_penalized_token_id(
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    token_id: u32,
) {
    if active_generation.request.repetition_penalty_thousandths() != 1_000
        && let Err(insertion_index) = active_generation
            .penalized_token_ids
            .binary_search(&token_id)
    {
        active_generation
            .penalized_token_ids
            .insert(insertion_index, token_id);
    }
}

fn sampling_plan(sampling_strategy: Qwen3_5SamplingStrategy) -> SamplingPlan {
    match sampling_strategy {
        Qwen3_5SamplingStrategy::HighestLogit => SamplingPlan::HighestLogit,
        Qwen3_5SamplingStrategy::TopKTopP {
            temperature_thousandths,
            top_k,
            top_p_thousandths,
            ..
        } => SamplingPlan::TopKTopP {
            temperature_thousandths,
            top_k,
            top_p_thousandths,
        },
    }
}

fn structured_logit_bias_values_if_visible(
    structured_generation: Option<&mut StructuredTokenConstraint>,
    is_inside_thinking: bool,
) -> Option<Vec<f32>> {
    if is_inside_thinking {
        None
    } else {
        structured_generation.map(StructuredTokenConstraint::logit_bias_values)
    }
}

#[cfg(test)]
mod tests {
    use astronomical_ipc_protocol::StructuredGenerationConstraint;

    use super::{SamplingPlan, sampling_plan, structured_logit_bias_values_if_visible};
    use crate::Qwen3_5SamplingStrategy;
    use crate::structured_generation::StructuredTokenConstraint;

    #[test]
    fn should_keep_highest_logit_sampling_out_of_the_temperature_pipeline() {
        assert_eq!(
            sampling_plan(Qwen3_5SamplingStrategy::HighestLogit),
            SamplingPlan::HighestLogit
        );
    }

    #[test]
    fn should_preserve_the_requested_top_k_and_top_p_limits() {
        assert_eq!(
            sampling_plan(Qwen3_5SamplingStrategy::TopKTopP {
                temperature_thousandths: 700,
                top_k: 37,
                top_p_thousandths: 920,
                seed: Some(11),
            }),
            SamplingPlan::TopKTopP {
                temperature_thousandths: 700,
                top_k: 37,
                top_p_thousandths: 920,
            }
        );
    }

    #[test]
    fn should_apply_structured_masks_only_after_reasoning() {
        let mut structured_generation = StructuredTokenConstraint::compile(
            &StructuredGenerationConstraint::Choice {
                choices: vec!["Juliet".to_owned()],
            },
            vec!["Juliet".to_owned(), "Romeo".to_owned()],
            vec![99],
            vec![vec![0]],
        )
        .expect("the fixture choice constraint should compile");

        assert!(
            structured_logit_bias_values_if_visible(Some(&mut structured_generation), true,)
                .is_none()
        );
        let visible_logit_bias_values =
            structured_logit_bias_values_if_visible(Some(&mut structured_generation), false)
                .expect("visible-token sampling should receive the structured mask");
        assert_eq!(visible_logit_bias_values, vec![0.0, f32::NEG_INFINITY]);
    }
}
