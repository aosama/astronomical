use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::InferenceEngineError;

use super::super::inference_execution::{fatal_engine_error, qwen3_5_moe_runtime_error};
use super::Qwen3_5MoEModel;

pub(in crate::qwen3_5_moe) fn random_state_for_seed(
    model: &Qwen3_5MoEModel,
    seed: u64,
) -> Result<MlxArray, InferenceEngineError> {
    model
        .runtime
        .random_key(seed)
        .map_err(qwen3_5_moe_runtime_error)
}

pub(in crate::qwen3_5_moe) fn validate_sampled_strategy(
    temperature_thousandths: u16,
    top_k: u16,
    top_p_thousandths: u16,
) -> Result<(), InferenceEngineError> {
    if temperature_thousandths == 0 || top_k == 0 || top_p_thousandths > 1_000 {
        return Err(fatal_engine_error(
            "sampled Qwen3.5-MoE strategy must use positive temperature, positive top-k, and top-p at most 1.0",
        ));
    }
    Ok(())
}

/// Builds one lazy Qwen3.5-MoE sample using a fully GPU-resident top-k + top-p pipeline.
///
/// The entire sampling graph — softmax, argpartition, top-p cumulative mask,
/// temperature scaling, and categorical sampling — remains lazy so generation
/// can submit it asynchronously before reading the sampled token on the host.
pub(in crate::qwen3_5_moe) fn build_qwen3_5_moe_sampled_token(
    model: &Qwen3_5MoEModel,
    final_logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
    random_state: &mut MlxArray,
) -> Result<MlxArray, InferenceEngineError> {
    validate_sampling_arguments(
        final_logits,
        temperature_thousandths,
        top_p_thousandths,
        top_k,
        model.config.vocabulary_size(),
    )?;

    let runtime = &model.runtime;
    let vocabulary_size = i32::try_from(model.config.vocabulary_size())
        .map_err(|_| fatal_engine_error("model vocabulary exceeds the MLX shape range"))?;
    let top_k_i32 = i32::from(top_k);

    // Step 1: Compute probabilities and select top-k indices (all lazy GPU ops).
    let probabilities = runtime
        .softmax_axis(final_logits, -1)
        .map_err(qwen3_5_moe_runtime_error)?;
    let negative_logits = runtime
        .negative(final_logits)
        .map_err(qwen3_5_moe_runtime_error)?;
    let partitioned_indices = runtime
        .argpartition_axis(&negative_logits, top_k_i32 - 1, -1)
        .map_err(qwen3_5_moe_runtime_error)?;
    let selected_indices = runtime
        .slice(
            &partitioned_indices,
            &[0, 0, 0],
            &[1, 1, top_k_i32],
            &[1, 1, 1],
        )
        .map_err(qwen3_5_moe_runtime_error)?;

    // Step 2: Gather the top-k logits and probabilities (lazy GPU ops).
    let selected_logits = runtime
        .take_along_axis(final_logits, &selected_indices, -1)
        .map_err(qwen3_5_moe_runtime_error)?;
    let selected_probabilities = runtime
        .take_along_axis(&probabilities, &selected_indices, -1)
        .map_err(qwen3_5_moe_runtime_error)?;

    // Step 3: Apply top-p mask entirely on the GPU.
    let masked_selected_logits = qwen3_5_moe_apply_top_p_mask(
        runtime,
        &selected_probabilities,
        &selected_logits,
        top_p_thousandths,
    )?;

    // Step 4: Scatter the masked logits back into the full vocabulary (lazy GPU op).
    let negative_infinity = runtime
        .array_from_f32(&[f32::NEG_INFINITY], &[])
        .map_err(qwen3_5_moe_runtime_error)?;
    let full_masked_logits = runtime
        .broadcast_to(&negative_infinity, &[1, 1, vocabulary_size])
        .map_err(qwen3_5_moe_runtime_error)?;
    let full_masked_logits = runtime
        .put_along_axis(
            &full_masked_logits,
            &selected_indices,
            &masked_selected_logits,
            -1,
        )
        .map_err(qwen3_5_moe_runtime_error)?;

    // Step 5: Scale by inverse temperature and sample (lazy GPU ops).
    let temperature = f32::from(temperature_thousandths) / 1_000.0;
    let scaled_logits = runtime
        .multiply_scalar(&full_masked_logits, temperature.recip())
        .map_err(qwen3_5_moe_runtime_error)?;
    let (next_random_state, sample_key) = runtime
        .split_random_key(random_state)
        .map_err(qwen3_5_moe_runtime_error)?;
    let sampled_token = runtime
        .categorical_sample_with_key(&scaled_logits, -1, &sample_key)
        .map_err(qwen3_5_moe_runtime_error)?;

    *random_state = next_random_state;
    Ok(sampled_token)
}

/// Applies the top-p (nucleus) mask entirely on the GPU using MLX graph operations.
///
/// This applies the standard top-p boundary while operating on pre-selected top-k
/// candidates. The algorithm:
///
/// 1. Sort probabilities in descending order.
/// 2. Compute the exclusive cumulative mass of more-probable candidates.
/// 3. Map the cumulative sums back to the original (unsorted) order.
/// 4. Keep a candidate only while the preceding mass is below top-p.
pub fn qwen3_5_moe_apply_top_p_mask(
    runtime: &MlxRuntime,
    selected_probabilities: &MlxArray,
    selected_logits: &MlxArray,
    top_p_thousandths: u16,
) -> Result<MlxArray, InferenceEngineError> {
    let selected_shape = selected_probabilities.shape();
    if selected_shape.len() != 3 || selected_shape[0] != 1 || selected_shape[1] != 1 {
        return Err(fatal_engine_error(
            "selected probabilities must have shape [1, 1, top_k]",
        ));
    }
    let top_k = selected_shape[2];
    if top_p_thousandths == 0 || top_p_thousandths >= 1_000 {
        // No top-p filtering needed. Use where_select with an always-true
        // condition to return a new lazy array that preserves the logits.
        let true_condition = runtime
            .array_from_f32(&[1.0], &[])
            .map_err(qwen3_5_moe_runtime_error)?;
        let true_broadcast = runtime
            .broadcast_to(&true_condition, &[1, 1, top_k])
            .map_err(qwen3_5_moe_runtime_error)?;
        let neg_inf = runtime
            .array_from_f32(&[f32::NEG_INFINITY], &[])
            .map_err(qwen3_5_moe_runtime_error)?;
        let neg_inf_broadcast = runtime
            .broadcast_to(&neg_inf, &[1, 1, top_k])
            .map_err(qwen3_5_moe_runtime_error)?;
        return runtime
            .where_select(&true_broadcast, selected_logits, &neg_inf_broadcast)
            .map_err(qwen3_5_moe_runtime_error);
    }

    let top_p = f32::from(top_p_thousandths) / 1_000.0;

    // Sort probabilities in descending order by negating and argsort(ascending).
    let negated_probabilities = runtime
        .negative(selected_probabilities)
        .map_err(qwen3_5_moe_runtime_error)?;
    let sorted_indices = runtime
        .argsort_axis(&negated_probabilities, -1)
        .map_err(qwen3_5_moe_runtime_error)?;
    // Cast indices to int32 to ensure compatibility with put_along_axis.
    let sorted_indices_i32 = runtime
        .astype(
            &sorted_indices,
            astronomical_runtime_integration::MlxDtype::Int32,
        )
        .map_err(qwen3_5_moe_runtime_error)?;
    let sorted_probabilities = runtime
        .take_along_axis(selected_probabilities, &sorted_indices_i32, -1)
        .map_err(qwen3_5_moe_runtime_error)?;

    // A candidate survives while the probability mass of all more-probable
    // candidates is still below top-p. The exclusive scan keeps the candidate
    // that crosses the threshold in ascending-tail order.
    let preceding_probability_mass = runtime
        .cumsum(&sorted_probabilities, -1, false, false)
        .map_err(qwen3_5_moe_runtime_error)?;

    // Build the mask in sorted order: keep where preceding mass < top-p.
    let top_p_constant = runtime
        .array_from_f32(&[top_p], &[])
        .map_err(qwen3_5_moe_runtime_error)?;
    let top_p_broadcast = runtime
        .broadcast_to(&top_p_constant, &[1, 1, top_k])
        .map_err(qwen3_5_moe_runtime_error)?;
    let keep_mask_sorted = runtime
        .greater(&top_p_broadcast, &preceding_probability_mass)
        .map_err(qwen3_5_moe_runtime_error)?;

    // Gather logits in sorted order and apply the mask.
    let sorted_logits = runtime
        .take_along_axis(selected_logits, &sorted_indices_i32, -1)
        .map_err(qwen3_5_moe_runtime_error)?;
    let negative_infinity = runtime
        .array_from_f32(&[f32::NEG_INFINITY], &[])
        .map_err(qwen3_5_moe_runtime_error)?;
    let neg_inf_broadcast = runtime
        .broadcast_to(&negative_infinity, &[1, 1, top_k])
        .map_err(qwen3_5_moe_runtime_error)?;
    let masked_sorted_logits = runtime
        .where_select(&keep_mask_sorted, &sorted_logits, &neg_inf_broadcast)
        .map_err(qwen3_5_moe_runtime_error)?;

    // Scatter back to original order using inverse permutation.
    // inverse_indices[sorted_indices[i]] = i
    let range_indices = runtime
        .arange_i32(0, top_k)
        .map_err(qwen3_5_moe_runtime_error)?;
    let range_indices_reshaped = runtime
        .reshape(&range_indices, &[1, 1, top_k])
        .map_err(qwen3_5_moe_runtime_error)?;
    let zeros_template = runtime
        .array_from_i32(&vec![0_i32; top_k as usize], &[1, 1, top_k])
        .map_err(qwen3_5_moe_runtime_error)?;
    let inverse_indices = runtime
        .put_along_axis(
            &zeros_template,
            &sorted_indices_i32,
            &range_indices_reshaped,
            -1,
        )
        .map_err(qwen3_5_moe_runtime_error)?;
    let masked_logits = runtime
        .take_along_axis(&masked_sorted_logits, &inverse_indices, -1)
        .map_err(qwen3_5_moe_runtime_error)?;

    Ok(masked_logits)
}

fn validate_sampling_arguments(
    final_logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
    vocabulary_size: u32,
) -> Result<(), InferenceEngineError> {
    let vocabulary_size = i32::try_from(vocabulary_size)
        .map_err(|_| fatal_engine_error("model vocabulary exceeds the MLX shape range"))?;
    if final_logits.shape() != [1, 1, vocabulary_size] {
        return Err(fatal_engine_error(format!(
            "sampled Qwen3.5-MoE logits must have shape [1, 1, {vocabulary_size}]"
        )));
    }
    if temperature_thousandths == 0 {
        return Err(fatal_engine_error(
            "sampled Qwen3.5-MoE temperature must be positive",
        ));
    }
    if top_p_thousandths > 1_000 {
        return Err(fatal_engine_error(
            "sampled Qwen3.5-MoE top-p must not exceed 1.0",
        ));
    }
    if top_k == 0 {
        return Err(fatal_engine_error(
            "sampled Qwen3.5-MoE top-k must be positive",
        ));
    }
    Ok(())
}
