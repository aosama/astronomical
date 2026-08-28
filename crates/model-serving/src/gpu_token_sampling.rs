//! Family-neutral GPU token sampling.
//!
//! One MLX graph — softmax, optional top-k, optional nucleus, temperature, and
//! `categorical` — is shared so a family cannot resolve a sampling policy and then
//! silently argmax. The ops match the already-proven Qwen3.5 pipeline rather than a
//! host-side sampler.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::InferenceEngineError;

/// Builds one lazy sampled token using a fully GPU-resident top-k + top-p pipeline.
///
/// `logits` must have shape `[1, 1, vocabulary]`. `top_k = None` means no k-truncation:
/// with `top_p >= 1.0` this is temperature scaling plus `categorical` over the full row,
/// which is the published Laguna 6-bit generation_config (temperature 1.0, top_p 1.0,
/// omitted top_k).
pub(crate) fn build_sampled_token(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: Option<u16>,
    random_state: &mut MlxArray,
) -> Result<MlxArray, InferenceEngineError> {
    let vocabulary_size =
        validate_sampled_logits(logits, temperature_thousandths, top_p_thousandths)?;
    let constrained_logits = match top_k {
        Some(top_k) => Some(masked_logits_after_top_k_and_top_p(
            runtime,
            logits,
            vocabulary_size,
            top_k,
            top_p_thousandths,
        )?),
        None if top_p_thousandths > 0 && top_p_thousandths < 1_000 => Some(apply_top_p_mask(
            runtime,
            &softmax_last_axis(runtime, logits)?,
            logits,
            top_p_thousandths,
        )?),
        None => None,
    };
    sample_categorical_after_temperature(
        runtime,
        constrained_logits.as_ref().unwrap_or(logits),
        temperature_thousandths,
        random_state,
    )
}

fn validate_sampled_logits(
    logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
) -> Result<i32, InferenceEngineError> {
    let logit_shape = logits.shape();
    if logit_shape.len() != 3 || logit_shape[0] != 1 || logit_shape[1] != 1 || logit_shape[2] <= 0 {
        return Err(sampling_failure(
            "sampled logits must have shape [1, 1, vocabulary] with a positive vocabulary",
        ));
    }
    if temperature_thousandths == 0 {
        return Err(sampling_failure("sampled temperature must be positive"));
    }
    if top_p_thousandths > 1_000 {
        return Err(sampling_failure("sampled top-p must not exceed 1.0"));
    }
    Ok(logit_shape[2])
}

fn masked_logits_after_top_k_and_top_p(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    vocabulary_size: i32,
    top_k: u16,
    top_p_thousandths: u16,
) -> Result<MlxArray, InferenceEngineError> {
    if top_k == 0 {
        return Err(sampling_failure("sampled top-k must be positive"));
    }
    let top_k_i32 = i32::from(top_k).min(vocabulary_size);
    let probabilities = softmax_last_axis(runtime, logits)?;
    let negative_logits = runtime.negative(logits).map_err(sampling_runtime_error)?;
    let partitioned_indices = runtime
        .argpartition_axis(&negative_logits, top_k_i32 - 1, -1)
        .map_err(sampling_runtime_error)?;
    let selected_indices = runtime
        .slice(
            &partitioned_indices,
            &[0, 0, 0],
            &[1, 1, top_k_i32],
            &[1, 1, 1],
        )
        .map_err(sampling_runtime_error)?;
    let selected_logits = runtime
        .take_along_axis(logits, &selected_indices, -1)
        .map_err(sampling_runtime_error)?;
    let selected_probabilities = runtime
        .take_along_axis(&probabilities, &selected_indices, -1)
        .map_err(sampling_runtime_error)?;
    let masked_selected_logits = apply_top_p_mask(
        runtime,
        &selected_probabilities,
        &selected_logits,
        top_p_thousandths,
    )?;
    let negative_infinity = runtime
        .array_from_f32(&[f32::NEG_INFINITY], &[])
        .map_err(sampling_runtime_error)?;
    let full_masked_logits = runtime
        .broadcast_to(&negative_infinity, &[1, 1, vocabulary_size])
        .map_err(sampling_runtime_error)?;
    runtime
        .put_along_axis(
            &full_masked_logits,
            &selected_indices,
            &masked_selected_logits,
            -1,
        )
        .map_err(sampling_runtime_error)
}

fn sample_categorical_after_temperature(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    temperature_thousandths: u16,
    random_state: &mut MlxArray,
) -> Result<MlxArray, InferenceEngineError> {
    let temperature = f32::from(temperature_thousandths) / 1_000.0;
    let scaled_logits = runtime
        .multiply_scalar(logits, temperature.recip())
        .map_err(sampling_runtime_error)?;
    let (next_random_state, sample_key) = runtime
        .split_random_key(random_state)
        .map_err(sampling_runtime_error)?;
    let sampled_token = runtime
        .categorical_sample_with_key(&scaled_logits, -1, &sample_key)
        .map_err(sampling_runtime_error)?;
    *random_state = next_random_state;
    Ok(sampled_token)
}

fn softmax_last_axis(
    runtime: &MlxRuntime,
    logits: &MlxArray,
) -> Result<MlxArray, InferenceEngineError> {
    runtime
        .softmax_axis(logits, -1)
        .map_err(sampling_runtime_error)
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
pub fn apply_top_p_mask(
    runtime: &MlxRuntime,
    selected_probabilities: &MlxArray,
    selected_logits: &MlxArray,
    top_p_thousandths: u16,
) -> Result<MlxArray, InferenceEngineError> {
    let selected_shape = selected_probabilities.shape();
    if selected_shape.len() != 3 || selected_shape[0] != 1 || selected_shape[1] != 1 {
        return Err(sampling_failure(
            "selected probabilities must have shape [1, 1, top_k]",
        ));
    }
    let top_k = selected_shape[2];
    if top_p_thousandths == 0 || top_p_thousandths >= 1_000 {
        // No top-p filtering needed. Use where_select with an always-true
        // condition to return a new lazy array that preserves the logits.
        let true_condition = runtime
            .array_from_f32(&[1.0], &[])
            .map_err(sampling_runtime_error)?;
        let true_broadcast = runtime
            .broadcast_to(&true_condition, &[1, 1, top_k])
            .map_err(sampling_runtime_error)?;
        let neg_inf = runtime
            .array_from_f32(&[f32::NEG_INFINITY], &[])
            .map_err(sampling_runtime_error)?;
        let neg_inf_broadcast = runtime
            .broadcast_to(&neg_inf, &[1, 1, top_k])
            .map_err(sampling_runtime_error)?;
        return runtime
            .where_select(&true_broadcast, selected_logits, &neg_inf_broadcast)
            .map_err(sampling_runtime_error);
    }

    let top_p = f32::from(top_p_thousandths) / 1_000.0;

    // Sort probabilities in descending order by negating and argsort(ascending).
    let negated_probabilities = runtime
        .negative(selected_probabilities)
        .map_err(sampling_runtime_error)?;
    let sorted_indices = runtime
        .argsort_axis(&negated_probabilities, -1)
        .map_err(sampling_runtime_error)?;
    // Cast indices to int32 to ensure compatibility with put_along_axis.
    let sorted_indices_i32 = runtime
        .astype(
            &sorted_indices,
            astronomical_runtime_integration::MlxDtype::Int32,
        )
        .map_err(sampling_runtime_error)?;
    let sorted_probabilities = runtime
        .take_along_axis(selected_probabilities, &sorted_indices_i32, -1)
        .map_err(sampling_runtime_error)?;

    // A candidate survives while the probability mass of all more-probable
    // candidates is still below top-p. The exclusive scan keeps the candidate
    // that crosses the threshold in ascending-tail order.
    let preceding_probability_mass = runtime
        .cumsum(&sorted_probabilities, -1, false, false)
        .map_err(sampling_runtime_error)?;

    // Build the mask in sorted order: keep where preceding mass < top-p.
    let top_p_constant = runtime
        .array_from_f32(&[top_p], &[])
        .map_err(sampling_runtime_error)?;
    let top_p_broadcast = runtime
        .broadcast_to(&top_p_constant, &[1, 1, top_k])
        .map_err(sampling_runtime_error)?;
    let keep_mask_sorted = runtime
        .greater(&top_p_broadcast, &preceding_probability_mass)
        .map_err(sampling_runtime_error)?;

    // Gather logits in sorted order and apply the mask.
    let sorted_logits = runtime
        .take_along_axis(selected_logits, &sorted_indices_i32, -1)
        .map_err(sampling_runtime_error)?;
    let negative_infinity = runtime
        .array_from_f32(&[f32::NEG_INFINITY], &[])
        .map_err(sampling_runtime_error)?;
    let neg_inf_broadcast = runtime
        .broadcast_to(&negative_infinity, &[1, 1, top_k])
        .map_err(sampling_runtime_error)?;
    let masked_sorted_logits = runtime
        .where_select(&keep_mask_sorted, &sorted_logits, &neg_inf_broadcast)
        .map_err(sampling_runtime_error)?;

    // Scatter back to original order using inverse permutation.
    // inverse_indices[sorted_indices[i]] = i
    let range_indices = runtime
        .arange_i32(0, top_k)
        .map_err(sampling_runtime_error)?;
    let range_indices_reshaped = runtime
        .reshape(&range_indices, &[1, 1, top_k])
        .map_err(sampling_runtime_error)?;
    let zeros_template = runtime
        .array_from_i32(&vec![0_i32; top_k as usize], &[1, 1, top_k])
        .map_err(sampling_runtime_error)?;
    let inverse_indices = runtime
        .put_along_axis(
            &zeros_template,
            &sorted_indices_i32,
            &range_indices_reshaped,
            -1,
        )
        .map_err(sampling_runtime_error)?;
    runtime
        .take_along_axis(&masked_sorted_logits, &inverse_indices, -1)
        .map_err(sampling_runtime_error)
}

fn sampling_runtime_error(runtime_error: impl std::fmt::Display) -> InferenceEngineError {
    sampling_failure(runtime_error.to_string())
}

fn sampling_failure(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: reason.into(),
    }
}
