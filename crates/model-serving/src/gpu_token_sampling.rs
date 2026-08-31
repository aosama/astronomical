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

pub(crate) fn masked_logits_after_top_k_and_top_p(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    vocabulary_size: i32,
    top_k: u16,
    top_p_thousandths: u16,
) -> Result<MlxArray, InferenceEngineError> {
    masked_logits_rows_after_top_k_and_top_p(
        runtime,
        logits,
        vocabulary_size,
        top_k,
        top_p_thousandths,
    )
}

fn masked_logits_rows_after_top_k_and_top_p(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    vocabulary_size: i32,
    top_k: u16,
    top_p_thousandths: u16,
) -> Result<MlxArray, InferenceEngineError> {
    if top_k == 0 {
        return Err(sampling_failure("sampled top-k must be positive"));
    }
    let row_count = logits
        .shape()
        .get(1)
        .copied()
        .ok_or_else(|| sampling_failure("masked logits rows require a rank-3 logits tensor"))?;
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
            &[1, row_count, top_k_i32],
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
        .broadcast_to(&negative_infinity, &[1, row_count, vocabulary_size])
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

/// Draws one Bernoulli acceptance coin per proposed token from its `min(1, p/q)`
/// acceptance probability using one random-key split.
///
/// `acceptance_probabilities` must have shape `[1, 1, draft_count]` with values in
/// `[0, 1]`. The returned `[1, draft_count]` array holds one per position, where
/// one means the drafted token is accepted. This preserves the sampling
/// distribution of the target because the acceptance coin is drawn from exactly
/// `min(1, p/q)` per position.
pub(crate) fn sample_acceptance_coins(
    runtime: &MlxRuntime,
    acceptance_probabilities: &MlxArray,
    random_state: &mut MlxArray,
    draft_count: usize,
) -> Result<MlxArray, InferenceEngineError> {
    let coin_shape = acceptance_probabilities.shape();
    if coin_shape
        != [
            1,
            1,
            i32::try_from(draft_count)
                .map_err(|_| sampling_failure("the draft count exceeds the MLX shape range"))?,
        ]
    {
        return Err(sampling_failure(
            "acceptance probabilities must have shape [1, 1, draft_count]",
        ));
    }
    let accept_column = reshape_to_coin_column(runtime, acceptance_probabilities, draft_count)?;
    let reject_column = {
        let one = runtime
            .array_from_f32(&[1.0], &[])
            .map_err(sampling_runtime_error)?;
        runtime
            .subtract(&one, &accept_column)
            .map_err(sampling_runtime_error)?
    };
    // MLX `categorical` samples with weights ∝ exp(logits), so the coin cases
    // must enter as true log-probabilities. With r in [0, 1], `ln(r)` equals
    // `log1p(r - 1)`: probability zero maps to -inf (never accepts) and one to
    // zero (always accepts).
    let one = runtime
        .array_from_f32(&[1.0], &[])
        .map_err(sampling_runtime_error)?;
    let accept_minus_one = runtime
        .subtract(&accept_column, &one)
        .map_err(sampling_runtime_error)?;
    let accept_case_logits = runtime
        .log1p(&accept_minus_one)
        .map_err(sampling_runtime_error)?;
    let reject_minus_one = runtime
        .subtract(&reject_column, &one)
        .map_err(sampling_runtime_error)?;
    let reject_case_logits = runtime
        .log1p(&reject_minus_one)
        .map_err(sampling_runtime_error)?;
    // Column order mirrors the coin contract: sampled index 0 means the
    // rejection case, index 1 means the drafted token is accepted.
    let coin_logits = runtime
        .concatenate_axis(&[&reject_case_logits, &accept_case_logits], -1)
        .map_err(sampling_runtime_error)?;
    let (next_random_state, coin_key) = runtime
        .split_random_key(random_state)
        .map_err(sampling_runtime_error)?;
    let coins = runtime
        .categorical_sample_with_key(&coin_logits, -1, &coin_key)
        .map_err(sampling_runtime_error)?;
    *random_state = next_random_state;
    Ok(coins)
}

fn reshape_to_coin_column(
    runtime: &MlxRuntime,
    acceptance_probabilities: &MlxArray,
    draft_count: usize,
) -> Result<MlxArray, InferenceEngineError> {
    let draft_count_i32 = i32::try_from(draft_count)
        .map_err(|_| sampling_failure("the draft count exceeds the MLX shape range"))?;
    runtime
        .reshape(acceptance_probabilities, &[1, draft_count_i32, 1])
        .map_err(sampling_runtime_error)
}

/// Samples one token index from relative probabilities over one vocabulary row.
///
/// This is the residual correction on rejected speculative acceptance: the
/// emitted token is drawn from mass proportional to `max(0, p − q)`, which
/// preserves the target distribution exactly. Zero entries cannot be emitted;
/// the caller must guarantee the residual row has positive mass.
pub(crate) fn sample_from_relative_probabilities(
    runtime: &MlxRuntime,
    relative_probabilities: &MlxArray,
    random_state: &mut MlxArray,
) -> Result<MlxArray, InferenceEngineError> {
    let probabilities_shape = relative_probabilities.shape();
    if probabilities_shape.len() != 3 || probabilities_shape[0] != 1 || probabilities_shape[1] != 1
    {
        return Err(sampling_failure(
            "residual probabilities must have shape [1, 1, vocabulary]",
        ));
    }
    let zero = runtime
        .array_from_f32(&[0.0], &[])
        .map_err(sampling_runtime_error)?;
    let positive_mask = runtime
        .greater(relative_probabilities, &zero)
        .map_err(sampling_runtime_error)?;
    // Weight ∝ exp(logits), so the residual rows must enter as true log
    // probabilities: with a single-token probability r in (0, 1], `ln(r)`
    // equals `log1p(r - 1)`, and zero mass maps to -inf so it can never emit.
    let one = runtime
        .array_from_f32(&[1.0], &[])
        .map_err(sampling_runtime_error)?;
    let relative_minus_one = runtime
        .subtract(relative_probabilities, &one)
        .map_err(sampling_runtime_error)?;
    let relative_log_probabilities = runtime
        .log1p(&relative_minus_one)
        .map_err(sampling_runtime_error)?;
    let negative_infinity_broadcast = runtime
        .broadcast_to(
            &runtime
                .array_from_f32(&[f32::NEG_INFINITY], &[])
                .map_err(sampling_runtime_error)?,
            &probabilities_shape,
        )
        .map_err(sampling_runtime_error)?;
    let residual_logits = runtime
        .where_select(
            &positive_mask,
            &relative_log_probabilities,
            &negative_infinity_broadcast,
        )
        .map_err(sampling_runtime_error)?;
    let (next_random_state, sample_key) = runtime
        .split_random_key(random_state)
        .map_err(sampling_runtime_error)?;
    let sampled_token = runtime
        .categorical_sample_with_key(&residual_logits, -1, &sample_key)
        .map_err(sampling_runtime_error)?;
    *random_state = next_random_state;
    Ok(sampled_token)
}

pub(crate) fn sample_categorical_after_temperature(
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
    if selected_shape.len() != 3 || selected_shape[0] != 1 || selected_shape[1] == 0 {
        return Err(sampling_failure(
            "selected probabilities must have shape [1, rows, top_k] with a positive row count",
        ));
    }
    let selected_row_count = selected_shape[1];
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
        .broadcast_to(&top_p_constant, &[1, selected_row_count, top_k])
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
        .broadcast_to(&negative_infinity, &[1, selected_row_count, top_k])
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
        .reshape(&range_indices, &[1, selected_row_count, top_k])
        .map_err(sampling_runtime_error)?;
    let zeros_template = runtime
        .array_from_i32(
            &vec![0_i32; selected_row_count as usize * top_k as usize],
            &[1, selected_row_count, top_k],
        )
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

/// Test-only re-export wrapper so the direct-MLX tests can pin the acceptance
/// coin distribution without widening the production sampling surface.
#[cfg(feature = "direct-mlx")]
pub fn sample_acceptance_coins_for_tests(
    runtime: &MlxRuntime,
    acceptance_probabilities: &MlxArray,
    random_state: &mut MlxArray,
    draft_count: usize,
) -> Result<MlxArray, InferenceEngineError> {
    sample_acceptance_coins(runtime, acceptance_probabilities, random_state, draft_count)
}

/// Test-only re-export wrapper for the residual-correction sampler.
#[cfg(feature = "direct-mlx")]
pub fn sample_from_relative_probabilities_for_tests(
    runtime: &MlxRuntime,
    relative_probabilities: &MlxArray,
    random_state: &mut MlxArray,
) -> Result<MlxArray, InferenceEngineError> {
    sample_from_relative_probabilities(runtime, relative_probabilities, random_state)
}

fn sampling_runtime_error(runtime_error: impl std::fmt::Display) -> InferenceEngineError {
    sampling_failure(runtime_error.to_string())
}

fn sampling_failure(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: reason.into(),
    }
}
