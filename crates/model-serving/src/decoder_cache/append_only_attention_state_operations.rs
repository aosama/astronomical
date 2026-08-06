use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use super::append_only_attention_state::{FULL_ATTENTION_OPERATION, STATE_DIMENSION_TOKEN_AXIS};

pub(super) fn build_updated_storage(
    runtime: &MlxRuntime,
    current_storage: Option<&MlxArray>,
    state_update: &MlxArray,
    previous_token_count: i32,
    full_attention_kv_state_growth_tokens: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let state_update_shape = state_update.shape();
    let update_token_count = state_update_shape[STATE_DIMENSION_TOKEN_AXIS];
    let next_token_count = previous_token_count
        .checked_add(update_token_count)
        .ok_or_else(|| full_attention_error("full-attention state token count overflowed"))?;
    let current_capacity_tokens =
        current_storage.map_or(0, |state| state.shape()[STATE_DIMENSION_TOKEN_AXIS]);

    let projected_capacity_tokens = projected_capacity_tokens(
        current_capacity_tokens,
        current_storage.is_some(),
        previous_token_count,
        update_token_count,
        full_attention_kv_state_growth_tokens,
    )?;
    let grown_state = if projected_capacity_tokens > current_capacity_tokens {
        // A partially used slab may contain unused tail capacity. Once an update
        // no longer fits, retain only the written prefix before appending a newly
        // rounded extension; otherwise the unused gap would become logical state.
        let retained_capacity_tokens = if current_storage.is_some()
            && previous_token_count % full_attention_kv_state_growth_tokens != 0
        {
            previous_token_count
        } else {
            current_capacity_tokens
        };
        let extension_capacity_tokens = projected_capacity_tokens
            .checked_sub(retained_capacity_tokens)
            .ok_or_else(|| full_attention_error("full-attention state growth underflowed"))?;
        let mut extension_shape = state_update_shape.clone();
        extension_shape[STATE_DIMENSION_TOKEN_AXIS] = extension_capacity_tokens;
        let state_extension = runtime.zeros(&extension_shape, state_update.dtype())?;
        match current_storage {
            Some(previous_state) => {
                let retained_prefix =
                    if previous_token_count % full_attention_kv_state_growth_tokens != 0 {
                        let mut retained_stops = previous_state.shape();
                        retained_stops[STATE_DIMENSION_TOKEN_AXIS] = previous_token_count;
                        Some(runtime.slice(
                            previous_state,
                            &[0, 0, 0, 0],
                            &retained_stops,
                            &[1, 1, 1, 1],
                        )?)
                    } else {
                        None
                    };
                let retained_state = retained_prefix.as_ref().unwrap_or(previous_state);
                Some(runtime.concatenate_axis(&[retained_state, &state_extension], 2)?)
            }
            None => Some(state_extension),
        }
    } else {
        None
    };
    let state_storage = grown_state
        .as_ref()
        .or(current_storage)
        .ok_or_else(|| full_attention_error("full-attention state storage is unavailable"))?;

    let mut update_starts = vec![0; state_update_shape.len()];
    update_starts[STATE_DIMENSION_TOKEN_AXIS] = previous_token_count;
    let mut update_stops = state_update_shape;
    update_stops[STATE_DIMENSION_TOKEN_AXIS] = next_token_count;
    let update_strides = vec![1; update_starts.len()];
    runtime.slice_update(
        state_storage,
        state_update,
        &update_starts,
        &update_stops,
        &update_strides,
    )
}

pub(super) fn projected_capacity_tokens(
    current_capacity_tokens: i32,
    has_current_storage: bool,
    previous_token_count: i32,
    update_token_count: i32,
    full_attention_kv_state_growth_tokens: i32,
) -> Result<i32, MlxRuntimeError> {
    let next_token_count = previous_token_count
        .checked_add(update_token_count)
        .ok_or_else(|| full_attention_error("full-attention state token count overflowed"))?;
    if next_token_count <= current_capacity_tokens {
        // The existing over-allocated slab already has enough unused room.
        return Ok(current_capacity_tokens);
    }
    // Round the incoming update, not the total context, to the configured growth
    // step. This mirrors the actual extension allocated by build_updated_storage.
    let rounded_update_tokens = update_token_count
        .checked_add(full_attention_kv_state_growth_tokens - 1)
        .and_then(|rounded_token_count| {
            rounded_token_count
                .checked_div(full_attention_kv_state_growth_tokens)
                .and_then(|growth_steps| {
                    growth_steps.checked_mul(full_attention_kv_state_growth_tokens)
                })
        })
        .ok_or_else(|| full_attention_error("full-attention state growth overflowed"))?;
    let retained_capacity_tokens = if has_current_storage
        && previous_token_count % full_attention_kv_state_growth_tokens != 0
    {
        previous_token_count
    } else {
        current_capacity_tokens
    };
    retained_capacity_tokens
        .checked_add(rounded_update_tokens)
        .ok_or_else(|| full_attention_error("full-attention state capacity overflowed"))
}

pub(super) fn active_view(
    runtime: &MlxRuntime,
    updated_state: &MlxArray,
    active_token_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let mut active_state_stops = updated_state.shape();
    active_state_stops[STATE_DIMENSION_TOKEN_AXIS] = active_token_count;
    runtime.slice(
        updated_state,
        &[0, 0, 0, 0],
        &active_state_stops,
        &[1, 1, 1, 1],
    )
}

pub(super) fn full_attention_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: FULL_ATTENTION_OPERATION,
        description: description.to_owned(),
    }
}
