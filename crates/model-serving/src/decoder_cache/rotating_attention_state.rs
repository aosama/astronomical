//! Bounded rotating key/value state with absolute position separate from ring index.
//!
//! Multi-token prefill exposes at most `window + chunk - 1` chronological tokens,
//! then commits only the final window. One-token decode updates one physical slot.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

const TOKEN_AXIS: usize = 2;
const OPERATION: &str = "update the in-memory rotating key/value state";

/// Bounded rotating attention keys and values.
pub struct RotatingKeyValueState {
    keys: Option<MlxArray>,
    values: Option<MlxArray>,
    window_size: i32,
    absolute_position: i32,
    ring_write_index: i32,
}

/// Physical retry checkpoint for one rotating update.
pub struct RotatingKeyValueStateAllocationCheckpoint {
    keys: Option<MlxArray>,
    values: Option<MlxArray>,
    window_size: i32,
    absolute_position: i32,
    ring_write_index: i32,
}

type RotatingUpdate = (MlxArray, MlxArray, MlxArray, MlxArray, i32);

impl RotatingKeyValueState {
    /// Creates empty rotating state for a positive caller-selected window.
    pub fn empty(window_size: i32) -> Result<Self, MlxRuntimeError> {
        if window_size <= 0 {
            return Err(rotating_error("rotating window size must be positive"));
        }
        Ok(Self {
            keys: None,
            values: None,
            window_size,
            absolute_position: 0,
            ring_write_index: 0,
        })
    }

    /// Returns the absolute token count written so far; this is the RoPE offset.
    #[must_use]
    pub const fn absolute_position(&self) -> i32 {
        self.absolute_position
    }

    /// Returns the next physical ring write slot.
    #[must_use]
    pub const fn ring_write_index(&self) -> i32 {
        self.ring_write_index
    }

    /// Returns how many tokens remain after the last commit.
    #[must_use]
    pub fn committed_token_count(&self) -> i32 {
        self.absolute_position.min(self.window_size)
    }

    /// Returns the caller-selected window.
    #[must_use]
    pub const fn window_size(&self) -> i32 {
        self.window_size
    }

    /// Returns the live key slab when one has been written.
    #[must_use]
    pub fn keys(&self) -> Option<&MlxArray> {
        self.keys.as_ref()
    }

    /// Returns the live value slab when one has been written.
    #[must_use]
    pub fn values(&self) -> Option<&MlxArray> {
        self.values.as_ref()
    }

    /// Returns logical bytes owned by retained key/value slabs.
    #[must_use]
    pub fn payload_byte_count(&self) -> u64 {
        self.keys
            .as_ref()
            .into_iter()
            .chain(self.values.as_ref())
            .map(|state| state.byte_count() as u64)
            .sum()
    }

    /// Appends new key/value rows and returns the complete attention view.
    pub fn update_and_fetch(
        &mut self,
        runtime: &MlxRuntime,
        new_keys: &MlxArray,
        new_values: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        performance_attribution
            .measure_operation(PerformanceOperation::RotatingKeyValueStateUpdate, |_| {
                self.update_and_fetch_inner(runtime, new_keys, new_values)
            })
    }

    fn update_and_fetch_inner(
        &mut self,
        runtime: &MlxRuntime,
        new_keys: &MlxArray,
        new_values: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        let new_token_count = validate_rotating_update(new_keys, new_values)?;
        let next_absolute_position = self
            .absolute_position
            .checked_add(new_token_count)
            .ok_or_else(|| rotating_error("rotating absolute position overflowed"))?;
        let (attention_keys, attention_values, committed_keys, committed_values, write_index) =
            if new_token_count == 1 {
                self.update_one_token(runtime, new_keys, new_values)?
            } else {
                self.update_multi_token(runtime, new_keys, new_values, new_token_count)?
            };
        self.keys = Some(committed_keys.retain()?);
        self.values = Some(committed_values.retain()?);
        self.absolute_position = next_absolute_position;
        self.ring_write_index = write_index;
        Ok((attention_keys, attention_values))
    }

    fn update_one_token(
        &self,
        runtime: &MlxRuntime,
        new_keys: &MlxArray,
        new_values: &MlxArray,
    ) -> Result<RotatingUpdate, MlxRuntimeError> {
        if self.absolute_position < self.window_size {
            return self.append_growing(runtime, new_keys, new_values);
        }
        let stored_keys = self
            .keys
            .as_ref()
            .ok_or_else(|| rotating_error("rotating keys must exist once the window is full"))?;
        let stored_values = self
            .values
            .as_ref()
            .ok_or_else(|| rotating_error("rotating values must exist once the window is full"))?;
        let write_index = if self.ring_write_index == self.window_size {
            0
        } else {
            self.ring_write_index
        };
        let next_keys = write_token_slot(runtime, stored_keys, new_keys, write_index)?;
        let next_values = write_token_slot(runtime, stored_values, new_values, write_index)?;
        Ok((
            next_keys.retain()?,
            next_values.retain()?,
            next_keys,
            next_values,
            write_index + 1,
        ))
    }

    fn update_multi_token(
        &self,
        runtime: &MlxRuntime,
        new_keys: &MlxArray,
        new_values: &MlxArray,
        new_token_count: i32,
    ) -> Result<RotatingUpdate, MlxRuntimeError> {
        let chronological_keys = self.chronological_append(runtime, self.keys(), new_keys)?;
        let chronological_values = self.chronological_append(runtime, self.values(), new_values)?;
        let maximum_attention_tokens = self
            .window_size
            .checked_add(new_token_count)
            .and_then(|window_plus_chunk| window_plus_chunk.checked_sub(1))
            .ok_or_else(|| rotating_error("rotating prefill transient overflowed"))?;
        let attention_token_count = token_count(&chronological_keys)?.min(maximum_attention_tokens);
        let attention_keys = take_last_tokens(runtime, &chronological_keys, attention_token_count)?;
        let attention_values =
            take_last_tokens(runtime, &chronological_values, attention_token_count)?;
        let committed_token_count = attention_token_count.min(self.window_size);
        let committed_keys = take_last_tokens(runtime, &attention_keys, committed_token_count)?;
        let committed_values = take_last_tokens(runtime, &attention_values, committed_token_count)?;
        Ok((
            attention_keys,
            attention_values,
            committed_keys,
            committed_values,
            committed_token_count,
        ))
    }

    fn chronological_append(
        &self,
        runtime: &MlxRuntime,
        stored: Option<&MlxArray>,
        new_tokens: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        match stored {
            Some(stored_tokens) => concatenate_tokens(
                runtime,
                &temporal_order(
                    runtime,
                    stored_tokens,
                    self.ring_write_index,
                    self.absolute_position,
                )?,
                new_tokens,
            ),
            None => new_tokens.retain(),
        }
    }

    fn append_growing(
        &self,
        runtime: &MlxRuntime,
        new_keys: &MlxArray,
        new_values: &MlxArray,
    ) -> Result<RotatingUpdate, MlxRuntimeError> {
        let next_keys = append_optional(runtime, self.keys(), new_keys)?;
        let next_values = append_optional(runtime, self.values(), new_values)?;
        let next_token_count = token_count(&next_keys)?;
        Ok((
            next_keys.retain()?,
            next_values.retain()?,
            next_keys,
            next_values,
            next_token_count,
        ))
    }

    /// Retains physical owners and counters for a retry.
    pub fn allocation_checkpoint(
        &self,
    ) -> Result<RotatingKeyValueStateAllocationCheckpoint, MlxRuntimeError> {
        validate_paired_owners(self.keys(), self.values(), "rotating state")?;
        Ok(RotatingKeyValueStateAllocationCheckpoint {
            keys: self.keys.as_ref().map(MlxArray::retain).transpose()?,
            values: self.values.as_ref().map(MlxArray::retain).transpose()?,
            window_size: self.window_size,
            absolute_position: self.absolute_position,
            ring_write_index: self.ring_write_index,
        })
    }

    /// Restores physical owners and counters from a retry checkpoint.
    pub fn restore_allocation_checkpoint(
        &mut self,
        checkpoint: RotatingKeyValueStateAllocationCheckpoint,
    ) -> Result<(), MlxRuntimeError> {
        validate_paired_owners(
            checkpoint.keys.as_ref(),
            checkpoint.values.as_ref(),
            "checkpoint",
        )?;
        self.keys = checkpoint.keys;
        self.values = checkpoint.values;
        self.window_size = checkpoint.window_size;
        self.absolute_position = checkpoint.absolute_position;
        self.ring_write_index = checkpoint.ring_write_index;
        Ok(())
    }

    /// Replaces state from a validated persistent prompt-cache boundary.
    pub fn restore_from_blocks(
        &mut self,
        restored_keys: MlxArray,
        restored_values: MlxArray,
        absolute_position: i32,
        ring_write_index: i32,
    ) -> Result<(), MlxRuntimeError> {
        let key_shape = restored_keys.shape();
        if key_shape.len() != 4
            || key_shape != restored_values.shape()
            || key_shape[TOKEN_AXIS] <= 0
        {
            return Err(rotating_error(
                "restored rotating slabs must have identical rank-four nonempty shapes",
            ));
        }
        if absolute_position < 0 || ring_write_index < 0 || ring_write_index > self.window_size {
            return Err(rotating_error(
                "restored rotating counters must fit the committed window",
            ));
        }
        self.keys = Some(restored_keys);
        self.values = Some(restored_values);
        self.absolute_position = absolute_position;
        self.ring_write_index = ring_write_index;
        Ok(())
    }
}

fn validate_rotating_update(
    new_keys: &MlxArray,
    new_values: &MlxArray,
) -> Result<i32, MlxRuntimeError> {
    let key_shape = new_keys.shape();
    if key_shape != new_values.shape() || key_shape.len() != 4 {
        return Err(rotating_error(
            "rotating keys and values must have identical rank-four shapes",
        ));
    }
    let new_token_count = key_shape[TOKEN_AXIS];
    if new_token_count <= 0 {
        return Err(rotating_error(
            "rotating update token count must be positive",
        ));
    }
    Ok(new_token_count)
}

fn validate_paired_owners(
    keys: Option<&MlxArray>,
    values: Option<&MlxArray>,
    owner_description: &'static str,
) -> Result<(), MlxRuntimeError> {
    if keys.is_some() != values.is_some() {
        return Err(rotating_error(match owner_description {
            "checkpoint" => "rotating checkpoint keys and values must both be present or absent",
            _ => "rotating keys and values must both be present or absent",
        }));
    }
    Ok(())
}

fn temporal_order(
    runtime: &MlxRuntime,
    stored: &MlxArray,
    ring_write_index: i32,
    absolute_position: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let stored_token_count = token_count(stored)?;
    if ring_write_index == stored_token_count || absolute_position <= stored_token_count {
        return slice_tokens(
            runtime,
            stored,
            0,
            stored_token_count.min(absolute_position),
        );
    }
    let newest = slice_tokens(runtime, stored, ring_write_index, stored_token_count)?;
    let oldest = slice_tokens(runtime, stored, 0, ring_write_index)?;
    concatenate_tokens(runtime, &newest, &oldest)
}

fn write_token_slot(
    runtime: &MlxRuntime,
    stored: &MlxArray,
    new_tokens: &MlxArray,
    ring_write_index: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = stored.shape();
    runtime.slice_update(
        stored,
        new_tokens,
        &[0, 0, ring_write_index, 0],
        &[shape[0], shape[1], ring_write_index + 1, shape[3]],
        &[1, 1, 1, 1],
    )
}

fn append_optional(
    runtime: &MlxRuntime,
    stored: Option<&MlxArray>,
    new_tokens: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    match stored {
        Some(stored_tokens) => concatenate_tokens(runtime, stored_tokens, new_tokens),
        None => new_tokens.retain(),
    }
}

fn concatenate_tokens(
    runtime: &MlxRuntime,
    left: &MlxArray,
    right: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    runtime.concatenate_axis(&[left, right], TOKEN_AXIS as i32)
}

fn take_last_tokens(
    runtime: &MlxRuntime,
    tokens: &MlxArray,
    keep_token_count: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let available_token_count = token_count(tokens)?;
    if keep_token_count >= available_token_count {
        return tokens.retain();
    }
    slice_tokens(
        runtime,
        tokens,
        available_token_count - keep_token_count,
        available_token_count,
    )
}

fn slice_tokens(
    runtime: &MlxRuntime,
    tokens: &MlxArray,
    start_token: i32,
    stop_token: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = tokens.shape();
    runtime.slice(
        tokens,
        &[0, 0, start_token, 0],
        &[shape[0], shape[1], stop_token, shape[3]],
        &[1, 1, 1, 1],
    )
}

fn token_count(tokens: &MlxArray) -> Result<i32, MlxRuntimeError> {
    let shape = tokens.shape();
    if shape.len() != 4 {
        return Err(rotating_error("rotating tensors must have rank four"));
    }
    Ok(shape[TOKEN_AXIS])
}

fn rotating_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: OPERATION,
        description: description.to_owned(),
    }
}
