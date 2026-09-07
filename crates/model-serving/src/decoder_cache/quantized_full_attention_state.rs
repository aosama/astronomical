//! Quantized append-only full-attention KV state.
//!
//! Stores keys and values as affine-quantized slabs `(packed, scales, biases)`
//! so long-context decode attention reads one bit-width of KV instead of
//! bfloat16. The append, growth, and active-view mechanics mirror
//! `FullAttentionKeyValueState`; only the tensor payload differs.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use super::append_only_attention_state::STATE_DIMENSION_TOKEN_AXIS;
use super::append_only_attention_state_operations::{active_view, build_updated_storage};

const QUANTIZED_STATE_OPERATION: &str = "update the in-memory quantized full-attention KV state";

/// One packed, scaled, and biased slab triple for one attention tensor.
#[derive(Debug)]
struct QuantizedSlab {
    packed: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

/// Owned quantized views over the written KV prefix for one update.
#[derive(Debug)]
pub struct QuantizedTensorViews {
    pub packed: MlxArray,
    pub scales: MlxArray,
    pub biases: MlxArray,
}

/// Owned quantized views for one layer's keys and values.
#[derive(Debug)]
pub struct QuantizedKeyValueViews {
    pub keys: QuantizedTensorViews,
    pub values: QuantizedTensorViews,
}

/// Single owner for one full-attention layer's quantized keys and values.
#[derive(Debug)]
pub struct QuantizedFullAttentionKeyValueState {
    keys: Option<QuantizedSlab>,
    values: Option<QuantizedSlab>,
    offset_tokens: i32,
    full_attention_kv_state_growth_tokens: i32,
    group_size: i32,
    bits: i32,
}

impl QuantizedFullAttentionKeyValueState {
    pub fn empty_with_growth_tokens(
        full_attention_kv_state_growth_tokens: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, MlxRuntimeError> {
        if full_attention_kv_state_growth_tokens <= 0 {
            return Err(quantized_state_error(
                "quantized KV-state growth tokens must be positive",
            ));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(quantized_state_error(
                "quantized KV-state group size must be 32, 64, or 128",
            ));
        }
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(quantized_state_error(
                "quantized KV-state bit width must be 2, 3, 4, 5, 6, or 8",
            ));
        }
        Ok(Self {
            keys: None,
            values: None,
            offset_tokens: 0,
            full_attention_kv_state_growth_tokens,
            group_size,
            bits,
        })
    }

    /// Returns the number of tokens written into the quantized slabs so far.
    #[must_use]
    pub fn offset_tokens(&self) -> i32 {
        self.offset_tokens
    }

    /// Read-only access to the packed keys slab for restore evaluation.
    #[must_use]
    pub fn quantized_keys_state(&self) -> Option<&MlxArray> {
        self.keys.as_ref().map(|slab| &slab.packed)
    }

    /// Read-only access to the packed values slab for restore evaluation.
    #[must_use]
    pub fn quantized_values_state(&self) -> Option<&MlxArray> {
        self.values.as_ref().map(|slab| &slab.packed)
    }

    #[must_use]
    pub const fn group_size(&self) -> i32 {
        self.group_size
    }

    #[must_use]
    pub const fn bits(&self) -> i32 {
        self.bits
    }

    /// Quantizes the update, writes it into the slabs, and returns views over
    /// the written prefix for the quantized attention passes.
    pub fn update_and_fetch(
        &mut self,
        runtime: &MlxRuntime,
        new_keys: &MlxArray,
        new_values: &MlxArray,
        previous_token_count: i32,
    ) -> Result<QuantizedKeyValueViews, MlxRuntimeError> {
        let (quantized_keys, quantized_values) = validate_and_quantize_update(
            self,
            runtime,
            new_keys,
            new_values,
            previous_token_count,
        )?;
        let update_token_count = new_keys.shape()[STATE_DIMENSION_TOKEN_AXIS];
        let next_token_count = previous_token_count
            .checked_add(update_token_count)
            .ok_or_else(|| quantized_state_error("quantized KV offset overflowed"))?;

        let next_keys = grow_slab(
            runtime,
            self.keys.as_ref(),
            quantized_keys,
            previous_token_count,
            self.full_attention_kv_state_growth_tokens,
        )?;
        let next_values = grow_slab(
            runtime,
            self.values.as_ref(),
            quantized_values,
            previous_token_count,
            self.full_attention_kv_state_growth_tokens,
        )?;

        let active_keys = active_slab_views(runtime, &next_keys, next_token_count)?;
        let active_values = active_slab_views(runtime, &next_values, next_token_count)?;

        self.keys = Some(next_keys);
        self.values = Some(next_values);
        self.offset_tokens = next_token_count;
        Ok(QuantizedKeyValueViews {
            keys: active_keys,
            values: active_values,
        })
    }

    /// Logical payload bytes owned by the packed, scale, and bias slabs.
    #[must_use]
    pub fn payload_byte_count(&self) -> u64 {
        self.keys
            .iter()
            .chain(self.values.iter())
            .flat_map(|slab| {
                [
                    u64::try_from(slab.packed.byte_count()).unwrap_or(u64::MAX),
                    u64::try_from(slab.scales.byte_count()).unwrap_or(u64::MAX),
                    u64::try_from(slab.biases.byte_count()).unwrap_or(u64::MAX),
                ]
            })
            .fold(0_u64, u64::saturating_add)
    }

    /// Replaces the storage from a restored bfloat16 prefix and reserves one
    /// growth step of capacity so the first update after restore does not
    /// copy the whole restored prefix. The prompt-cache SSD format is
    /// bfloat16, so a quantized state re-quantizes on restore.
    pub fn restore_from_bf16_blocks_with_growth_headroom(
        &mut self,
        runtime: &MlxRuntime,
        restored_keys: MlxArray,
        restored_values: MlxArray,
    ) -> Result<(), MlxRuntimeError> {
        let restored_keys_shape = restored_keys.shape();
        if restored_keys_shape.len() != 4
            || restored_keys_shape != restored_values.shape()
            || restored_keys_shape[STATE_DIMENSION_TOKEN_AXIS] <= 0
        {
            return Err(quantized_state_error(
                "restored K and V slabs must have identical rank-four nonempty shapes",
            ));
        }
        let restored_token_count = restored_keys_shape[STATE_DIMENSION_TOKEN_AXIS];
        let (keys_packed, keys_scales, keys_biases) =
            runtime.quantize_affine(&restored_keys, self.group_size, self.bits)?;
        let (values_packed, values_scales, values_biases) =
            runtime.quantize_affine(&restored_values, self.group_size, self.bits)?;
        self.keys = Some(headroom_slab(
            runtime,
            (keys_packed, keys_scales, keys_biases),
            self.full_attention_kv_state_growth_tokens,
        )?);
        self.values = Some(headroom_slab(
            runtime,
            (values_packed, values_scales, values_biases),
            self.full_attention_kv_state_growth_tokens,
        )?);
        self.offset_tokens = restored_token_count;
        Ok(())
    }

    /// Dequantizes one token-range slice of the stored keys or values back to
    /// the storage dtype, which keeps the SSD prompt-cache format bfloat16.
    pub fn dequantized_token_range(
        &self,
        runtime: &MlxRuntime,
        is_keys: bool,
        start_tokens: i32,
        end_tokens: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let slab = if is_keys {
            self.keys.as_ref()
        } else {
            self.values.as_ref()
        }
        .ok_or_else(|| quantized_state_error("quantized KV state has no storage to dequantize"))?;
        let packed = slice_token_range(runtime, &slab.packed, start_tokens, end_tokens)?;
        let scales = slice_token_range(runtime, &slab.scales, start_tokens, end_tokens)?;
        let biases = slice_token_range(runtime, &slab.biases, start_tokens, end_tokens)?;
        runtime.dequantize_affine(&packed, &scales, &biases, self.group_size, self.bits)
    }
}

fn validate_and_quantize_update(
    state: &QuantizedFullAttentionKeyValueState,
    runtime: &MlxRuntime,
    new_keys: &MlxArray,
    new_values: &MlxArray,
    previous_token_count: i32,
) -> Result<(QuantizedSlab, QuantizedSlab), MlxRuntimeError> {
    let key_shape = new_keys.shape();
    if key_shape.len() != 4
        || key_shape != new_values.shape()
        || key_shape[STATE_DIMENSION_TOKEN_AXIS] <= 0
    {
        return Err(quantized_state_error(
            "new K and V tensors must have identical rank-four nonempty shapes",
        ));
    }
    if previous_token_count != state.offset_tokens {
        return Err(quantized_state_error(
            "append position does not match the in-memory quantized KV state offset",
        ));
    }
    if state.keys.is_some() != state.values.is_some() {
        return Err(quantized_state_error(
            "in-memory K and V storage must both be present or absent",
        ));
    }
    let (keys_packed, keys_scales, keys_biases) =
        runtime.quantize_affine(new_keys, state.group_size, state.bits)?;
    let (values_packed, values_scales, values_biases) =
        runtime.quantize_affine(new_values, state.group_size, state.bits)?;
    Ok((
        QuantizedSlab {
            packed: keys_packed,
            scales: keys_scales,
            biases: keys_biases,
        },
        QuantizedSlab {
            packed: values_packed,
            scales: values_scales,
            biases: values_biases,
        },
    ))
}

/// Splices the quantized update into the current slabs, growing every slab by
/// one configured step when the update no longer fits. The shared bfloat16
/// storage builder already owns all three mechanics: initial allocation,
/// in-place splice, and retain-prefix-then-extend growth.
fn grow_slab(
    runtime: &MlxRuntime,
    current_slab: Option<&QuantizedSlab>,
    quantized_update: QuantizedSlab,
    previous_token_count: i32,
    growth_tokens: i32,
) -> Result<QuantizedSlab, MlxRuntimeError> {
    let packed = build_updated_storage(
        runtime,
        current_slab.map(|slab| &slab.packed),
        &quantized_update.packed,
        previous_token_count,
        growth_tokens,
    )?
    .retain()?;
    let scales = build_updated_storage(
        runtime,
        current_slab.map(|slab| &slab.scales),
        &quantized_update.scales,
        previous_token_count,
        growth_tokens,
    )?
    .retain()?;
    let biases = build_updated_storage(
        runtime,
        current_slab.map(|slab| &slab.biases),
        &quantized_update.biases,
        previous_token_count,
        growth_tokens,
    )?
    .retain()?;
    Ok(QuantizedSlab {
        packed,
        scales,
        biases,
    })
}

/// Appends one growth step of zero capacity to a freshly quantized slab
/// triple so the first post-restore update splices instead of copying.
fn headroom_slab(
    runtime: &MlxRuntime,
    quantized: (MlxArray, MlxArray, MlxArray),
    growth_tokens: i32,
) -> Result<QuantizedSlab, MlxRuntimeError> {
    let (packed, scales, biases) = quantized;
    Ok(QuantizedSlab {
        packed: append_zero_extension(runtime, packed, growth_tokens)?,
        scales: append_zero_extension(runtime, scales, growth_tokens)?,
        biases: append_zero_extension(runtime, biases, growth_tokens)?,
    })
}

fn append_zero_extension(
    runtime: &MlxRuntime,
    storage: MlxArray,
    growth_tokens: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let mut extension_shape = storage.shape();
    extension_shape[STATE_DIMENSION_TOKEN_AXIS] = growth_tokens;
    let extension = runtime.zeros(&extension_shape, storage.dtype())?;
    runtime.concatenate_axis(&[&storage, &extension], STATE_DIMENSION_TOKEN_AXIS as i32)
}

/// Returns strided views over the written prefix of every slab component.
fn active_slab_views(
    runtime: &MlxRuntime,
    slab: &QuantizedSlab,
    active_token_count: i32,
) -> Result<QuantizedTensorViews, MlxRuntimeError> {
    Ok(QuantizedTensorViews {
        packed: active_view(runtime, &slab.packed, active_token_count)?,
        scales: active_view(runtime, &slab.scales, active_token_count)?,
        biases: active_view(runtime, &slab.biases, active_token_count)?,
    })
}

fn slice_token_range(
    runtime: &MlxRuntime,
    tensor: &MlxArray,
    start_tokens: i32,
    end_tokens: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = tensor.shape();
    let mut starts = vec![0; shape.len()];
    let mut stops = shape;
    starts[STATE_DIMENSION_TOKEN_AXIS] = start_tokens;
    stops[STATE_DIMENSION_TOKEN_AXIS] = end_tokens;
    let strides = vec![1; starts.len()];
    runtime.slice(tensor, &starts, &stops, &strides)
}

fn quantized_state_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: QUANTIZED_STATE_OPERATION,
        description: description.to_owned(),
    }
}
