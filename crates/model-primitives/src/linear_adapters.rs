//! Add-on weights on a model's linear layers, applied in the forward pass.
//!
//! An add-on gives its layer a second, thin path: the input is narrowed to a
//! few numbers by `down` (`[rank, inputs]`) and widened back by `up`
//! (`[outputs, rank]`), and the result is added to the layer's own output.
//! Serving reads them from `decision_adapters.safetensors`; training passes
//! the arrays it is moving. Both go through [`LinearAdapters::adapted_output`],
//! so the trained model is scored by exactly the computation that serves it.
//!
//! The add-ons are applied in 32-bit floats and never merged into the 8-bit
//! weights: re-rounding a merged weight would blur the small changes
//! training made.

use std::collections::BTreeMap;

use decision_contract::decision_adapters_file::DecisionAdapters;
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError};

/// One layer's add-on on the GPU.
pub struct LinearAdapterArrays {
    /// `[rank, inputs]`.
    pub down: MlxArray,
    /// `[outputs, rank]`, with any scale folded in.
    pub up: MlxArray,
}

/// Every add-on of a model, by the shipped weight prefix of its layer.
#[derive(Default)]
pub struct LinearAdapters {
    by_target: BTreeMap<String, LinearAdapterArrays>,
}

impl LinearAdapters {
    /// Uploads a folder's add-on file.
    ///
    /// # Errors
    /// Returns the MLX description when an upload fails.
    pub fn upload(
        context: &MlxBindingsContext,
        decision_adapters: &DecisionAdapters,
    ) -> Result<Self, MlxCError> {
        let mut by_target = BTreeMap::new();
        for adapter in &decision_adapters.adapters {
            // A layer's widths and a rank are far below `i32::MAX`; the
            // engine has already checked them against the model's layers.
            let down = context.array_from_f32(
                &adapter.down_values,
                &[adapter.rank as i32, adapter.input_width as i32],
            )?;
            let up = context.array_from_f32(
                &adapter.up_values,
                &[adapter.output_width as i32, adapter.rank as i32],
            )?;
            by_target.insert(adapter.target.clone(), LinearAdapterArrays { down, up });
        }
        Ok(Self { by_target })
    }

    /// Pairs `trainable_arrays`, laid out as `down, up` per target in
    /// `targets` order, with their targets; the arrays are shared, not copied.
    ///
    /// # Errors
    /// Returns the MLX description when a handle cannot be shared.
    pub fn from_trainable_arrays(
        targets: &[String],
        trainable_arrays: &[MlxArray],
    ) -> Result<Self, MlxCError> {
        let mut by_target = BTreeMap::new();
        let (adapter_pairs, _) = trainable_arrays.as_chunks::<2>();
        for (target, [down, up]) in targets.iter().zip(adapter_pairs) {
            by_target.insert(
                target.clone(),
                LinearAdapterArrays {
                    down: down.retain()?,
                    up: up.retain()?,
                },
            );
        }
        Ok(Self { by_target })
    }

    /// The add-ons of `targets` as `down, up` per target, in `targets`
    /// order, sharing the arrays: the layout
    /// [`Self::from_trainable_arrays`] reads back.
    ///
    /// # Errors
    /// Names a target that has no add-on, or returns the MLX description
    /// when a handle cannot be shared.
    pub fn arrays_of(&self, targets: &[String]) -> Result<Vec<MlxArray>, MlxCError> {
        let mut target_arrays = Vec::with_capacity(targets.len() * 2);
        for target in targets {
            let Some(adapter_arrays) = self.by_target.get(target) else {
                return Err(MlxCError {
                    operation: "read a layer's add-ons",
                    description: format!("{target} has no add-on"),
                });
            };
            target_arrays.push(adapter_arrays.down.retain()?);
            target_arrays.push(adapter_arrays.up.retain()?);
        }
        Ok(target_arrays)
    }

    /// Whether no layer has an add-on.
    pub fn is_empty(&self) -> bool {
        self.by_target.is_empty()
    }

    /// The layers that have an add-on, in name order.
    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.by_target.keys().map(String::as_str)
    }

    /// `base_output + (input · downᵀ) · upᵀ` when `target` has an add-on,
    /// otherwise `base_output` unchanged.
    ///
    /// # Errors
    /// Returns the MLX description when a product cannot be built.
    pub fn adapted_output(
        &self,
        context: &MlxBindingsContext,
        target: &str,
        layer_input: &MlxArray,
        base_output: MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let Some(adapter) = self.by_target.get(target) else {
            return Ok(base_output);
        };
        let narrowed = context.matmul(
            layer_input,
            &context.transpose_axes(&adapter.down, &[1, 0])?,
        )?;
        let widened = context.matmul(&narrowed, &context.transpose_axes(&adapter.up, &[1, 0])?)?;
        // The add-on is computed in 32-bit floats, but the layer's output
        // keeps its own dtype: a bf16 backbone (Kev, Clef) widened to 32 bits
        // here would run every later operation at another precision, and a
        // folder whose add-ons are zero would no longer serve the shipped
        // scores bit for bit. Laya's layers are already 32-bit.
        let layer_dtype_add_on = if widened.dtype() == base_output.dtype() {
            widened
        } else {
            context.astype(&widened, base_output.dtype())?
        };
        context.add(&base_output, &layer_dtype_add_on)
    }
}

/// Applies `adapters`, when there are any, to one layer's output.
///
/// # Errors
/// Returns the MLX description when a product cannot be built.
pub fn adapted_layer_output(
    adapters: Option<&LinearAdapters>,
    context: &MlxBindingsContext,
    target: &str,
    layer_input: &MlxArray,
    base_output: MlxArray,
) -> Result<MlxArray, MlxCError> {
    match adapters {
        Some(linear_adapters) => {
            linear_adapters.adapted_output(context, target, layer_input, base_output)
        }
        None => Ok(base_output),
    }
}

#[cfg(test)]
#[path = "linear_adapters_tests.rs"]
mod tests;
