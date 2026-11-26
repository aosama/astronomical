//! Kept dense layers in register order (issues #2568 and #2585).
//!
//! A model that hosts a drafter verifies a draft tree of about sixteen rows
//! every round, and at that many rows the register-ordered kernel multiplies
//! the dense 27B's projections in about two thirds of the time the
//! row-batched kernel takes on MLX's layout. The resident engine converts
//! every projection when its drafter binds; here the same conversion is
//! applied to each layer the streamer keeps, so a model with every layer kept
//! verifies as the resident engine does.
//!
//! Only kept layers convert. A streamed layer is read from the SSD for one
//! forward in MLX's layout, and converting it would cost more than the pass
//! it serves; kept and streamed layers sit side by side because each
//! projection multiplies through the kernel its own storage names.
//!
//! A converted layer takes the same bytes as the one it replaces (the codes
//! word for word, scale and bias pairs in place of two arrays), so the
//! streamer's payload figures and the governor's plan are unchanged.

use std::rc::Rc;

use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;

use super::dense_layer_streamer::Qwen3_5DenseLayerStreamer;
use super::streaming_model::Qwen3_5StreamingModel;
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model::register_ordered_projections::convert_layer_projections_to_register_order;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::row_batched_matmul::RowBatchedMatmulKernels;

/// How a layer read from the SSD is bound: the model's configuration and
/// whether its normalization offsets are stored the Hugging Face way.
#[derive(Clone, Copy)]
pub struct LayerBinding<'model> {
    pub qwen3_5_config: &'model Qwen3_5Config,
    pub raw_hugging_face_norm_offsets_apply: bool,
}

impl Qwen3_5DenseLayerStreamer {
    /// Reads one layer to be kept and evaluates it, in register order when
    /// `register_order_kernels` are given. Converting before the layer is
    /// kept means no forward ever sees it in both layouts.
    pub(super) fn read_layer_to_keep(
        &self,
        runtime: &MlxRuntime,
        layer_binding: LayerBinding<'_>,
        layer_index: usize,
        register_order_kernels: Option<&RowBatchedMatmulKernels>,
    ) -> Result<Qwen3_5DecoderLayerWeights, TextExecutionError> {
        let mut layer_weights = self.read_layer(
            runtime,
            layer_binding.qwen3_5_config,
            layer_binding.raw_hugging_face_norm_offsets_apply,
            layer_index,
        )?;
        if let Some(register_order_kernels) = register_order_kernels {
            convert_layer_projections_to_register_order(
                runtime,
                register_order_kernels,
                &mut layer_weights,
            )?;
        }
        let mut layer_arrays = Vec::new();
        layer_weights.append_array_references(&mut layer_arrays);
        runtime.evaluate_arrays(&layer_arrays)?;
        Ok(layer_weights)
    }

    /// Converts the layers kept now and returns how many projections were
    /// converted. A layer a forward still holds is left in MLX's layout,
    /// which is as exact and only slower; the conversion runs while a
    /// drafter binds, when no forward does.
    pub fn convert_kept_layers_to_register_order(
        &self,
        runtime: &MlxRuntime,
        register_order_kernels: &RowBatchedMatmulKernels,
    ) -> Result<usize, TextExecutionError> {
        let mut converted_projection_count = 0;
        for kept_layer_weights in self.kept_layer_weights.borrow_mut().iter_mut() {
            let Some(solely_held_layer_weights) = Rc::get_mut(kept_layer_weights) else {
                continue;
            };
            converted_projection_count += convert_layer_projections_to_register_order(
                runtime,
                register_order_kernels,
                solely_held_layer_weights,
            )?;
        }
        Ok(converted_projection_count)
    }
}

impl Qwen3_5StreamingModel {
    /// The kernels a layer read to be kept is converted with: present once a
    /// drafter's tree kernels are declared, absent for a model that decodes
    /// a row a step, whose projections keep MLX's layout.
    pub(super) fn kept_layer_register_order_kernels(&self) -> Option<&RowBatchedMatmulKernels> {
        self.optional_accelerators
            .register_ordered_matmul_kernels
            .as_ref()
    }

    /// Converts the dense layers kept now, for a drafter that is binding.
    /// Returns 0 for a sparse model, whose expert stacks stay in MLX's
    /// layout for the gathered expert kernels, and before the kernels are
    /// declared.
    #[doc(hidden)]
    pub fn convert_kept_dense_layers_to_register_order(&self) -> Result<usize, TextExecutionError> {
        let (Some(dense_layer_streamer), Some(register_order_kernels)) = (
            &self.dense_layer_streamer,
            self.kept_layer_register_order_kernels(),
        ) else {
            return Ok(0);
        };
        dense_layer_streamer
            .convert_kept_layers_to_register_order(&self.runtime, register_order_kernels)
    }

    /// Converts a dense model's output head, for a drafter that is binding.
    /// A sparse model's head keeps MLX's layout with the rest of it: its
    /// tree rows multiply the expert stacks through the gathered kernels,
    /// so its drafted reply is held to rounding either way.
    pub(super) fn convert_dense_output_head_to_register_order(
        &mut self,
    ) -> Result<usize, TextExecutionError> {
        if self.dense_layer_streamer.is_none() {
            return Ok(0);
        }
        let Some(register_order_kernels) = self
            .optional_accelerators
            .register_ordered_matmul_kernels
            .take()
        else {
            return Ok(0);
        };
        let converted_head = self.convert_output_head_to_register_order(&register_order_kernels);
        self.optional_accelerators.register_ordered_matmul_kernels = Some(register_order_kernels);
        converted_head
    }
}
