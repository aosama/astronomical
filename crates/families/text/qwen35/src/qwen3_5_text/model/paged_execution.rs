//! MLX gathered execution against one Rust-streamed expert layer.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

use crate::qwen3_5_text::model::Qwen3_5Model;
use crate::sparse_experts::{
    ExpertAssignmentOrder, StackedExpertProjection, gather_expert_projection,
};
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};
use engine_core::expert_paging::QuantizedExpertPageManifest;
use text_generation::execution_error::TextExecutionError;

use super::super::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use super::astro_expert_cache_plus_streamed_page_route;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::streamed_expert_assignment_steps::remaining_gate_up_rotations;
use crate::sparse_experts;

/// Per-assignment routed expert rows before any weighted reduction.
pub(super) enum RoutedExpertAssignmentOutputs {
    /// Sorted gather rows plus the inverse permutation the sorted reduction needs.
    Sorted {
        sorted_outputs: MlxArray,
        inverse_order: MlxArray,
    },
    /// Original-order assignment rows.
    Unsorted { assignment_outputs: MlxArray },
}

/// Where a page's slot of each routed expert comes from.
pub(super) enum PageSlots<'route> {
    /// Remapped on the GPU from the routed expert ids, through the page's
    /// manifest as it stood when the graph was built.
    RemappedOnDevice {
        selected_indices: &'route MlxArray,
        sorted_unique_expert_ids: &'route [usize],
        page_manifest: &'route QuantizedExpertPageManifest,
    },
    /// Named by a lookup inside the graph (`in_graph_table_serving`), which
    /// may seat experts in the page while the graph is evaluated.
    LookedUp(&'route MlxArray),
}

impl Qwen3_5Model {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        page_manifest: &QuantizedExpertPageManifest,
        selected_indices: &MlxArray,
        sorted_unique_expert_ids: &[usize],
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        self.forward_moe_from_page_slots(
            PageSlots::RemappedOnDevice {
                selected_indices,
                sorted_unique_expert_ids,
                page_manifest,
            },
            selected_scores,
            paged_expert_weights,
            hidden_states,
            mixture_of_experts_weights,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }

    /// One layer's sparse expert output multiplied from a page, with the
    /// shared (non-expert) outputs folded in, measured as one step.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_from_page_slots(
        &self,
        page_slots: PageSlots<'_>,
        selected_scores: &MlxArray,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let paged_output = performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |measured_attribution| {
                self.build_paged_sparse_output(
                    hidden_states,
                    mixture_of_experts_weights,
                    paged_expert_weights,
                    page_slots,
                    selected_scores,
                    should_use_compiled_elementwise_graphs,
                    measured_attribution,
                )
            },
        )?;
        performance_attribution.record_counter(
            PerformanceCounter::RustStreamedExpertProjectionGraphCount,
            3,
        );
        Ok(paged_output)
    }

    /// Builds the paged sparse expert output for one decode forward: finds
    /// the routed experts' page slots, streams the weights through the
    /// gathered projection, and folds the shared (non-expert) outputs in.
    ///
    /// Extracted from the `PagedMoeGraphConstruction` measurement so the
    /// measured region reads as one named step instead of an inline closure.
    #[allow(clippy::too_many_arguments)]
    fn build_paged_sparse_output(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        page_slots: PageSlots<'_>,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let page_slot_indices = self.resolved_page_slots(page_slots)?;
        let routed_assignment_outputs = self.streamed_expert_assignment_outputs(
            hidden_states,
            paged_expert_weights,
            &page_slot_indices,
            performance_attribution,
        )?;
        self.combine_routed_assignments_and_shared_outputs(
            hidden_states,
            mixture_of_experts_weights,
            (&routed_assignment_outputs, selected_scores),
            should_use_compiled_elementwise_graphs,
        )
    }

    /// The page slot of every routed assignment, as a lazy array.
    fn resolved_page_slots(
        &self,
        page_slots: PageSlots<'_>,
    ) -> Result<MlxArray, TextExecutionError> {
        match page_slots {
            PageSlots::LookedUp(table_slots) => Ok(table_slots.retain()?),
            PageSlots::RemappedOnDevice {
                selected_indices,
                sorted_unique_expert_ids,
                page_manifest,
            } => Ok(
                astro_expert_cache_plus_streamed_page_route::qwen3_5_moe_remap_expert_page_slots(
                    &self.runtime,
                    selected_indices,
                    sorted_unique_expert_ids,
                    page_manifest,
                )?,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_streamed_weights(
        &self,
        hidden_states: &MlxArray,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_expert_indices: &MlxArray,
        selected_scores: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let selected_outputs = self.streamed_expert_assignment_outputs(
            hidden_states,
            paged_expert_weights,
            selected_expert_indices,
            performance_attribution,
        )?;
        self.reduced_routed_assignment_outputs(&selected_outputs, selected_scores)
    }

    /// Runs the routed expert chain through one page and returns one row per
    /// assignment, before any weighted reduction.
    ///
    /// Issue #373: mixed serving gathers the covered assignments from the warm
    /// table and the missing assignments from the streamed page, then restores
    /// the original assignment order so ONE weighted reduction consumes the
    /// same values in the same order as the single-page path. Returning rows
    /// instead of a reduced output is what makes that possible.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn streamed_expert_assignment_outputs(
        &self,
        hidden_states: &MlxArray,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_expert_indices: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<RoutedExpertAssignmentOutputs, TextExecutionError> {
        let expanded_states = self.runtime.expand_dims(hidden_states, -2)?;
        let expanded_states = self.runtime.expand_dims(&expanded_states, -3)?;
        let sorted_assignments =
            self.sorted_assignments_when_worthwhile(&expanded_states, selected_expert_indices)?;
        let (expert_inputs, expert_indices, indices_are_sorted) = match sorted_assignments.as_ref()
        {
            Some((sorted_states, sorted_indices, _, _)) => (sorted_states, sorted_indices, true),
            None => (&expanded_states, selected_expert_indices, false),
        };
        let (selected_gate, selected_up) = self.selected_gate_and_up(
            (
                hidden_states,
                selected_expert_indices,
                sorted_assignments.as_ref(),
            ),
            (expert_inputs, expert_indices, indices_are_sorted),
            paged_expert_weights,
            performance_attribution,
        )?;
        let selected_activated = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &selected_gate,
            &selected_up,
        )?;
        let selected_outputs = self.streamed_expert_linear(
            &selected_activated,
            &paged_expert_weights.down_projection,
            paged_expert_weights.down_activation_rotation,
            expert_indices,
            indices_are_sorted,
            performance_attribution,
        )?;
        self.routed_assignment_outputs_from(
            selected_outputs,
            sorted_assignments,
            indices_are_sorted,
        )
    }

    /// `(gate, up)` through the sorted-expert matmul from the token rows when
    /// the chunk takes it, through MLX's gathered multiply otherwise.
    fn selected_gate_and_up(
        &self,
        (hidden_states, selected_expert_indices, sorted_assignments): (
            &MlxArray,
            &MlxArray,
            Option<&super::streamed_expert_assignment_steps::SortedAssignments>,
        ),
        gathered_inputs: (&MlxArray, &MlxArray, bool),
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, MlxArray), TextExecutionError> {
        match self.token_row_gate_and_up(
            hidden_states,
            paged_expert_weights,
            sorted_assignments,
            selected_expert_indices,
        )? {
            Some(gate_and_up) => Ok(gate_and_up),
            None => self.gathered_gate_and_up(
                gathered_inputs,
                paged_expert_weights,
                performance_attribution,
            ),
        }
    }

    /// `(gate, up)` through MLX's gathered multiply, on the sorted copy of the
    /// rows when the chunk was sorted, with the shared Hadamard-32 rotation
    /// when both projections were quantized under it.
    fn gathered_gate_and_up(
        &self,
        (expert_inputs, expert_indices, indices_are_sorted): (&MlxArray, &MlxArray, bool),
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, MlxArray), TextExecutionError> {
        let gate_activation_rotation = paged_expert_weights.gate_activation_rotation;
        let up_activation_rotation = paged_expert_weights.up_activation_rotation;
        let shared_gate_up_activations = self.shared_gate_up_activations(
            expert_inputs,
            gate_activation_rotation,
            up_activation_rotation,
            performance_attribution,
        )?;
        let gate_up_activations = shared_gate_up_activations.as_ref().unwrap_or(expert_inputs);
        let (gate_projection_rotation, up_projection_rotation) = remaining_gate_up_rotations(
            shared_gate_up_activations.is_some(),
            gate_activation_rotation,
            up_activation_rotation,
        );
        let selected_up = self.streamed_expert_linear(
            gate_up_activations,
            &paged_expert_weights.up_projection,
            up_projection_rotation,
            expert_indices,
            indices_are_sorted,
            performance_attribution,
        )?;
        let selected_gate = self.streamed_expert_linear(
            gate_up_activations,
            &paged_expert_weights.gate_projection,
            gate_projection_rotation,
            expert_indices,
            indices_are_sorted,
            performance_attribution,
        )?;
        Ok((selected_gate, selected_up))
    }

    fn streamed_expert_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights,
        activation_rotation: jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
        selected_expert_indices: &MlxArray,
        are_expert_indices_sorted: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
        // Streamed pages and resident layers intentionally converge here on the
        // same canonical operation. Paging decides which arrays are alive; this
        // adapter only describes their matrix layout and assignment ordering.
        // The layer's full expert count, not this page's, so the quantized
        // gather picks the same kernel whichever page it multiplies.
        let assignment_order = sparse_experts::expert_assignment_order(
            are_expert_indices_sorted,
            selected_expert_indices.element_count(),
            self.config.expert_count() as usize,
        );
        let rotated_activation_owner = self.hadamard32_rotated_activations(
            activations,
            activation_rotation,
            performance_attribution,
        )?;
        let projection_activations = rotated_activation_owner.as_ref().unwrap_or(activations);
        let projection_inputs = StreamedProjectionInputs {
            projection_activations,
            selected_expert_indices,
            assignment_order,
        };
        match affine_weights {
            Qwen3_5AffineWeights::Jangtq2Codebook {
                packed_code_words,
                row_scales,
                codebook_bit_width,
            } => {
                let codebook_slices = Jangtq2CodebookProjectionSlices {
                    packed_code_words,
                    row_scales,
                    codebook_bit_width: *codebook_bit_width,
                };
                self.project_through_jangtq2_codebook(
                    &projection_inputs,
                    &codebook_slices,
                    performance_attribution,
                )
            }
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => self
                .project_through_native_bfloat16(
                    &projection_inputs,
                    weight,
                    performance_attribution,
                ),
            Qwen3_5AffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => {
                let quantized_slices = QuantizedProjectionSlices {
                    packed_weight,
                    quantization_scales,
                    quantization_biases,
                    quantization_group_size: *quantization_group_size,
                    quantization_bits: *quantization_bits,
                };
                self.project_through_quantized(
                    &projection_inputs,
                    &quantized_slices,
                    performance_attribution,
                )
            }
        }
    }

    /// Rotates the activation rows through the blockwise Hadamard-32 transform
    /// only when the projection's codebook was quantized under that rotation;
    /// the measured region covers exactly the rotation work, nothing else.
    fn hadamard32_rotated_activations(
        &self,
        activations: &MlxArray,
        activation_rotation: jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        let rotated_activation_owner = match activation_rotation {
            jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation => None,
            jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32 => {
                Some(performance_attribution.measure_operation(
                    PerformanceOperation::Jangtq2ActivationRotationHadamard32,
                    |_| {
                        jangtq2_qwen35::mlx_decode::rotate_activations_hadamard32(
                            &self.runtime,
                            activations,
                        )
                    },
                )?)
            }
        };
        Ok(rotated_activation_owner)
    }

    /// Routes one jangtq2 codebook projection: a single decode token multiplies
    /// straight from the packed codes through the fused kernel, and any wider
    /// forward decodes a transient copy for the gather_mm path.
    fn project_through_jangtq2_codebook(
        &self,
        projection_inputs: &StreamedProjectionInputs<'_>,
        codebook_slices: &Jangtq2CodebookProjectionSlices<'_>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        if let Some(projected) = self.project_through_fused_jangtq2_decode_kernel(
            projection_inputs,
            codebook_slices,
            performance_attribution,
        )? {
            return Ok(projected);
        }
        self.project_through_decoded_jangtq2_codebook(
            projection_inputs,
            codebook_slices,
            performance_attribution,
        )
    }

    /// Attempts the fused single-decode-token projection and returns `None`
    /// when this worker lacks the fused kernel or the forward routes more
    /// assignments than one token, handing the work to the decoded-copy path.
    fn project_through_fused_jangtq2_decode_kernel(
        &self,
        projection_inputs: &StreamedProjectionInputs<'_>,
        codebook_slices: &Jangtq2CodebookProjectionSlices<'_>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        if let Some(fused_kernel) = self.fused_jangtq2_gathered_projection_kernel.as_ref()
            && let Some(activation_rows) = decode_token_activation_rows(
                &self.runtime,
                projection_inputs.projection_activations,
                projection_inputs.selected_expert_indices,
                usize::try_from(self.config.experts_per_token()).unwrap_or(0),
            )?
        {
            // One decode token: multiply straight from the packed
            // codes. The kernel returns `[assignments, rows]`;
            // restore gather_mm's `[..., assignments, 1, rows]`
            // layout so the SwiGLU and reduction stay unchanged.
            let projected = performance_attribution.measure_operation(
                PerformanceOperation::DecodeFusedJangtq2GatheredProjection,
                |_| {
                    let flat_expert_indices = self.runtime.reshape(
                        projection_inputs.selected_expert_indices,
                        &[projection_inputs.selected_expert_indices.element_count() as i32],
                    )?;
                    fused_kernel.apply(
                        &self.runtime,
                        &activation_rows,
                        &flat_expert_indices,
                        codebook_slices.packed_code_words,
                        codebook_slices.row_scales,
                        codebook_slices.codebook_bit_width,
                    )
                },
            )?;
            let mut gathered_shape = projection_inputs.selected_expert_indices.shape();
            gathered_shape.push(1);
            gathered_shape.push(codebook_slices.packed_code_words.shape()[1]);
            return Ok(Some(self.runtime.reshape(&projected, &gathered_shape)?));
        }
        Ok(None)
    }

    fn project_through_decoded_jangtq2_codebook(
        &self,
        projection_inputs: &StreamedProjectionInputs<'_>,
        codebook_slices: &Jangtq2CodebookProjectionSlices<'_>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        // Multi-token prefill (or a worker without the fused kernel):
        // decode a transient bfloat16 copy and gather through
        // gather_mm, whose matrix multiply amortizes the decode over
        // the chunk's tokens. The copy dies with this forward.
        let decoded_weight = performance_attribution.measure_operation(
            PerformanceOperation::Jangtq2CodebookPageDecode,
            |_| {
                jangtq2_qwen35::mlx_decode::decode_expert_weight_rows(
                    &self.runtime,
                    codebook_slices.packed_code_words,
                    codebook_slices.row_scales,
                    codebook_slices.codebook_bit_width,
                )
            },
        )?;
        let packed_shape = codebook_slices.packed_code_words.shape();
        // The decoded bfloat16 volume is the round-trip cost the fused
        // prefill matmul exists to remove; pricing it per forward makes
        // chunk-size amortization visible in every request's record.
        let decoded_element_count = (packed_shape[0] as u64)
            .saturating_mul(packed_shape[1] as u64)
            .saturating_mul(
                (packed_shape[2] as u64).saturating_mul(32)
                    / u64::from(codebook_slices.codebook_bit_width),
            );
        performance_attribution.record_counter(
            PerformanceCounter::Jangtq2CodebookDecodedOutputByteCount,
            decoded_element_count.saturating_mul(std::mem::size_of::<u16>() as u64),
        );
        let transposed_weights = self.runtime.transpose_axes(&decoded_weight, &[0, 2, 1])?;
        Ok(gather_expert_projection(
            &self.runtime,
            projection_inputs.projection_activations,
            StackedExpertProjection::Dense {
                transposed_weights: &transposed_weights,
            },
            projection_inputs.selected_expert_indices,
            projection_inputs.assignment_order,
            performance_attribution,
        )?)
    }

    fn project_through_native_bfloat16(
        &self,
        projection_inputs: &StreamedProjectionInputs<'_>,
        weight: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        // The transpose is a lazy view from checkpoint linear layout to
        // the `[expert, input, output]` layout expected by gather_mm.
        let transposed_weights = self.runtime.transpose_axes(weight, &[0, 2, 1])?;
        Ok(gather_expert_projection(
            &self.runtime,
            projection_inputs.projection_activations,
            StackedExpertProjection::Dense {
                transposed_weights: &transposed_weights,
            },
            projection_inputs.selected_expert_indices,
            projection_inputs.assignment_order,
            performance_attribution,
        )?)
    }

    fn project_through_quantized(
        &self,
        projection_inputs: &StreamedProjectionInputs<'_>,
        quantized_slices: &QuantizedProjectionSlices<'_>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        Ok(gather_expert_projection(
            &self.runtime,
            projection_inputs.projection_activations,
            // Pass the streamed packed page directly to gather_qmm. Taking
            // selected matrices first would recreate the memory expansion
            // that gathered execution exists to avoid.
            StackedExpertProjection::Affine {
                packed_weights: quantized_slices.packed_weight,
                scales: quantized_slices.quantization_scales,
                biases: quantized_slices.quantization_biases,
                group_size: quantized_slices.quantization_group_size,
                bits: quantized_slices.quantization_bits,
            },
            projection_inputs.selected_expert_indices,
            projection_inputs.assignment_order,
            performance_attribution,
        )?)
    }
}

/// Activation rows, route indices, and assignment ordering shared by every
/// streamed projection arm, so each arm's parameters describe only its
/// weight layout.
struct StreamedProjectionInputs<'a> {
    projection_activations: &'a MlxArray,
    selected_expert_indices: &'a MlxArray,
    assignment_order: ExpertAssignmentOrder,
}

/// The packed on-disk jangtq2 codebook arrays one streamed projection reads.
struct Jangtq2CodebookProjectionSlices<'a> {
    packed_code_words: &'a MlxArray,
    row_scales: &'a MlxArray,
    codebook_bit_width: u32,
}

/// The packed affine quantization arrays one streamed projection reads.
struct QuantizedProjectionSlices<'a> {
    packed_weight: &'a MlxArray,
    quantization_scales: &'a MlxArray,
    quantization_biases: &'a MlxArray,
    quantization_group_size: i32,
    quantization_bits: i32,
}

/// Returns the activation rows a single decode token presents to the fused
/// gathered projection kernel, or `None` for a multi-token forward.
///
/// The unsorted decode path carries one activation row shared by every
/// assignment (`[1, 1, 1, width]`); the sorted path carries one row per
/// assignment (`[assignments, 1, width]`). Both flatten to `[rows, width]`
/// with `rows` equal to one or to the assignment count. A chunk of several
/// tokens has more assignments than one token routes, and keeps the
/// materializing path: the kernel is a matrix-vector product and the
/// chunk's matrix multiply belongs on `gather_mm`.
pub(super) fn decode_token_activation_rows(
    runtime: &MlxRuntime,
    projection_activations: &MlxArray,
    selected_expert_indices: &MlxArray,
    experts_per_token: usize,
) -> Result<Option<MlxArray>, TextExecutionError> {
    let activation_shape = projection_activations.shape();
    let Some(&input_width) = activation_shape.last() else {
        return Ok(None);
    };
    if input_width <= 0 {
        return Ok(None);
    }
    let activation_row_count = projection_activations.element_count() / input_width as usize;
    let assignment_count = selected_expert_indices.element_count();
    if assignment_count == 0 || assignment_count > experts_per_token {
        return Ok(None);
    }
    if activation_row_count != 1 && activation_row_count != assignment_count {
        return Ok(None);
    }
    Ok(Some(runtime.reshape(
        projection_activations,
        &[activation_row_count as i32, input_width],
    )?))
}
