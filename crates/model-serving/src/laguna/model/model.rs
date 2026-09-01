//! Laguna model: embed, descriptor-ordered layers, final norm, output head.

use astronomical_ipc_protocol::{ExpertMemoryMode, graph_submission_layer_interval};
use astronomical_runtime_integration::{MlxArray, MlxCompiledSwiGlu, MlxMetalKernel, MlxRuntime};

use crate::ExpertResidencyTelemetry;
use crate::MlxAllocationAdmission;
use crate::expert_paging::ExpertWeightMemoryCacheStatistics;
use crate::kernel_capability::{CustomMetalKernelFamily, WorkerKernelCapabilities};
use crate::laguna::artifacts::LagunaGlobalTensorRole;
use crate::laguna::normalization::{LagunaFeedForwardDescriptor, LagunaTargetContract};
use crate::laguna::paging::LagunaExpertPagingPlan;
use crate::memory::{ExpertResidencyPlan, MemoryPhase};
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use crate::sparse_experts::sorted_expert_weighted_sum_kernel;

use super::attention::LagunaAttentionMaskCache;
use super::decoder_layer::forward_decoder_layer;
use super::decoder_state::LagunaDecoderState;
use super::error::LagunaExecutionError;
use super::expert_coverage::validate_sparse_coverage;
use super::expert_residency::LagunaExpertResidencyState;
use super::weights::LagunaNativeWeights;

/// Executable Laguna model bound to one canonical contract and native weights.
pub struct LagunaModel {
    pub(super) contract: LagunaTargetContract,
    pub(super) weights: LagunaNativeWeights,
    compiled_swiglu: MlxCompiledSwiGlu,
    sorted_expert_reduction_kernel: Option<MlxMetalKernel>,
    pub(super) residency: LagunaExpertResidencyState,
    pub(super) expert_allocation_budget: Option<MlxAllocationAdmission>,
    prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
}

impl LagunaModel {
    /// Constructs a model from a normalized contract, bound native weights,
    /// and this worker process's retained kernel-capability verdicts.
    pub fn new(
        contract: LagunaTargetContract,
        weights: LagunaNativeWeights,
        worker_kernel_capabilities: &WorkerKernelCapabilities,
    ) -> Result<Self, LagunaExecutionError> {
        if contract.layers().is_empty() {
            return Err(LagunaExecutionError::invalid_geometry(
                "a Laguna model must contain at least one layer descriptor",
            ));
        }
        let has_sparse_layers = contract
            .layers()
            .iter()
            .any(|layer| matches!(layer.feed_forward(), LagunaFeedForwardDescriptor::Moe(_)));
        let sorted_expert_reduction_kernel = match has_sparse_layers {
            false => None,
            true if worker_kernel_capabilities
                .is_custom_kernel_supported(CustomMetalKernelFamily::SortedExpertWeightedSum) =>
            {
                Some(sorted_expert_weighted_sum_kernel()?)
            }
            true => {
                tracing::info!(
                    "sorted expert weighted-sum kernel demoted to the MLX fallback for this worker process"
                );
                None
            }
        };
        let compiled_swiglu = MlxCompiledSwiGlu::new()?;
        Ok(Self {
            contract,
            weights,
            compiled_swiglu,
            sorted_expert_reduction_kernel,
            residency: LagunaExpertResidencyState::new(),
            expert_allocation_budget: None,
            // Production chunking defaults to one decoder layer so macOS can
            // retire each completed group before the next layer is encoded.
            prefill_graph_submission_layer_interval: 0,
            experimental_ssd_paging_prefill_graph_submission_layer_interval: 1,
            experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
        })
    }

    /// Overrides command-buffer submission intervals after construction.
    #[must_use]
    pub fn with_graph_submission_layer_intervals(
        mut self,
        prefill_graph_submission_layer_interval: u32,
        experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
        experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    ) -> Self {
        self.prefill_graph_submission_layer_interval = prefill_graph_submission_layer_interval;
        self.experimental_ssd_paging_prefill_graph_submission_layer_interval =
            experimental_ssd_paging_prefill_graph_submission_layer_interval;
        self.experimental_ssd_paging_generation_graph_submission_layer_interval =
            experimental_ssd_paging_generation_graph_submission_layer_interval;
        self
    }

    /// Attaches page sources so sparse layers without stacked experts can stream.
    pub fn with_paging_plan(
        mut self,
        paging_plan: LagunaExpertPagingPlan,
    ) -> Result<Self, LagunaExecutionError> {
        validate_sparse_coverage(&self.contract, &self.weights, Some(&paging_plan))?;
        let maximum_expert_page_bytes = paging_plan.sparse_layers().iter().try_fold(
            0_u64,
            |maximum_page_bytes, sparse_layer| {
                Ok::<u64, crate::laguna::paging::LagunaPagingError>(
                    maximum_page_bytes.max(
                        sparse_layer
                            .complete_layer_payload_byte_count()?
                            .max(sparse_layer.routed_page_payload_byte_count()?),
                    ),
                )
            },
        )?;
        self.expert_allocation_budget = Some(MlxAllocationAdmission::new(
            maximum_expert_page_bytes,
            u64::MAX,
        ));
        self.residency.attach_paging_plan(paging_plan);
        Ok(self)
    }

    /// Installs the byte ceiling that may keep a just-loaded complete layer.
    pub fn with_retained_expert_ceiling(
        self,
        retained_expert_ceiling_bytes: u64,
    ) -> Result<Self, LagunaExecutionError> {
        self.residency
            .set_retained_expert_ceiling(retained_expert_ceiling_bytes)?;
        Ok(self)
    }

    /// Updates the live retained-expert ceiling and reclaims overflow.
    pub fn set_retained_expert_ceiling(
        &self,
        retained_expert_ceiling_bytes: u64,
    ) -> Result<(), LagunaExecutionError> {
        self.residency
            .set_retained_expert_ceiling(retained_expert_ceiling_bytes)
    }

    /// Returns the canonical contract this model executes.
    #[must_use]
    pub const fn contract(&self) -> &LagunaTargetContract {
        &self.contract
    }

    pub(super) const fn weights(&self) -> &LagunaNativeWeights {
        &self.weights
    }

    pub(super) const fn residency(&self) -> &LagunaExpertResidencyState {
        &self.residency
    }

    pub(super) const fn compiled_swiglu(&self) -> &MlxCompiledSwiGlu {
        &self.compiled_swiglu
    }

    /// Returns whether sparse experts are fully resident, mixed, or streamed.
    #[must_use]
    pub fn expert_memory_mode(&self) -> ExpertMemoryMode {
        self.residency
            .expert_memory_mode(&self.contract, &self.weights)
    }

    /// Returns complete-layer versus routed-page ownership after the last forward.
    #[must_use]
    pub fn expert_residency_telemetry(&self) -> ExpertResidencyTelemetry {
        self.residency
            .expert_residency_telemetry(&self.contract, &self.weights)
    }

    /// Returns cache counters including streamed page-load counts.
    #[must_use]
    pub fn expert_weight_memory_cache_statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        self.residency
            .expert_weight_memory_cache_statistics(&self.contract, &self.weights)
    }

    /// Returns the last published phase-aware residency plan.
    #[must_use]
    pub fn active_expert_residency_plan(&self) -> Option<ExpertResidencyPlan> {
        self.residency.active_plan().clone()
    }

    /// Embeds tokens, runs every layer descriptor, and projects terminal logits.
    pub fn forward(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
        decoder_state: &mut LagunaDecoderState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, LagunaExecutionError> {
        let memory_phase = if token_count_from_token_ids(token_ids)? > 1 {
            MemoryPhase::Prefill
        } else {
            MemoryPhase::Decode
        };
        self.forward_with_memory_phase(
            runtime,
            token_ids,
            decoder_state,
            memory_phase,
            performance_attribution,
        )
    }

    /// Engine orchestration supplies phase explicitly because a valid prefill chunk can hold one token.
    pub fn forward_prefill(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
        decoder_state: &mut LagunaDecoderState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, LagunaExecutionError> {
        self.forward_with_memory_phase(
            runtime,
            token_ids,
            decoder_state,
            MemoryPhase::Prefill,
            performance_attribution,
        )
    }

    /// Decode phase remains explicit so its retention evidence cannot consume prompt policy.
    pub(in crate::laguna) fn forward_decode(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
        decoder_state: &mut LagunaDecoderState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, LagunaExecutionError> {
        self.forward_with_memory_phase(
            runtime,
            token_ids,
            decoder_state,
            MemoryPhase::Decode,
            performance_attribution,
        )
    }

    fn forward_with_memory_phase(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
        decoder_state: &mut LagunaDecoderState,
        memory_phase: MemoryPhase,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, LagunaExecutionError> {
        let terminal_normalized = self.forward_terminal_hidden_states(
            runtime,
            token_ids,
            decoder_state,
            memory_phase,
            performance_attribution,
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::FinalLogitsGraphConstruction,
            |_| {
                if self.contract.model().has_tied_embeddings() {
                    let embedding_weight = self
                        .weights
                        .global(LagunaGlobalTensorRole::TokenEmbedding)?;
                    let transposed_embedding = runtime.transpose_axes(embedding_weight, &[1, 0])?;
                    Ok(runtime.matmul(&terminal_normalized, &transposed_embedding)?)
                } else {
                    self.weights
                        .global_linear(LagunaGlobalTensorRole::OutputHead)?
                        .project(runtime, &terminal_normalized)
                }
            },
        )
    }

    /// Runs one nonterminal prompt chunk and returns a real evaluation root
    /// without constructing vocabulary logits that no sampler can consume.
    pub(in crate::laguna) fn forward_prompt_chunk_without_logits(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
        decoder_state: &mut LagunaDecoderState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, LagunaExecutionError> {
        self.forward_terminal_hidden_states(
            runtime,
            token_ids,
            decoder_state,
            MemoryPhase::Prefill,
            performance_attribution,
        )
    }

    fn forward_terminal_hidden_states(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
        decoder_state: &mut LagunaDecoderState,
        memory_phase: MemoryPhase,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, LagunaExecutionError> {
        self.residency.refresh_explicit_phase_plan(memory_phase);
        let embedding_weight = self
            .weights
            .global(LagunaGlobalTensorRole::TokenEmbedding)?;
        let mut hidden_states = runtime.take_axis(embedding_weight, token_ids, 0)?;
        if hidden_states.shape().len() == 2 {
            hidden_states = runtime.reshape(
                &hidden_states,
                &[1, hidden_states.shape()[0], hidden_states.shape()[1]],
            )?;
        }
        let expected_hidden_size =
            i32::try_from(self.contract.model().hidden_size()).unwrap_or(i32::MAX);
        let actual_hidden_size = *hidden_states.shape().last().unwrap_or(&0);
        if actual_hidden_size != expected_hidden_size {
            return Err(LagunaExecutionError::RuntimeOperation {
                description: format!(
                    "embedded activations have last dimension {actual_hidden_size} but the contract hidden size is {expected_hidden_size}; embedding weight shape is {:?}",
                    embedding_weight.shape()
                ),
            });
        }
        let rms_norm_epsilon = self.contract.model().rms_norm_epsilon() as f32;
        let router_logit_softcap = self.contract.model().router_logit_softcap();
        // Full-attention layers share one mask shape and sliding layers share
        // another. Retain each lazy mask once for this forward instead of
        // rebuilding equivalent token-by-token mask graphs in every layer.
        let mut attention_mask_cache = LagunaAttentionMaskCache::default();
        let graph_submission_layer_interval = usize::try_from(graph_submission_layer_interval(
            token_count_from_hidden_states(&hidden_states)?,
            !matches!(self.expert_memory_mode(), ExpertMemoryMode::Resident),
            self.prefill_graph_submission_layer_interval,
            self.experimental_ssd_paging_prefill_graph_submission_layer_interval,
            self.experimental_ssd_paging_generation_graph_submission_layer_interval,
        ))
        .unwrap_or(0);
        let decoder_layer_count = self.contract.layers().len();
        for (layer_index, layer_descriptor) in self.contract.layers().iter().enumerate() {
            hidden_states = forward_decoder_layer(
                runtime,
                &hidden_states,
                self,
                layer_descriptor,
                decoder_state,
                &mut attention_mask_cache,
                memory_phase,
                rms_norm_epsilon,
                router_logit_softcap,
                self.sorted_expert_reduction_kernel.as_ref(),
                performance_attribution,
            )?;
            if graph_submission_layer_interval > 0
                && (layer_index + 1).is_multiple_of(graph_submission_layer_interval)
                && layer_index + 1 < decoder_layer_count
            {
                // Intermediate mlx_async_eval commits the completed layer group
                // so a fully resident prefill does not encode all 40 layers as
                // one tape against the MLX ceiling.
                performance_attribution.measure_operation(
                    PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                    |_performance_attribution| runtime.async_eval_arrays(&[&hidden_states]),
                )?;
            }
        }
        let normalized = runtime.rms_norm(
            &hidden_states,
            self.weights
                .global(LagunaGlobalTensorRole::FinalNormalization)?,
            rms_norm_epsilon,
        )?;
        // Autoregressive serving needs only the final prompt position to choose
        // the next token. Slice before the large vocabulary projection so a
        // multi-token prefill does not compute and materialize unused logits
        // for every earlier prompt position.
        let terminal_normalized = last_token_hidden_states(runtime, &normalized)?;
        Ok(terminal_normalized)
    }

    /// Selects the highest-logit vocabulary row on the GPU and copies one token ID.
    pub fn highest_logit_token_id(
        runtime: &MlxRuntime,
        logits: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u32, LagunaExecutionError> {
        let last_token_logits = last_token_vocabulary_logits(runtime, logits)?;
        let selected_token = performance_attribution
            .measure_operation(PerformanceOperation::TokenSamplingGraphConstruction, |_| {
                runtime.argmax_axis(&last_token_logits, 0)
            })?;
        performance_attribution
            .measure_operation(
                PerformanceOperation::GeneratedTokenItemSynchronizationWait,
                |_| selected_token.item_u32(),
            )
            .map_err(LagunaExecutionError::from)
    }
}

fn token_count_from_hidden_states(hidden_states: &MlxArray) -> Result<i32, LagunaExecutionError> {
    let hidden_shape = hidden_states.shape();
    if hidden_shape.len() != 3 || hidden_shape[1] <= 0 {
        return Err(LagunaExecutionError::invalid_geometry(
            "Laguna hidden states must have rank three and a positive token count",
        ));
    }
    Ok(hidden_shape[1])
}

fn last_token_hidden_states(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
) -> Result<MlxArray, LagunaExecutionError> {
    let hidden_shape = hidden_states.shape();
    if hidden_shape.len() != 3 || hidden_shape[1] <= 0 || hidden_shape[2] <= 0 {
        return Err(LagunaExecutionError::invalid_geometry(
            "Laguna terminal hidden states require [batch, tokens, hidden] geometry",
        ));
    }
    let token_count = hidden_shape[1];
    runtime
        .slice(
            hidden_states,
            &[0, token_count - 1, 0],
            &[hidden_shape[0], token_count, hidden_shape[2]],
            &[1, 1, 1],
        )
        .map_err(LagunaExecutionError::from)
}

pub(in crate::laguna) fn last_token_vocabulary_logits(
    runtime: &MlxRuntime,
    logits: &MlxArray,
) -> Result<MlxArray, LagunaExecutionError> {
    let logit_shape = logits.shape();
    let vocabulary_size = *logit_shape.last().unwrap_or(&0);
    if logit_shape.len() < 2 || vocabulary_size <= 0 {
        return Err(LagunaExecutionError::invalid_geometry(
            "Laguna logits are missing a vocabulary axis",
        ));
    }
    let token_count = logit_shape[logit_shape.len() - 2];
    let last_token_start = token_count.saturating_sub(1);
    let last_token_logits = if logit_shape.len() == 3 {
        runtime.slice(
            logits,
            &[0, last_token_start, 0],
            &[1, token_count, vocabulary_size],
            &[1, 1, 1],
        )?
    } else {
        runtime.slice(
            logits,
            &[last_token_start, 0],
            &[token_count, vocabulary_size],
            &[1, 1],
        )?
    };
    Ok(runtime.reshape(&last_token_logits, &[vocabulary_size])?)
}

fn token_count_from_token_ids(token_ids: &MlxArray) -> Result<usize, LagunaExecutionError> {
    let token_shape = token_ids.shape();
    let token_axis = token_shape.last().copied().unwrap_or(0);
    usize::try_from(token_axis)
        .map_err(|_| LagunaExecutionError::invalid_geometry("token count exceeds the usize range"))
}
