//! What one Flash-Next request remembers beyond its layers' attention and
//! recurrent state: the n-gram history and convolution window, and every
//! full-attention layer's indexer keys.
//!
//! It lives in the request decoder state's family slot, so it is
//! checkpointed and restored with the rest of the request whenever a prompt
//! chunk is retried, and it counts toward the request's memory.

use std::any::Any;

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntimeError;
use text_generation::RequestDecoderStateStack;
use text_generation::decoder_state::family_decoder_state::FamilyDecoderState;
use text_generation::execution_error::TextExecutionError;

use super::growing_sequence::GrowingSequence;

/// One full-attention layer's indexer memory.
pub struct IndexerLayerState {
    /// Raw indexer keys of every token so far, `[1, tokens, d]`.
    pub keys: GrowingSequence,
    /// Normalized, rotated mean keys of the complete blocks computed so
    /// far, `[1, 1, blocks, d]`; only built once a query passes the budget.
    pub block_keys: Option<GrowingSequence>,
}

impl IndexerLayerState {
    fn retained(&self) -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            keys: self.keys.retained()?,
            block_keys: self
                .block_keys
                .as_ref()
                .map(GrowingSequence::retained)
                .transpose()?,
        })
    }

    fn append_storage<'state>(&'state self, arrays: &mut Vec<&'state MlxArray>) {
        arrays.push(self.keys.storage());
        if let Some(block_keys) = &self.block_keys {
            arrays.push(block_keys.storage());
        }
    }
}

/// Flash-Next's per-request memory.
pub struct Qwen4ExpRequestState {
    /// The last tokens of history, which the next forward's first n-grams
    /// reach back into.
    pub ngram_context: Vec<u32>,
    /// The n-gram layer's last convolution inputs, `[1, window, lane width]`;
    /// `None` until the first forward.
    pub ngram_convolution_window: Option<MlxArray>,
    /// Per decoder layer: the indexer's memory, on full-attention layers.
    pub indexer_by_layer: Vec<Option<IndexerLayerState>>,
    /// The token ids of the forward in progress, for the n-gram layer.
    pub forward_token_ids: Vec<u32>,
}

impl Qwen4ExpRequestState {
    #[must_use]
    pub fn new(layer_count: usize, ngram_context: Vec<u32>) -> Self {
        Self {
            ngram_context,
            ngram_convolution_window: None,
            indexer_by_layer: empty_layer_slots(layer_count),
            forward_token_ids: Vec::new(),
        }
    }

    /// The installed state of `request_decoder_state`, installing a fresh
    /// one that starts from `initial_ngram_context` on a request's first
    /// forward.
    pub fn installed<'stack>(
        request_decoder_state: &'stack mut RequestDecoderStateStack,
        initial_ngram_context: &[u32],
    ) -> Result<&'stack mut Self, TextExecutionError> {
        if request_decoder_state.family_state::<Self>().is_none() {
            let layer_count = request_decoder_state.layer_count();
            request_decoder_state.install_family_state(Box::new(Self::new(
                layer_count,
                initial_ngram_context.to_vec(),
            )));
        }
        request_decoder_state
            .family_state_mut::<Self>()
            .ok_or(TextExecutionError::InvalidInput {
                description: "the request decoder state holds another family's state",
            })
    }
}

impl FamilyDecoderState for Qwen4ExpRequestState {
    fn payload_byte_count(&self) -> u64 {
        let window_bytes = self
            .ngram_convolution_window
            .as_ref()
            .map_or(0, MlxArray::byte_count);
        let mut indexer_storage = Vec::new();
        for indexer in self.indexer_by_layer.iter().flatten() {
            indexer.append_storage(&mut indexer_storage);
        }
        let mut key_bytes = 0_usize;
        for storage in indexer_storage {
            key_bytes += storage.byte_count();
        }
        u64::try_from(window_bytes + key_bytes).unwrap_or(u64::MAX)
    }

    fn append_evaluation_arrays<'state>(&'state self, arrays: &mut Vec<&'state MlxArray>) {
        arrays.extend(&self.ngram_convolution_window);
        for indexer in self.indexer_by_layer.iter().flatten() {
            indexer.append_storage(arrays);
        }
    }

    fn snapshot(&self) -> Result<Box<dyn FamilyDecoderState>, MlxRuntimeError> {
        let mut indexer_by_layer = Vec::with_capacity(self.indexer_by_layer.len());
        for indexer in &self.indexer_by_layer {
            indexer_by_layer.push(
                indexer
                    .as_ref()
                    .map(IndexerLayerState::retained)
                    .transpose()?,
            );
        }
        Ok(Box::new(Self {
            ngram_context: self.ngram_context.clone(),
            ngram_convolution_window: self
                .ngram_convolution_window
                .as_ref()
                .map(MlxArray::retain)
                .transpose()?,
            indexer_by_layer,
            forward_token_ids: self.forward_token_ids.clone(),
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn empty_layer_slots(layer_count: usize) -> Vec<Option<IndexerLayerState>> {
    let mut slots = Vec::with_capacity(layer_count);
    slots.resize_with(layer_count, Option::default);
    slots
}

impl crate::qwen3_5_core::model::Qwen3_5ModelBase {
    /// The token history a fresh request starts its n-gram hashing from.
    #[must_use]
    pub fn initial_ngram_context(&self) -> Vec<u32> {
        self.ngram_table
            .as_ref()
            .map(|ngram_table| ngram_table.hash_tables().initial_context())
            .unwrap_or_default()
    }
}

/// One layer's validated state and, for Flash-Next, the request's
/// Flash-Next state, borrowed together for one layer forward.
pub fn layer_and_qwen4_exp_state(
    request_decoder_state: &mut RequestDecoderStateStack,
    layer_index: usize,
) -> Result<
    (
        &mut crate::AstroDecoderStateCacheState,
        Option<&mut Qwen4ExpRequestState>,
    ),
    TextExecutionError,
> {
    let (layer_model_state, qwen4_exp_state) = request_decoder_state
        .layer_and_family_state_mut::<Qwen4ExpRequestState>(layer_index)
        .ok_or(TextExecutionError::InvalidRequestDecoderState {
            layer_index,
            description: "decoder layer entry is missing",
        })?;
    if !layer_model_state.tensors_are_allocated_consistently() {
        return Err(
            text_generation::execution_error::invalid_request_decoder_state(
                layer_index,
                "decoder layer tensors must be allocated together or absent together",
            ),
        );
    }
    Ok((layer_model_state, qwen4_exp_state))
}

impl crate::qwen3_5_core::model::Qwen3_5ModelBase {
    /// Records the token ids of the forward about to run, for the n-gram
    /// layer; installs the Flash-Next state on a request's first forward.
    /// Nothing to do for a Qwen 3.5 model.
    pub fn record_qwen4_exp_forward_tokens(
        &self,
        request_decoder_state: &mut RequestDecoderStateStack,
        token_ids: Vec<u32>,
    ) -> Result<(), TextExecutionError> {
        if self.config.qwen4_exp_architecture().is_none() {
            return Ok(());
        }
        let initial_context = self.initial_ngram_context();
        Qwen4ExpRequestState::installed(request_decoder_state, &initial_context)?
            .forward_token_ids = token_ids;
        Ok(())
    }

    /// The first decoder layer's input for `token_indices`: their
    /// embeddings, or for Flash-Next the lanes they start, after recording
    /// the tokens for its n-gram layer.
    pub fn embed_forward_tokens(
        &self,
        token_indices: &MlxArray,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, TextExecutionError> {
        self.record_qwen4_exp_forward_token_indices(request_decoder_state, token_indices)?;
        self.decoder_layer_input(self.context_free_token_meanings(token_indices)?)
    }

    /// One layer's state and Flash-Next state for a forward over
    /// `token_ids`, recording the tokens first on a Flash-Next model; used
    /// where each layer runs over one span at a time.
    pub fn layer_states_for_tokens<'stack>(
        &self,
        request_decoder_state: &'stack mut RequestDecoderStateStack,
        layer_index: usize,
        token_ids: &[u32],
    ) -> Result<
        (
            &'stack mut crate::AstroDecoderStateCacheState,
            Option<&'stack mut Qwen4ExpRequestState>,
        ),
        TextExecutionError,
    > {
        if self.config.qwen4_exp_architecture().is_some() {
            self.record_qwen4_exp_forward_tokens(request_decoder_state, token_ids.to_vec())?;
        }
        layer_and_qwen4_exp_state(request_decoder_state, layer_index)
    }

    /// [`Self::record_qwen4_exp_forward_tokens`] for token indices still on
    /// the GPU: reads them back, which a Flash-Next forward needs anyway to
    /// address its n-gram rows.
    pub fn record_qwen4_exp_forward_token_indices(
        &self,
        request_decoder_state: &mut RequestDecoderStateStack,
        token_indices: &MlxArray,
    ) -> Result<(), TextExecutionError> {
        if self.config.qwen4_exp_architecture().is_none() {
            return Ok(());
        }
        self.runtime.evaluate_arrays(&[token_indices])?;
        let mut token_ids = Vec::new();
        for token_id in token_indices.to_vec_i32()? {
            token_ids.push(u32::try_from(token_id).map_err(|_| {
                TextExecutionError::InvalidInput {
                    description: "a token id is negative",
                }
            })?);
        }
        self.record_qwen4_exp_forward_tokens(request_decoder_state, token_ids)
    }
}
