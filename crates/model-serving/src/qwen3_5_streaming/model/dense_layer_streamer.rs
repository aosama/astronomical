//! Dense decoder layers kept in memory or read from the SSD per forward.
//!
//! A dense model that does not fit its limit keeps a prefix of its decoder
//! layers resident and reads the rest for every forward (issue #1985). The
//! streamer owns every decoder layer of such a model: kept layers are read
//! once through the bounded-range reader and evaluated, and streamed layers
//! are read again each time they run. Layers are never bound from the shard
//! maps for execution, because a map entry would pin an evaluated layer's
//! bytes for the model's lifetime.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::rc::Rc;

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxArray;

use super::dense_layer_reads::{
    DenseLayerShardRead, DenseLayerTensorRange, dense_layer_shard_reads,
};
use crate::TensorProfile;
use crate::artifact_validation::TensorSemanticRole;
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::streamed_layer_binding::{
    bind_decoder_layer_from_tensors, decoder_layer_index_of, decoder_layer_tensor_profiles,
};

/// Layers a streamed forward holds at once: the one computing and the one
/// being read for it next. A structural count, not a memory amount; the
/// bytes come from the measured layer sizes.
pub(crate) const DENSE_STREAMING_LAYER_SLOT_COUNT: u64 = 2;

/// The decoder layers of a dense model the streaming engine serves.
#[derive(Debug)]
pub(crate) struct Qwen3_5DenseLayerStreamer {
    /// Duplicated shard descriptors by model shard position; `None` for a
    /// shard that holds no decoder layer tensor.
    shard_files: Vec<Option<File>>,
    layer_shard_reads: Vec<Vec<DenseLayerShardRead>>,
    layer_tensor_profiles: Vec<Vec<TensorProfile>>,
    layer_payload_bytes: Vec<u64>,
    /// Layers `0..kept_layer_weights.len()` stay resident. Interior
    /// mutability lets request admission give kept layers back, so they
    /// stream, when a request needs their room.
    kept_layer_weights: RefCell<Vec<Rc<Qwen3_5DecoderLayerWeights>>>,
    /// Working memory the load-time probe measured, reused when a live
    /// limit change plans again; zero when the probe could not run.
    measured_working_memory_bytes: Cell<u64>,
}

/// Where a forward finds one decoder layer.
pub(crate) enum DenseLayerWeights {
    Kept(Rc<Qwen3_5DecoderLayerWeights>),
    Streamed(Box<Qwen3_5DecoderLayerWeights>),
}

impl DenseLayerWeights {
    pub(crate) fn decoder_layer_weights(&self) -> &Qwen3_5DecoderLayerWeights {
        match self {
            Self::Kept(kept_layer_weights) => kept_layer_weights,
            Self::Streamed(streamed_layer_weights) => streamed_layer_weights,
        }
    }

    /// Whether the forward must submit after this layer so its read bytes
    /// free before the next streamed layer arrives.
    pub(crate) const fn is_streamed(&self) -> bool {
        matches!(self, Self::Streamed(_))
    }
}

impl Qwen3_5DenseLayerStreamer {
    /// Plans every decoder layer's reads from the validated headers and
    /// duplicates the shard descriptors, before the engine takes the sources.
    pub(crate) fn from_validated_artifact(
        validated_artifact: &ValidatedQwen3_5Artifact,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let qwen3_5_config = validated_artifact.config();
        let layer_count = qwen3_5_config.layer_count() as usize;
        let shard_position_by_source_id = shard_position_by_source_id(validated_artifact)?;
        let mut shard_files = Vec::new();
        shard_files.resize_with(shard_position_by_source_id.len(), Option::default);
        let mut tensor_ranges_by_layer_then_shard: Vec<
            BTreeMap<usize, Vec<DenseLayerTensorRange>>,
        > = vec![BTreeMap::new(); layer_count];
        for tensor_location in validated_artifact.tensor_inventory().locations() {
            if tensor_location.semantic_role() != TensorSemanticRole::Target {
                continue;
            }
            let Some(layer_index) = decoder_layer_index_of(tensor_location.canonical_name()) else {
                continue;
            };
            let (tensor_range, shard_position) = located_layer_tensor(
                validated_artifact,
                tensor_location.canonical_name(),
                &shard_position_by_source_id,
                &mut shard_files,
            )?;
            tensor_ranges_by_layer_then_shard
                .get_mut(layer_index)
                .ok_or_else(|| Qwen3_5ExecutionError::MissingDecoderLayerWeights { layer_index })?
                .entry(shard_position)
                .or_default()
                .push(tensor_range);
        }
        Self::from_layer_ranges(
            qwen3_5_config,
            shard_files,
            tensor_ranges_by_layer_then_shard,
        )
    }

    fn from_layer_ranges(
        qwen3_5_config: &Qwen3_5Config,
        shard_files: Vec<Option<File>>,
        tensor_ranges_by_layer_then_shard: Vec<BTreeMap<usize, Vec<DenseLayerTensorRange>>>,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let mut layer_shard_reads = Vec::with_capacity(tensor_ranges_by_layer_then_shard.len());
        let mut layer_payload_bytes = Vec::with_capacity(tensor_ranges_by_layer_then_shard.len());
        for layer_ranges in tensor_ranges_by_layer_then_shard {
            let shard_reads = dense_layer_shard_reads(layer_ranges);
            let mut payload_bytes = 0_u64;
            for shard_read in &shard_reads {
                payload_bytes = payload_bytes.saturating_add(shard_read.payload_bytes);
            }
            layer_payload_bytes.push(payload_bytes);
            layer_shard_reads.push(shard_reads);
        }
        Ok(Self {
            shard_files,
            layer_shard_reads,
            layer_tensor_profiles: decoder_layer_tensor_profiles(qwen3_5_config)?,
            layer_payload_bytes,
            kept_layer_weights: RefCell::new(Vec::new()),
            measured_working_memory_bytes: Cell::new(0),
        })
    }

    /// Every decoder layer's payload, in layer order.
    pub(crate) fn layer_payload_bytes(&self) -> &[u64] {
        &self.layer_payload_bytes
    }

    /// How many leading layers stay resident.
    pub(crate) fn kept_layer_count(&self) -> usize {
        self.kept_layer_weights.borrow().len()
    }

    /// Payload of the layers kept resident.
    pub(crate) fn kept_layer_payload_bytes(&self) -> u64 {
        let mut kept_layer_payload_bytes = 0_u64;
        for layer_payload_bytes in &self.layer_payload_bytes[..self.kept_layer_count()] {
            kept_layer_payload_bytes =
                kept_layer_payload_bytes.saturating_add(*layer_payload_bytes);
        }
        kept_layer_payload_bytes
    }

    pub(crate) fn record_measured_working_memory_bytes(&self, working_memory_bytes: u64) {
        self.measured_working_memory_bytes.set(working_memory_bytes);
    }

    pub(crate) fn measured_working_memory_bytes(&self) -> u64 {
        self.measured_working_memory_bytes.get()
    }

    /// Bytes the layer slots occupy while layers stream.
    pub(crate) fn streaming_slot_bytes(&self) -> u64 {
        self.largest_layer_payload_bytes()
            .saturating_mul(DENSE_STREAMING_LAYER_SLOT_COUNT)
    }

    /// The bytes one streamed layer occupies while it is read and used.
    pub(crate) fn largest_layer_payload_bytes(&self) -> u64 {
        let mut largest_layer_payload_bytes = 0_u64;
        for layer_payload_bytes in &self.layer_payload_bytes {
            largest_layer_payload_bytes = largest_layer_payload_bytes.max(*layer_payload_bytes);
        }
        largest_layer_payload_bytes
    }

    /// Gives back kept layers from the last one down until at least
    /// `required_bytes` are freed or none are kept, returning the bytes freed.
    /// Those layers stream from then on.
    pub(crate) fn release_trailing_kept_layers(&self, required_bytes: u64) -> u64 {
        let mut kept_layer_weights = self.kept_layer_weights.borrow_mut();
        let mut released_bytes = 0_u64;
        while released_bytes < required_bytes && !kept_layer_weights.is_empty() {
            let released_layer_index = kept_layer_weights.len() - 1;
            kept_layer_weights.pop();
            released_bytes =
                released_bytes.saturating_add(self.layer_payload_bytes[released_layer_index]);
        }
        released_bytes
    }

    /// Payload of the layers read per forward.
    pub(crate) fn streamed_layer_payload_bytes(&self) -> u64 {
        let mut streamed_layer_payload_bytes = 0_u64;
        for layer_payload_bytes in &self.layer_payload_bytes[self.kept_layer_count()..] {
            streamed_layer_payload_bytes =
                streamed_layer_payload_bytes.saturating_add(*layer_payload_bytes);
        }
        streamed_layer_payload_bytes
    }

    /// The weights a forward uses for `layer_index`: the kept copy, or a
    /// fresh lazy read that frees its bytes once the forward completes.
    pub(crate) fn layer_weights(
        &self,
        runtime: &MlxRuntime,
        qwen3_5_config: &Qwen3_5Config,
        raw_hugging_face_norm_offsets_apply: bool,
        layer_index: usize,
    ) -> Result<DenseLayerWeights, Qwen3_5ExecutionError> {
        if let Some(kept_layer_weights) = self.kept_layer_weights.borrow().get(layer_index) {
            return Ok(DenseLayerWeights::Kept(Rc::clone(kept_layer_weights)));
        }
        Ok(DenseLayerWeights::Streamed(Box::new(self.read_layer(
            runtime,
            qwen3_5_config,
            raw_hugging_face_norm_offsets_apply,
            layer_index,
        )?)))
    }

    /// Makes layers `0..kept_layer_count` resident, reading and evaluating
    /// any not yet kept and dropping any beyond the new count.
    pub(crate) fn keep_layers(
        &self,
        runtime: &MlxRuntime,
        qwen3_5_config: &Qwen3_5Config,
        raw_hugging_face_norm_offsets_apply: bool,
        kept_layer_count: usize,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let kept_layer_count = kept_layer_count.min(self.layer_payload_bytes.len());
        self.kept_layer_weights
            .borrow_mut()
            .truncate(kept_layer_count);
        while self.kept_layer_count() < kept_layer_count {
            let layer_weights = self.read_layer(
                runtime,
                qwen3_5_config,
                raw_hugging_face_norm_offsets_apply,
                self.kept_layer_count(),
            )?;
            let mut layer_arrays = Vec::new();
            layer_weights.append_array_references(&mut layer_arrays);
            runtime.evaluate_arrays(&layer_arrays)?;
            self.kept_layer_weights
                .borrow_mut()
                .push(Rc::new(layer_weights));
        }
        Ok(())
    }

    /// Reads one layer's tensors through bounded loads and binds them.
    pub(crate) fn read_layer(
        &self,
        runtime: &MlxRuntime,
        qwen3_5_config: &Qwen3_5Config,
        raw_hugging_face_norm_offsets_apply: bool,
        layer_index: usize,
    ) -> Result<Qwen3_5DecoderLayerWeights, Qwen3_5ExecutionError> {
        let shard_reads = self
            .layer_shard_reads
            .get(layer_index)
            .ok_or(Qwen3_5ExecutionError::MissingDecoderLayerWeights { layer_index })?;
        let mut layer_tensors = HashMap::new();
        for shard_read in shard_reads {
            self.read_shard_tensors(runtime, shard_read, &mut layer_tensors)?;
        }
        bind_decoder_layer_from_tensors(
            runtime,
            qwen3_5_config,
            layer_index,
            &self.layer_tensor_profiles[layer_index],
            raw_hugging_face_norm_offsets_apply,
            layer_tensors,
        )
    }

    fn read_shard_tensors(
        &self,
        runtime: &MlxRuntime,
        shard_read: &DenseLayerShardRead,
        layer_tensors: &mut HashMap<String, MlxArray>,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let shard_file = self
            .shard_files
            .get(shard_read.shard_position)
            .and_then(Option::as_ref)
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "a dense layer read names a shard without a retained descriptor",
            })?;
        let duplicated_shard_file = shard_file.try_clone().map_err(|source| {
            Qwen3_5ExecutionError::WeightSourceDescriptor {
                description: "could not duplicate a dense layer shard descriptor",
                source,
            }
        })?;
        let layer_load = runtime.load_safetensors_from_bounded_ranges(
            duplicated_shard_file,
            shard_read.synthetic_header_bytes.clone(),
            shard_read.read_intervals.clone(),
            shard_read.payload_bytes,
            None,
        )?;
        for canonical_name in &shard_read.tensor_names {
            let tensor = layer_load.tensor(canonical_name)?;
            layer_tensors.insert(canonical_name.clone(), tensor);
        }
        Ok(())
    }
}

/// Model shard position of every validated source.
fn shard_position_by_source_id(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<HashMap<crate::artifact_validation::TensorSourceId, usize>, Qwen3_5ExecutionError> {
    let model_shard_source_ids = validated_artifact
        .source_ids_for_file_names(validated_artifact.shard_index().model_shard_file_names())?;
    let mut shard_position_by_source_id = HashMap::with_capacity(model_shard_source_ids.len());
    for (shard_position, source_id) in model_shard_source_ids.into_iter().enumerate() {
        shard_position_by_source_id.insert(source_id, shard_position);
    }
    Ok(shard_position_by_source_id)
}

/// One layer tensor's range and shard position, duplicating that shard's
/// descriptor the first time a tensor needs it.
fn located_layer_tensor(
    validated_artifact: &ValidatedQwen3_5Artifact,
    canonical_name: &str,
    shard_position_by_source_id: &HashMap<crate::artifact_validation::TensorSourceId, usize>,
    shard_files: &mut [Option<File>],
) -> Result<(DenseLayerTensorRange, usize), Qwen3_5ExecutionError> {
    let (tensor_file_range, safetensors_source) = validated_artifact
        .tensor_file_range(canonical_name)
        .ok_or_else(|| missing_layer_tensor(canonical_name))?;
    let source_id = validated_artifact
        .tensor_inventory()
        .location(canonical_name)
        .ok_or_else(|| missing_layer_tensor(canonical_name))?
        .source_id();
    let shard_position = *shard_position_by_source_id
        .get(&source_id)
        .ok_or_else(|| missing_layer_tensor(canonical_name))?;
    let shard_file_slot = shard_files
        .get_mut(shard_position)
        .ok_or_else(|| missing_layer_tensor(canonical_name))?;
    if shard_file_slot.is_none() {
        *shard_file_slot = Some(safetensors_source.duplicate_file().map_err(|source| {
            Qwen3_5ExecutionError::WeightSourceDescriptor {
                description: "could not duplicate a dense layer shard descriptor",
                source,
            }
        })?);
    }
    Ok((
        DenseLayerTensorRange {
            canonical_name: canonical_name.to_owned(),
            stored_dtype: tensor_file_range.dtype.to_owned(),
            shape: tensor_file_range.shape.to_vec(),
            file_start_offset: tensor_file_range.file_start_offset,
            file_end_offset: tensor_file_range.file_end_offset,
        },
        shard_position,
    ))
}

fn missing_layer_tensor(canonical_name: &str) -> Qwen3_5ExecutionError {
    Qwen3_5ExecutionError::MissingTensor {
        tensor_name: canonical_name.to_owned(),
    }
}
