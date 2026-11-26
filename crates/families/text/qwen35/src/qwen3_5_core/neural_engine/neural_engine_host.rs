//! What Neural Engine prompt reading asks of whichever engine holds a dense
//! model (issue #2585).
//!
//! The programs are built from each decoder layer's weights, read one layer
//! at a time, and attach to the model base. An engine that binds every layer
//! in the base answers from there. An engine that keeps its dense layers in
//! a store of its own answers from that store, and offers them only while
//! every layer is held in memory: a layer read from the SSD for one forward
//! has no weights to compile a program from between requests.

use std::path::{Path, PathBuf};

use crate::qwen3_5_core::model::base::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5DecoderFeedForwardWeights, Qwen3_5DecoderLayerWeights,
};

/// The read failure every caller reports the same way.
pub const LAYER_READ_FAILURE: &str = "the model has no such decoder layer";

/// What reads one decoder layer's weights.
pub type LayerReader<'reader> = dyn FnMut(&Qwen3_5DecoderLayerWeights) + 'reader;
/// Whether a layer could be read, and if not why.
pub type LayerRead = Result<(), &'static str>;

pub trait NeuralEngineHost {
    fn model_base(&self) -> &Qwen3_5ModelBase;

    fn model_base_mut(&mut self) -> &mut Qwen3_5ModelBase;

    /// How many decoder layers the model has, when every one is a dense
    /// layer held in memory; `None` for any other model.
    fn dense_layer_count_in_memory(&self) -> Option<usize>;

    /// The model's weights held in memory now, which the programs must fit
    /// beside under the MLX memory ceiling.
    fn model_payload_bytes_in_memory(&self) -> u64;

    /// Hands one decoder layer's weights to `read`.
    ///
    /// # Errors
    /// When the layer does not exist or cannot be had in memory.
    fn read_dense_layer(&self, layer_index: usize, read: &mut LayerReader<'_>) -> LayerRead;
}

/// A model whose decoder layers are all bound in its base.
impl NeuralEngineHost for Qwen3_5ModelBase {
    fn model_base(&self) -> &Qwen3_5ModelBase {
        self
    }

    fn model_base_mut(&mut self) -> &mut Qwen3_5ModelBase {
        self
    }

    fn dense_layer_count_in_memory(&self) -> Option<usize> {
        let bound_layers = &self.weights.decoder_layer_weights;
        let mut is_every_layer_dense = !bound_layers.is_empty();
        for decoder_layer_weights in bound_layers {
            is_every_layer_dense &= matches!(
                decoder_layer_weights.mlp_weights,
                Qwen3_5DecoderFeedForwardWeights::Dense(_)
            );
        }
        is_every_layer_dense.then_some(bound_layers.len())
    }

    fn model_payload_bytes_in_memory(&self) -> u64 {
        self.resident_model_payload_byte_count()
    }

    fn read_dense_layer(&self, layer_index: usize, read: &mut LayerReader<'_>) -> LayerRead {
        let decoder_layer_weights = self
            .weights
            .decoder_layer_weights
            .get(layer_index)
            .ok_or(LAYER_READ_FAILURE)?;
        read(decoder_layer_weights);
        Ok(())
    }
}

/// One program's compile input to write: from which layer of which model,
/// into which folder.
pub struct LayerPackageRequest<'request> {
    pub host: &'request dyn NeuralEngineHost,
    pub decoder_layer_index: usize,
    pub package_directory: &'request Path,
}

/// Where a package was written, or why it was not.
pub type PackageWritten = Result<PathBuf, &'static str>;

impl LayerPackageRequest<'_> {
    /// Writes the package from the layer's weights. `write_package` answers
    /// `None` when the layer lacks the part the program is made from, which
    /// is reported as `layer_lacks_the_part`.
    ///
    /// # Errors
    /// When the layer cannot be read, lacks the part, or the package cannot
    /// be written.
    pub fn write_from_layer(
        &self,
        layer_lacks_the_part: &'static str,
        write_package: &mut dyn FnMut(&Qwen3_5DecoderLayerWeights) -> Option<PackageWritten>,
    ) -> PackageWritten {
        let mut written_package = Err(layer_lacks_the_part);
        self.host
            .read_dense_layer(self.decoder_layer_index, &mut |decoder_layer_weights| {
                if let Some(package_outcome) = write_package(decoder_layer_weights) {
                    written_package = package_outcome;
                }
            })?;
        written_package
    }
}
