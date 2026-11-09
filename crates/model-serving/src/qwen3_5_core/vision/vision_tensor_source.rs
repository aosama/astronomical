//! Reopenable files behind the vision tower's tensors.
//!
//! An evaluated MLX array keeps its bytes for as long as any handle to it
//! lives, and a SafeTensors map entry is such a handle. Releasing the tower
//! after an image request therefore cannot reuse the map that produced the
//! evaluated arrays: it must load the same files again, which only parses the
//! headers, and bind fresh lazy arrays. This module keeps the descriptors that
//! make that reload possible without reopening paths, so the bytes served
//! later still come from the file identity validated at load time.

use std::collections::HashMap;
use std::fs::File;

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};

use super::{EmbeddedVisionTensorLocation, Qwen3_5ExecutionError};

/// The files holding the vision tensors and where each tensor lives in them.
#[derive(Debug)]
pub(crate) struct Qwen3_5VisionTensorSource {
    source_files: Vec<File>,
    vision_tensor_name_to_location: HashMap<String, EmbeddedVisionTensorLocation>,
}

impl Qwen3_5VisionTensorSource {
    /// Keeps `source_files` in the order the locations' `shard_position`
    /// values index.
    pub(crate) fn new(
        source_files: Vec<File>,
        vision_tensor_name_to_location: HashMap<String, EmbeddedVisionTensorLocation>,
    ) -> Self {
        Self {
            source_files,
            vision_tensor_name_to_location,
        }
    }

    /// A source with no files, for weights built directly in tests.
    #[cfg(test)]
    pub(crate) fn without_files() -> Self {
        Self::new(Vec::new(), HashMap::new())
    }

    /// Loads every source file as a new lazy SafeTensors map.
    ///
    /// Each load reads a duplicated descriptor, so the retained descriptors
    /// stay usable for the next release.
    pub(crate) fn load_lazy_safetensors(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<Vec<MlxSafetensors>, Qwen3_5ExecutionError> {
        let mut source_safetensors = Vec::with_capacity(self.source_files.len());
        for source_file in &self.source_files {
            let duplicated_source_file = source_file.try_clone().map_err(|source| {
                Qwen3_5ExecutionError::WeightSourceDescriptor {
                    description: "could not duplicate a vision weight file descriptor",
                    source,
                }
            })?;
            source_safetensors.push(runtime.load_safetensors(duplicated_source_file, None)?);
        }
        Ok(source_safetensors)
    }

    /// Returns where one canonical vision tensor is stored.
    pub(crate) fn location(
        &self,
        vision_tensor_name: &str,
    ) -> Result<&EmbeddedVisionTensorLocation, Qwen3_5ExecutionError> {
        self.vision_tensor_name_to_location
            .get(vision_tensor_name)
            .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                tensor_name: vision_tensor_name.to_owned(),
            })
    }
}
