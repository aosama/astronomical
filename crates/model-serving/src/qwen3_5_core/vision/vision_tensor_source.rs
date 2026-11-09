//! Reopenable files behind the vision tower's tensors.
//!
//! An evaluated MLX array keeps its bytes for as long as any handle to it
//! lives, and a SafeTensors map entry is such a handle. Releasing the tower
//! after an image request therefore cannot reuse the map that produced the
//! evaluated arrays: it must load the same files again, which only parses the
//! headers, and bind fresh lazy arrays. This module keeps the descriptors that
//! make that reload possible without reopening paths, so the bytes served
//! later still come from the file identity validated at load time.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};

use super::{Qwen3_5ExecutionError, VisionTensorLocation};

#[cfg(test)]
#[path = "vision_tensor_source_tests.rs"]
mod tests;

/// The files holding the vision tensors and where each tensor lives in them.
#[derive(Debug)]
pub(crate) struct Qwen3_5VisionTensorSource {
    source_files: Vec<File>,
    vision_tensor_name_to_location: HashMap<String, VisionTensorLocation>,
}

impl Qwen3_5VisionTensorSource {
    /// Keeps `source_files` in the order the locations'
    /// `source_file_position` values index.
    pub(crate) fn new(
        source_files: Vec<File>,
        vision_tensor_name_to_location: HashMap<String, VisionTensorLocation>,
    ) -> Self {
        Self {
            source_files,
            vision_tensor_name_to_location,
        }
    }

    /// Keeps only the files that hold a vision tensor and renumbers the
    /// locations to match.
    ///
    /// An embedded tower may sit in one shard of many; every file kept here is
    /// parsed again on each release, so the language-only shards are dropped
    /// and their duplicated descriptors close.
    pub(crate) fn keeping_referenced_files(
        source_files: Vec<File>,
        mut vision_tensor_name_to_location: HashMap<String, VisionTensorLocation>,
    ) -> Self {
        let mut original_to_kept_position = BTreeMap::new();
        for vision_tensor_location in vision_tensor_name_to_location.values() {
            // usize::MAX until a file fills the position: a location naming a
            // file that does not exist then stays out of range, so binding
            // reports it instead of silently reading another file.
            original_to_kept_position
                .insert(vision_tensor_location.source_file_position, usize::MAX);
        }
        let mut kept_source_files = Vec::with_capacity(original_to_kept_position.len());
        for (original_position, source_file) in source_files.into_iter().enumerate() {
            if let Some(kept_position) = original_to_kept_position.get_mut(&original_position) {
                *kept_position = kept_source_files.len();
                kept_source_files.push(source_file);
            }
        }
        for vision_tensor_location in vision_tensor_name_to_location.values_mut() {
            vision_tensor_location.source_file_position = original_to_kept_position
                .get(&vision_tensor_location.source_file_position)
                .copied()
                .unwrap_or(usize::MAX);
        }
        Self::new(kept_source_files, vision_tensor_name_to_location)
    }

    /// A source with no files, for weights built directly in tests.
    #[cfg(test)]
    pub(crate) fn without_files() -> Self {
        Self::new(Vec::new(), HashMap::new())
    }

    /// The number of files this source reads.
    #[cfg(test)]
    pub(crate) fn source_file_count(&self) -> usize {
        self.source_files.len()
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
    ) -> Result<&VisionTensorLocation, Qwen3_5ExecutionError> {
        self.vision_tensor_name_to_location
            .get(vision_tensor_name)
            .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                tensor_name: vision_tensor_name.to_owned(),
            })
    }
}
