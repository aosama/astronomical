//! Small real safetensors artifacts for the tests, written to a temporary folder.

use std::fs;
use std::path::Path;

use crate::artifact_reference::ArtifactReference;
use crate::artifact_source::ArtifactSource;

/// One tensor for a fixture file, as bfloat16.
pub struct FixtureTensor<'name> {
    pub name: &'name str,
    pub float_values: Vec<f32>,
}

/// Varied values fixed by the seed, so two fixtures match only when they were
/// given the same seed.
pub fn seeded_values(seed: u32, value_count: usize) -> Vec<f32> {
    let mut float_values = Vec::with_capacity(value_count);
    for value_index in 0..value_count {
        let mixed = (seed + 1) * 31 + value_index as u32 * 7;
        float_values.push((mixed % 101) as f32 / 10.0 + 0.25);
    }
    float_values
}

pub fn write_bfloat16_file(file_path: &Path, tensors: &[FixtureTensor<'_>]) {
    let mut shaped_tensors = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        shaped_tensors.push((
            tensor.name,
            vec![tensor.float_values.len() as u64],
            tensor.float_values.as_slice(),
        ));
    }
    write_shaped_bfloat16_file(file_path, &shaped_tensors);
}

/// Tensors as `(name, shape, values)`, for layouts where the shape itself matters.
pub fn write_shaped_bfloat16_file(file_path: &Path, tensors: &[(&str, Vec<u64>, &[f32])]) {
    let mut header_entries = serde_json::Map::new();
    let mut tensor_bytes = Vec::new();
    for (tensor_name, tensor_shape, float_values) in tensors {
        let start_offset = tensor_bytes.len();
        for float_value in *float_values {
            let bfloat16_bits = (float_value.to_bits() >> 16) as u16;
            tensor_bytes.extend_from_slice(&bfloat16_bits.to_le_bytes());
        }
        header_entries.insert(
            (*tensor_name).to_owned(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": tensor_shape,
                "data_offsets": [start_offset, tensor_bytes.len()],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&header_entries).unwrap();
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.extend_from_slice(&tensor_bytes);
    if let Some(parent_directory) = file_path.parent() {
        fs::create_dir_all(parent_directory).unwrap();
    }
    fs::write(file_path, file_bytes).unwrap();
}

pub async fn open_directory(directory: &Path) -> ArtifactSource {
    ArtifactSource::open(&ArtifactReference::LocalDirectory(directory.to_path_buf()))
        .await
        .unwrap()
}
