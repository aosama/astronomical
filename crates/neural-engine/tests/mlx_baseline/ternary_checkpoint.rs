//! Reads one dense MLP layer of the folded ternary 27B checkpoint straight from
//! its safetensors file, so the journeys measure real weights instead of a
//! synthetic stand-in: their value distribution decides whether shared lookup
//! tables hold them, which is the question.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use astronomical_neural_engine::half_precision::{f16_bits_to_f32, f32_to_f16_bits};
use serde_json::Value;

const SNAPSHOTS_RELATIVE_TO_HOME: &str =
    ".cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots";
const WEIGHTS_FILE_NAME: &str = "model.safetensors";
const LAYER_TENSOR_PREFIX: &str = "language_model.model.layers";
const SAFETENSORS_LENGTH_PREFIX_BYTES: u64 = 8;
const CODE_BITS: usize = 2;
const CODE_MASK: u32 = 0b11;
const CODES_PER_PACKED_WORD: usize = 32 / CODE_BITS;
pub(super) const BLOCK_INPUT_COLUMNS: usize = 128;

/// One packed projection, `[rows][input channels]`, as the checkpoint stores it.
pub(super) struct TernaryProjection {
    pub(super) row_count: usize,
    pub(super) input_channels: usize,
    pub(super) packed_codes: Vec<u32>,
    pub(super) fp16_scales: Vec<u16>,
    pub(super) fp16_biases: Vec<u16>,
    pub(super) signs: Vec<f32>,
}

impl TernaryProjection {
    pub(super) fn blocks_per_row(&self) -> usize {
        self.input_channels / BLOCK_INPUT_COLUMNS
    }

    pub(super) fn packed_words_per_row(&self) -> usize {
        self.input_channels / CODES_PER_PACKED_WORD
    }

    /// `fp16(scale · code + bias)`, row-major.
    pub(super) fn fp16_weights(&self) -> Vec<u16> {
        let blocks_per_row = self.blocks_per_row();
        let words_per_row = self.packed_words_per_row();
        let mut fp16_weights = Vec::with_capacity(self.row_count * self.input_channels);
        for row_index in 0..self.row_count {
            for input_channel in 0..self.input_channels {
                let word = self.packed_codes
                    [row_index * words_per_row + input_channel / CODES_PER_PACKED_WORD];
                let code =
                    (word >> (CODE_BITS * (input_channel % CODES_PER_PACKED_WORD))) & CODE_MASK;
                let block_index = row_index * blocks_per_row + input_channel / BLOCK_INPUT_COLUMNS;
                let scale = f16_bits_to_f32(self.fp16_scales[block_index]);
                let bias = f16_bits_to_f32(self.fp16_biases[block_index]);
                fp16_weights.push(f32_to_f16_bits(scale * code as f32 + bias));
            }
        }
        fp16_weights
    }
}

/// One MLP layer: the gate, up, and down projections.
pub(super) struct TernaryMlpLayer {
    pub(super) gate: TernaryProjection,
    pub(super) up: TernaryProjection,
    pub(super) down: TernaryProjection,
}

/// The checkpoint's weights file with its parsed header.
pub(super) struct TernaryCheckpoint {
    weights_file: File,
    tensor_data_start: u64,
    header: Value,
}

impl TernaryCheckpoint {
    /// Opens the Hugging Face snapshot; the journey cannot run without it.
    pub(super) fn open() -> Self {
        let weights_path = weights_path();
        let mut weights_file = File::open(&weights_path)
            .expect("the 27B checkpoint must be downloaded for this journey");
        let mut length_prefix = [0_u8; SAFETENSORS_LENGTH_PREFIX_BYTES as usize];
        weights_file
            .read_exact(&mut length_prefix)
            .expect("the safetensors length prefix reads");
        let header_byte_count = u64::from_le_bytes(length_prefix);
        let mut header_bytes = vec![0_u8; header_byte_count as usize];
        weights_file
            .read_exact(&mut header_bytes)
            .expect("the safetensors header reads");
        Self {
            weights_file,
            tensor_data_start: SAFETENSORS_LENGTH_PREFIX_BYTES + header_byte_count,
            header: serde_json::from_slice(&header_bytes).expect("the header is JSON"),
        }
    }

    pub(super) fn mlp_layer(&mut self, layer_index: usize) -> TernaryMlpLayer {
        TernaryMlpLayer {
            gate: self.projection(layer_index, "gate_proj"),
            up: self.projection(layer_index, "up_proj"),
            down: self.projection(layer_index, "down_proj"),
        }
    }

    fn projection(&mut self, layer_index: usize, projection_name: &str) -> TernaryProjection {
        let prefix = format!("{LAYER_TENSOR_PREFIX}.{layer_index}.mlp.{projection_name}");
        let (packed_bytes, packed_shape) = self.tensor_bytes(&format!("{prefix}.weight"));
        let (scale_bytes, _) = self.tensor_bytes(&format!("{prefix}.scales"));
        let (bias_bytes, _) = self.tensor_bytes(&format!("{prefix}.biases"));
        let (sign_bytes, sign_shape) = self.tensor_bytes(&format!("{prefix}.signs"));
        TernaryProjection {
            row_count: packed_shape[0],
            input_channels: sign_shape[0],
            packed_codes: little_endian_u32s(&packed_bytes),
            fp16_scales: little_endian_u16s(&scale_bytes),
            fp16_biases: little_endian_u16s(&bias_bytes),
            signs: little_endian_f32s(&sign_bytes),
        }
    }

    fn tensor_bytes(&mut self, tensor_name: &str) -> (Vec<u8>, Vec<usize>) {
        let description = &self.header[tensor_name];
        let start_offset = description["data_offsets"][0]
            .as_u64()
            .expect("the tensor has a start offset");
        let end_offset = description["data_offsets"][1]
            .as_u64()
            .expect("the tensor has an end offset");
        let mut shape = Vec::new();
        for dimension in description["shape"]
            .as_array()
            .expect("the tensor has a shape")
        {
            shape.push(dimension.as_u64().expect("a dimension is a number") as usize);
        }
        let mut tensor_bytes = vec![0_u8; (end_offset - start_offset) as usize];
        self.weights_file
            .seek(SeekFrom::Start(self.tensor_data_start + start_offset))
            .expect("the tensor offset is seekable");
        self.weights_file
            .read_exact(&mut tensor_bytes)
            .expect("the tensor reads");
        (tensor_bytes, shape)
    }
}

fn weights_path() -> PathBuf {
    let home_directory = std::env::var_os("HOME").expect("HOME locates the model cache");
    let snapshots_directory = PathBuf::from(home_directory).join(SNAPSHOTS_RELATIVE_TO_HOME);
    let mut snapshot_entries = fs::read_dir(&snapshots_directory)
        .expect("the 27B checkpoint must be downloaded for this journey");
    let first_snapshot = snapshot_entries
        .next()
        .expect("the 27B checkpoint has a snapshot")
        .expect("the snapshot entry reads");
    first_snapshot.path().join(WEIGHTS_FILE_NAME)
}

fn little_endian_u32s(tensor_bytes: &[u8]) -> Vec<u32> {
    let (word_arrays, _) = tensor_bytes.as_chunks::<4>();
    let mut decoded_words = Vec::with_capacity(word_arrays.len());
    for word_bytes in word_arrays {
        decoded_words.push(u32::from_le_bytes(*word_bytes));
    }
    decoded_words
}

fn little_endian_u16s(tensor_bytes: &[u8]) -> Vec<u16> {
    let (value_arrays, _) = tensor_bytes.as_chunks::<2>();
    let mut decoded_values = Vec::with_capacity(value_arrays.len());
    for value_bytes in value_arrays {
        decoded_values.push(u16::from_le_bytes(*value_bytes));
    }
    decoded_values
}

fn little_endian_f32s(tensor_bytes: &[u8]) -> Vec<f32> {
    let mut decoded_values = Vec::with_capacity(tensor_bytes.len() / 4);
    for word in little_endian_u32s(tensor_bytes) {
        decoded_values.push(f32::from_bits(word));
    }
    decoded_values
}
