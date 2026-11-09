//! Contracts for embeddings read from the SSD: the rows they produce equal
//! the resident table's lookup bit for bit, for quantized and bfloat16
//! tables, and out-of-table tokens or mismatched forms are rejected.

use std::io::Write;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use super::disk_embedding_rows::{
    AffineEmbeddingTables, DiskEmbeddingTable, Qwen3_5DiskEmbeddingRows,
};
use super::embedding_row_reads::{EmbeddingRowReadError, EmbeddingTableLayout};
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

const VOCABULARY_ROW_COUNT: usize = 16;
const HIDDEN_SIZE: usize = 128;
const QUANTIZATION_BITS: i32 = 4;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const PACKED_WORDS_PER_ROW: usize = HIDDEN_SIZE * 4 / 32;
const GROUPS_PER_ROW: usize = HIDDEN_SIZE / 64;
/// Repeats a token, and reaches the first and last rows of the table.
const SHAKESPEARE_LIKE_TOKEN_IDS: [i32; 5] = [3, 7, 3, 15, 0];

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// The upper half of an f32: the bfloat16 with the same leading bits.
fn bfloat16_bytes(value: f32) -> [u8; 2] {
    let float_bytes = value.to_le_bytes();
    [float_bytes[2], float_bytes[3]]
}

/// One stored tensor of the fixture shard.
struct FixtureTensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    payload: Vec<u8>,
}

/// Writes the tensors as one SafeTensors shard and returns its path and the
/// absolute file offset where each tensor's data starts.
fn write_fixture_shard(
    shard_directory: &std::path::Path,
    fixture_tensors: &[FixtureTensor],
) -> (std::path::PathBuf, Vec<u64>) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut relative_starts = Vec::with_capacity(fixture_tensors.len());
    for fixture_tensor in fixture_tensors {
        relative_starts.push(payload.len() as u64);
        let tensor_start = payload.len();
        payload.extend_from_slice(&fixture_tensor.payload);
        header.insert(
            fixture_tensor.name.to_owned(),
            serde_json::json!({
                "dtype": fixture_tensor.dtype,
                "shape": fixture_tensor.shape,
                "data_offsets": [tensor_start, payload.len()],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&header).expect("the header should serialize");
    let shard_path = shard_directory.join("model.safetensors");
    let mut shard_file = std::fs::File::create(&shard_path).expect("the shard should be created");
    shard_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .and_then(|()| shard_file.write_all(&header_bytes))
        .and_then(|()| shard_file.write_all(&payload))
        .expect("the shard should be written");
    let data_section_start = 8 + header_bytes.len() as u64;
    let mut absolute_starts = Vec::with_capacity(relative_starts.len());
    for relative_start in relative_starts {
        absolute_starts.push(data_section_start + relative_start);
    }
    (shard_path, absolute_starts)
}

fn table_over(
    shard_path: &std::path::Path,
    first_row_file_offset: u64,
    fixture_tensor: &FixtureTensor,
) -> DiskEmbeddingTable {
    DiskEmbeddingTable {
        shard_file: std::fs::File::open(shard_path).expect("the shard should open"),
        layout: EmbeddingTableLayout {
            first_row_file_offset,
            row_byte_count: fixture_tensor.payload.len() / fixture_tensor.shape[0],
            row_count: fixture_tensor.shape[0],
        },
        stored_dtype: fixture_tensor.dtype.to_owned(),
        row_shape: fixture_tensor.shape[1..].to_vec(),
    }
}

/// A 4-bit table whose packed words, scales and biases all differ by row.
fn quantized_fixture_tensors() -> Vec<FixtureTensor> {
    let mut packed_words = Vec::with_capacity(VOCABULARY_ROW_COUNT * PACKED_WORDS_PER_ROW * 4);
    let mut scales = Vec::with_capacity(VOCABULARY_ROW_COUNT * GROUPS_PER_ROW * 2);
    let mut biases = Vec::with_capacity(VOCABULARY_ROW_COUNT * GROUPS_PER_ROW * 2);
    for row_index in 0..VOCABULARY_ROW_COUNT {
        for word_index in 0..PACKED_WORDS_PER_ROW {
            let packed_word = (row_index as u32).wrapping_mul(2_654_435_761)
                ^ (word_index as u32).wrapping_mul(40_503);
            packed_words.extend_from_slice(&packed_word.to_le_bytes());
        }
        for group_index in 0..GROUPS_PER_ROW {
            scales.extend_from_slice(&bfloat16_bytes(0.25 + row_index as f32 / 8.0));
            biases.extend_from_slice(&bfloat16_bytes(group_index as f32 - row_index as f32));
        }
    }
    vec![
        FixtureTensor {
            name: "language_model.model.embed_tokens.weight",
            dtype: "U32",
            shape: vec![VOCABULARY_ROW_COUNT, PACKED_WORDS_PER_ROW],
            payload: packed_words,
        },
        FixtureTensor {
            name: "language_model.model.embed_tokens.scales",
            dtype: "BF16",
            shape: vec![VOCABULARY_ROW_COUNT, GROUPS_PER_ROW],
            payload: scales,
        },
        FixtureTensor {
            name: "language_model.model.embed_tokens.biases",
            dtype: "BF16",
            shape: vec![VOCABULARY_ROW_COUNT, GROUPS_PER_ROW],
            payload: biases,
        },
    ]
}

fn bfloat16_fixture_tensor() -> FixtureTensor {
    let mut table_values = Vec::with_capacity(VOCABULARY_ROW_COUNT * 8 * 2);
    for row_index in 0..VOCABULARY_ROW_COUNT {
        for column_index in 0..8 {
            table_values.extend_from_slice(&bfloat16_bytes(
                row_index as f32 * 10.0 + column_index as f32,
            ));
        }
    }
    FixtureTensor {
        name: "language_model.model.embed_tokens.weight",
        dtype: "BF16",
        shape: vec![VOCABULARY_ROW_COUNT, 8],
        payload: table_values,
    }
}

fn token_indices(runtime: &MlxRuntime, token_ids: &[i32]) -> MlxArray {
    runtime
        .array_from_i32(token_ids, &[1, token_ids.len() as i32])
        .expect("the token ids should upload")
}

fn as_float32_values(runtime: &MlxRuntime, embeddings: &MlxArray) -> (Vec<i32>, Vec<f32>) {
    let float32_embeddings = runtime
        .astype(embeddings, MlxDtype::Float32)
        .expect("the embeddings should widen");
    runtime
        .evaluate_arrays(&[&float32_embeddings])
        .expect("the embeddings should evaluate");
    (
        float32_embeddings.shape(),
        float32_embeddings
            .to_vec_f32()
            .expect("the embeddings should be readable"),
    )
}

/// The resident form of the quantized fixture, read whole from the shard.
fn resident_quantized_form(
    runtime: &MlxRuntime,
    shard_path: &std::path::Path,
) -> Qwen3_5AffineWeights {
    let resident_map = runtime
        .load_safetensors(
            std::fs::File::open(shard_path).expect("the shard should open"),
            None,
        )
        .expect("the resident map should load");
    Qwen3_5AffineWeights::Quantized {
        packed_weight: resident_map
            .tensor("language_model.model.embed_tokens.weight")
            .expect("the packed table should be present"),
        quantization_scales: resident_map
            .tensor("language_model.model.embed_tokens.scales")
            .expect("the scales should be present"),
        quantization_biases: resident_map
            .tensor("language_model.model.embed_tokens.biases")
            .expect("the biases should be present"),
        quantization_bits: QUANTIZATION_BITS,
        quantization_group_size: QUANTIZATION_GROUP_SIZE,
    }
}

#[test]
#[serial]
fn should_embed_quantized_tokens_exactly_like_the_resident_table() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let fixture_tensors = quantized_fixture_tensors();
    let (shard_path, absolute_starts) =
        write_fixture_shard(shard_directory.path(), &fixture_tensors);
    let disk_embedding_rows = Qwen3_5DiskEmbeddingRows {
        weight_table: table_over(&shard_path, absolute_starts[0], &fixture_tensors[0]),
        affine_tables: Some(AffineEmbeddingTables {
            scales_table: table_over(&shard_path, absolute_starts[1], &fixture_tensors[1]),
            biases_table: table_over(&shard_path, absolute_starts[2], &fixture_tensors[2]),
        }),
    };
    let resident_form = resident_quantized_form(&runtime, &shard_path);
    let Qwen3_5AffineWeights::Quantized {
        packed_weight,
        quantization_scales,
        quantization_biases,
        ..
    } = &resident_form
    else {
        panic!("the fixture form should be quantized");
    };
    let indices = token_indices(&runtime, &SHAKESPEARE_LIKE_TOKEN_IDS);
    let resident_embeddings = runtime
        .dequantize_affine(
            &runtime
                .take_axis(packed_weight, &indices, 0)
                .expect("rows should gather"),
            &runtime
                .take_axis(quantization_scales, &indices, 0)
                .expect("scales should gather"),
            &runtime
                .take_axis(quantization_biases, &indices, 0)
                .expect("biases should gather"),
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
        )
        .expect("the resident rows should dequantize");

    let disk_embeddings = disk_embedding_rows
        .embed_tokens(&runtime, &indices, &resident_form)
        .expect("the disk rows should embed");

    let (disk_shape, disk_values) = as_float32_values(&runtime, &disk_embeddings);
    let (resident_shape, resident_values) = as_float32_values(&runtime, &resident_embeddings);
    assert_eq!(disk_shape, [1, 5, HIDDEN_SIZE as i32]);
    assert_eq!(disk_shape, resident_shape);
    assert_eq!(
        disk_values, resident_values,
        "disk rows must match the resident table exactly"
    );
}

#[test]
#[serial]
fn should_embed_bfloat16_tokens_exactly_like_the_resident_table() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let fixture_tensors = vec![bfloat16_fixture_tensor()];
    let (shard_path, absolute_starts) =
        write_fixture_shard(shard_directory.path(), &fixture_tensors);
    let disk_embedding_rows = Qwen3_5DiskEmbeddingRows {
        weight_table: table_over(&shard_path, absolute_starts[0], &fixture_tensors[0]),
        affine_tables: None,
    };
    let resident_table = runtime
        .load_safetensors(
            std::fs::File::open(&shard_path).expect("the shard should open"),
            None,
        )
        .expect("the resident map should load")
        .tensor("language_model.model.embed_tokens.weight")
        .expect("the table should be present");
    let indices = token_indices(&runtime, &SHAKESPEARE_LIKE_TOKEN_IDS);
    let resident_embeddings = runtime
        .take_axis(&resident_table, &indices, 0)
        .expect("the resident rows should gather");

    let disk_embeddings = disk_embedding_rows
        .embed_tokens(
            &runtime,
            &indices,
            &Qwen3_5AffineWeights::NativeBfloat16 {
                weight: resident_table,
            },
        )
        .expect("the disk rows should embed");

    assert_eq!(
        as_float32_values(&runtime, &disk_embeddings),
        as_float32_values(&runtime, &resident_embeddings)
    );
}

#[test]
#[serial]
fn should_reject_a_token_one_past_the_last_row() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let fixture_tensors = vec![bfloat16_fixture_tensor()];
    let (shard_path, absolute_starts) =
        write_fixture_shard(shard_directory.path(), &fixture_tensors);
    let disk_embedding_rows = Qwen3_5DiskEmbeddingRows {
        weight_table: table_over(&shard_path, absolute_starts[0], &fixture_tensors[0]),
        affine_tables: None,
    };
    let resident_form = Qwen3_5AffineWeights::NativeBfloat16 {
        weight: runtime
            .array_from_f32(&[0.0], &[1])
            .expect("a placeholder array should build"),
    };

    let embedding_error = disk_embedding_rows
        .embed_tokens(
            &runtime,
            &token_indices(&runtime, &[VOCABULARY_ROW_COUNT as i32]),
            &resident_form,
        )
        .expect_err("a token past the table must not embed");

    assert!(matches!(
        embedding_error,
        Qwen3_5ExecutionError::EmbeddingRowRead(EmbeddingRowReadError::TokenOutsideTable {
            token_id: 16,
            row_count: 16,
        })
    ));
}

#[test]
#[serial]
fn should_reject_tables_that_do_not_match_the_bound_form() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let fixture_tensors = quantized_fixture_tensors();
    let (shard_path, absolute_starts) =
        write_fixture_shard(shard_directory.path(), &fixture_tensors);
    // Quantized tables without their scales and biases, against a quantized form.
    let disk_embedding_rows = Qwen3_5DiskEmbeddingRows {
        weight_table: table_over(&shard_path, absolute_starts[0], &fixture_tensors[0]),
        affine_tables: None,
    };
    let resident_form = resident_quantized_form(&runtime, &shard_path);

    let embedding_error = disk_embedding_rows
        .embed_tokens(&runtime, &token_indices(&runtime, &[1, 2]), &resident_form)
        .expect_err("mismatched tables must not embed");

    assert!(matches!(
        embedding_error,
        Qwen3_5ExecutionError::InvalidTensor { .. }
    ));
}

fn bfloat16_disk_rows(shard_directory: &std::path::Path) -> Qwen3_5DiskEmbeddingRows {
    let fixture_tensors = vec![bfloat16_fixture_tensor()];
    let (shard_path, absolute_starts) = write_fixture_shard(shard_directory, &fixture_tensors);
    Qwen3_5DiskEmbeddingRows {
        weight_table: table_over(&shard_path, absolute_starts[0], &fixture_tensors[0]),
        affine_tables: None,
    }
}

#[test]
#[serial]
fn should_embed_one_decode_token_exactly_like_the_resident_table() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let disk_embedding_rows = bfloat16_disk_rows(shard_directory.path());
    let resident_table = runtime
        .load_safetensors(
            std::fs::File::open(shard_directory.path().join("model.safetensors"))
                .expect("the shard should open"),
            None,
        )
        .expect("the resident map should load")
        .tensor("language_model.model.embed_tokens.weight")
        .expect("the table should be present");
    let decode_token_index = token_indices(&runtime, &[9]);
    let resident_embedding = runtime
        .take_axis(&resident_table, &decode_token_index, 0)
        .expect("the resident row should gather");

    let disk_embedding = disk_embedding_rows
        .embed_tokens(
            &runtime,
            &decode_token_index,
            &Qwen3_5AffineWeights::NativeBfloat16 {
                weight: resident_table,
            },
        )
        .expect("the decode token should embed");

    let disk_embedding_values = as_float32_values(&runtime, &disk_embedding);
    assert_eq!(disk_embedding_values.0, [1, 1, 8]);
    assert_eq!(
        disk_embedding_values,
        as_float32_values(&runtime, &resident_embedding)
    );
}

#[test]
#[serial]
fn should_reject_token_indices_that_are_not_shaped_one_by_tokens() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let disk_embedding_rows = bfloat16_disk_rows(shard_directory.path());
    let flat_token_indices = runtime
        .array_from_i32(&[1, 2, 3], &[3])
        .expect("flat token ids should upload");
    let placeholder_form = Qwen3_5AffineWeights::NativeBfloat16 {
        weight: runtime
            .array_from_f32(&[0.0], &[1])
            .expect("a placeholder array should build"),
    };

    let embedding_error = disk_embedding_rows
        .embed_tokens(&runtime, &flat_token_indices, &placeholder_form)
        .expect_err("flat token indices must be rejected");

    assert!(matches!(
        embedding_error,
        Qwen3_5ExecutionError::InvalidInput { .. }
    ));
}
