use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PersistentPromptCacheBlockHeader, PersistentPromptCacheModelContract,
    qwen3_5_moe_decoder_cache_layout,
};

use crate::common::qwen3_5_moe::certified_ornith_config;

const EXPECTED_PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT: usize = 2_048;
const EXPECTED_PERSISTENT_PROMPT_CACHE_FORMAT_VERSION: &str = "8";

#[test]
fn should_accept_a_well_formed_ornith_persistent_prompt_cache_kv_block_header() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let persistent_prompt_cache_kv_block_path = temporary_directory.path().join("kv.safetensors");
    write_synthetic_persistent_prompt_cache_file(
        &persistent_prompt_cache_kv_block_path,
        SyntheticPersistentPromptCacheTensorLayout::KvOnly,
        |_| false,
    );

    let persistent_prompt_cache_kv_block_file = File::open(&persistent_prompt_cache_kv_block_path)
        .expect("the test should reopen the synthetic KV block");
    let kv_block_header = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &persistent_prompt_cache_kv_block_file,
        &persistent_prompt_cache_kv_block_path,
        &persistent_prompt_cache_model_contract(),
    )
    .expect("a well-formed Ornith persistent prompt-cache KV block should validate");

    assert_eq!(
        kv_block_header.format_version(),
        EXPECTED_PERSISTENT_PROMPT_CACHE_FORMAT_VERSION
    );
    assert_eq!(
        kv_block_header.model_id(),
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID
    );
    assert_eq!(
        kv_block_header.model_revision(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION
    );
    assert_eq!(
        kv_block_header.block_token_count(),
        EXPECTED_PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT
    );
    let expected_kv_tensor_count = certified_ornith_config()
        .full_attention_decoder_layer_indexes()
        .len()
        * 2;
    assert_eq!(kv_block_header.tensor_count(), expected_kv_tensor_count);
}

fn persistent_prompt_cache_model_contract() -> PersistentPromptCacheModelContract {
    PersistentPromptCacheModelContract::new(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        qwen3_5_moe_decoder_cache_layout(&certified_ornith_config())
            .expect("the certified Ornith configuration should build a decoder-cache layout"),
    )
}

#[test]
fn should_accept_a_well_formed_ornith_persistent_prompt_cache_recurrent_snapshot_header() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let persistent_prompt_cache_recurrent_snapshot_path =
        temporary_directory.path().join("snapshot.safetensors");
    write_synthetic_persistent_prompt_cache_file(
        &persistent_prompt_cache_recurrent_snapshot_path,
        SyntheticPersistentPromptCacheTensorLayout::RecurrentOnly,
        |_| false,
    );

    let persistent_prompt_cache_recurrent_snapshot_file =
        File::open(&persistent_prompt_cache_recurrent_snapshot_path)
            .expect("the test should reopen the synthetic recurrent snapshot");
    let recurrent_snapshot_header =
        PersistentPromptCacheBlockHeader::read_recurrent_snapshot_from_file(
            &persistent_prompt_cache_recurrent_snapshot_file,
            &persistent_prompt_cache_recurrent_snapshot_path,
            &persistent_prompt_cache_model_contract(),
        )
        .expect("a well-formed Ornith persistent prompt-cache recurrent snapshot should validate");

    assert_eq!(
        recurrent_snapshot_header.format_version(),
        EXPECTED_PERSISTENT_PROMPT_CACHE_FORMAT_VERSION
    );
    assert_eq!(
        recurrent_snapshot_header.model_id(),
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID
    );
    assert_eq!(
        recurrent_snapshot_header.model_revision(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION
    );
    assert_eq!(
        recurrent_snapshot_header.block_token_count(),
        EXPECTED_PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT
    );
    let expected_recurrent_tensor_count = certified_ornith_config()
        .linear_attention_decoder_layer_indexes()
        .len()
        * 2;
    assert_eq!(
        recurrent_snapshot_header.tensor_count(),
        expected_recurrent_tensor_count
    );
}

#[test]
fn should_reject_a_kv_block_missing_a_required_tensor() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let persistent_prompt_cache_kv_block_path = temporary_directory.path().join("kv.safetensors");
    write_synthetic_persistent_prompt_cache_file(
        &persistent_prompt_cache_kv_block_path,
        SyntheticPersistentPromptCacheTensorLayout::KvOnly,
        |tensor_name| tensor_name == "layer_3_attention.keys",
    );

    let persistent_prompt_cache_kv_block_file = File::open(&persistent_prompt_cache_kv_block_path)
        .expect("the test should reopen the synthetic KV block");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &persistent_prompt_cache_kv_block_file,
        &persistent_prompt_cache_kv_block_path,
        &persistent_prompt_cache_model_contract(),
    );

    assert!(rejection.is_err());
}

#[test]
fn should_reject_a_recurrent_snapshot_missing_a_required_tensor() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let persistent_prompt_cache_recurrent_snapshot_path =
        temporary_directory.path().join("snapshot.safetensors");
    write_synthetic_persistent_prompt_cache_file(
        &persistent_prompt_cache_recurrent_snapshot_path,
        SyntheticPersistentPromptCacheTensorLayout::RecurrentOnly,
        |tensor_name| tensor_name == "layer_0_linear.gated_delta_recurrent",
    );

    let persistent_prompt_cache_recurrent_snapshot_file =
        File::open(&persistent_prompt_cache_recurrent_snapshot_path)
            .expect("the test should reopen the synthetic recurrent snapshot");
    let rejection = PersistentPromptCacheBlockHeader::read_recurrent_snapshot_from_file(
        &persistent_prompt_cache_recurrent_snapshot_file,
        &persistent_prompt_cache_recurrent_snapshot_path,
        &persistent_prompt_cache_model_contract(),
    );

    assert!(rejection.is_err());
}

#[test]
fn should_reject_a_kv_block_containing_recurrent_tensors() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let persistent_prompt_cache_kv_block_path = temporary_directory.path().join("kv.safetensors");
    write_synthetic_persistent_prompt_cache_file(
        &persistent_prompt_cache_kv_block_path,
        SyntheticPersistentPromptCacheTensorLayout::KvAndRecurrent,
        |_| false,
    );

    let persistent_prompt_cache_kv_block_file = File::open(&persistent_prompt_cache_kv_block_path)
        .expect("the test should reopen the synthetic KV block");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &persistent_prompt_cache_kv_block_file,
        &persistent_prompt_cache_kv_block_path,
        &persistent_prompt_cache_model_contract(),
    );

    assert!(rejection.is_err());
}

#[test]
fn should_reject_a_recurrent_snapshot_containing_kv_tensors() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let persistent_prompt_cache_recurrent_snapshot_path =
        temporary_directory.path().join("snapshot.safetensors");
    write_synthetic_persistent_prompt_cache_file(
        &persistent_prompt_cache_recurrent_snapshot_path,
        SyntheticPersistentPromptCacheTensorLayout::KvAndRecurrent,
        |_| false,
    );

    let persistent_prompt_cache_recurrent_snapshot_file =
        File::open(&persistent_prompt_cache_recurrent_snapshot_path)
            .expect("the test should reopen the synthetic recurrent snapshot");
    let rejection = PersistentPromptCacheBlockHeader::read_recurrent_snapshot_from_file(
        &persistent_prompt_cache_recurrent_snapshot_file,
        &persistent_prompt_cache_recurrent_snapshot_path,
        &persistent_prompt_cache_model_contract(),
    );

    assert!(rejection.is_err());
}

#[test]
fn should_reject_a_truncated_current_format_kv_block_file() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let persistent_prompt_cache_kv_block_path = temporary_directory.path().join("kv.safetensors");
    write_synthetic_persistent_prompt_cache_file(
        &persistent_prompt_cache_kv_block_path,
        SyntheticPersistentPromptCacheTensorLayout::KvOnly,
        |_| false,
    );
    let original_file_size_bytes = std::fs::metadata(&persistent_prompt_cache_kv_block_path)
        .expect("the test should read the synthetic KV block metadata")
        .len();
    let persistent_prompt_cache_kv_block_file = OpenOptions::new()
        .write(true)
        .open(&persistent_prompt_cache_kv_block_path)
        .expect("the test should open the synthetic KV block for truncation");
    persistent_prompt_cache_kv_block_file
        .set_len(original_file_size_bytes / 2)
        .expect("the test should truncate the synthetic KV block");

    let persistent_prompt_cache_kv_block_file = File::open(&persistent_prompt_cache_kv_block_path)
        .expect("the test should reopen the truncated KV block");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &persistent_prompt_cache_kv_block_file,
        &persistent_prompt_cache_kv_block_path,
        &persistent_prompt_cache_model_contract(),
    );

    assert!(rejection.is_err());
}

#[test]
fn should_reject_format_four_state_after_execution_math_changes() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let stale_persistent_prompt_cache_kv_block_path = temporary_directory
        .path()
        .join("format-four-kv.safetensors");
    write_synthetic_persistent_prompt_cache_file_with_format_version(
        &stale_persistent_prompt_cache_kv_block_path,
        SyntheticPersistentPromptCacheTensorLayout::KvOnly,
        |_| false,
        "4",
    );

    let stale_persistent_prompt_cache_kv_block_file =
        File::open(&stale_persistent_prompt_cache_kv_block_path)
            .expect("the test should reopen the format-four KV block");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &stale_persistent_prompt_cache_kv_block_file,
        &stale_persistent_prompt_cache_kv_block_path,
        &persistent_prompt_cache_model_contract(),
    )
    .expect_err("state produced before the execution-math change must be rejected");

    let rejection_text = rejection.to_string();
    assert!(
        rejection_text.contains("format version is 4, expected 8"),
        "format-four state should fail with an actionable compatibility error: {rejection_text}"
    );
}

#[test]
fn should_reject_format_seven_state_after_expert_residency_correction() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let stale_persistent_prompt_cache_kv_block_path = temporary_directory
        .path()
        .join("format-seven-kv.safetensors");
    write_synthetic_persistent_prompt_cache_file_with_format_version(
        &stale_persistent_prompt_cache_kv_block_path,
        SyntheticPersistentPromptCacheTensorLayout::KvOnly,
        |_| false,
        "7",
    );

    let stale_persistent_prompt_cache_kv_block_file =
        File::open(&stale_persistent_prompt_cache_kv_block_path)
            .expect("the test should reopen the format-seven KV block");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &stale_persistent_prompt_cache_kv_block_file,
        &stale_persistent_prompt_cache_kv_block_path,
        &persistent_prompt_cache_model_contract(),
    )
    .expect_err("state produced before the expert-residency correction must be rejected");

    let rejection_text = rejection.to_string();
    assert!(
        rejection_text.contains("format version is 7, expected 8"),
        "format-seven state should fail with an actionable compatibility error: {rejection_text}"
    );
}

#[derive(Clone, Copy)]
enum SyntheticPersistentPromptCacheTensorLayout {
    KvOnly,
    RecurrentOnly,
    KvAndRecurrent,
}

fn write_synthetic_persistent_prompt_cache_file(
    persistent_prompt_cache_file_path: &Path,
    synthetic_tensor_layout: SyntheticPersistentPromptCacheTensorLayout,
    should_omit_tensor_name: impl Fn(&str) -> bool,
) {
    write_synthetic_persistent_prompt_cache_file_with_format_version(
        persistent_prompt_cache_file_path,
        synthetic_tensor_layout,
        should_omit_tensor_name,
        EXPECTED_PERSISTENT_PROMPT_CACHE_FORMAT_VERSION,
    );
}

fn write_synthetic_persistent_prompt_cache_file_with_format_version(
    persistent_prompt_cache_file_path: &Path,
    synthetic_tensor_layout: SyntheticPersistentPromptCacheTensorLayout,
    should_omit_tensor_name: impl Fn(&str) -> bool,
    persistent_prompt_cache_format_version: &str,
) {
    let tensor_entries = synthetic_persistent_prompt_cache_tensor_entries(
        synthetic_tensor_layout,
        should_omit_tensor_name,
    );
    let mut header_json = String::from("{");
    let mut current_data_section_offset: u64 = 0;
    for (tensor_entry_index, (tensor_name, dtype, shape, byte_count)) in
        tensor_entries.iter().enumerate()
    {
        if tensor_entry_index > 0 {
            header_json.push(',');
        }
        let next_data_section_offset = current_data_section_offset + *byte_count;
        header_json.push_str(&format!(
            r#""{tensor_name}":{{"dtype":"{dtype}","shape":[{shape}],"data_offsets":[{current_data_section_offset},{next_data_section_offset}]}}"#,
            shape = shape
                .iter()
                .map(|dimension| dimension.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ));
        current_data_section_offset = next_data_section_offset;
    }
    header_json.push_str(&format!(
        r#", "__metadata__":{{"format_version":"{persistent_prompt_cache_format_version}","model_id":"{ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID}","model_revision":"{ORNITH_1_0_35B_OPTIQ_4BIT_REVISION}","block_token_count":"{EXPECTED_PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT}"}}"#
    ));
    header_json.push('}');

    let header_bytes = header_json.into_bytes();
    let header_padding_bytes = (8 - header_bytes.len() % 8) % 8;
    let padded_header_length_bytes = header_bytes.len() + header_padding_bytes;
    let mut persistent_prompt_cache_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(persistent_prompt_cache_file_path)
        .expect("the test should create the synthetic persistent prompt-cache file");
    persistent_prompt_cache_file
        .write_all(&(padded_header_length_bytes as u64).to_le_bytes())
        .expect("the test should write the header length prefix");
    persistent_prompt_cache_file
        .write_all(&header_bytes)
        .expect("the test should write the header");
    persistent_prompt_cache_file
        .write_all(&vec![b' '; header_padding_bytes])
        .expect("the test should write the header padding");
    let data_section_start_offset = 8 + padded_header_length_bytes as u64;
    persistent_prompt_cache_file
        .set_len(data_section_start_offset + current_data_section_offset)
        .expect("the test should size the synthetic safetensors payload region");
    persistent_prompt_cache_file
        .sync_all()
        .expect("the test should sync the synthetic persistent prompt-cache file");
}

fn synthetic_persistent_prompt_cache_tensor_entries(
    synthetic_tensor_layout: SyntheticPersistentPromptCacheTensorLayout,
    should_omit_tensor_name: impl Fn(&str) -> bool,
) -> Vec<(String, &'static str, Vec<usize>, u64)> {
    let ornith_config = certified_ornith_config();
    let decoder_layer_count = ornith_config.layer_count() as usize;
    let key_value_head_count = ornith_config.key_value_head_count() as usize;
    let head_dimension = ornith_config.head_dimension() as usize;
    let linear_convolution_kernel_dimension =
        ornith_config.linear_convolution_kernel_dimension() as usize;
    let linear_convolution_dimension = (ornith_config.linear_key_head_count() as usize)
        .saturating_mul(ornith_config.linear_key_head_dimension() as usize)
        .saturating_mul(2)
        .saturating_add(
            (ornith_config.linear_value_head_count() as usize)
                .saturating_mul(ornith_config.linear_value_head_dimension() as usize),
        );
    let linear_value_head_count = ornith_config.linear_value_head_count() as usize;
    let linear_value_head_dimension = ornith_config.linear_value_head_dimension() as usize;
    let linear_key_head_dimension = ornith_config.linear_key_head_dimension() as usize;
    let mut tensor_entries = Vec::new();
    for layer_index in 0..decoder_layer_count {
        let layer_is_full_attention = ornith_config.decoder_layer_is_full_attention(layer_index);
        if layer_is_full_attention
            && matches!(
                synthetic_tensor_layout,
                SyntheticPersistentPromptCacheTensorLayout::KvOnly
                    | SyntheticPersistentPromptCacheTensorLayout::KvAndRecurrent
            )
        {
            for tensor_suffix in ["attention.keys", "attention.values"] {
                let tensor_name = format!("layer_{layer_index}_{tensor_suffix}");
                if should_omit_tensor_name(&tensor_name) {
                    continue;
                }
                let shape = vec![
                    1,
                    key_value_head_count,
                    EXPECTED_PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT,
                    head_dimension,
                ];
                tensor_entries.push((
                    tensor_name,
                    "BF16",
                    shape.clone(),
                    tensor_byte_count(&shape, 2),
                ));
            }
        }
        if !layer_is_full_attention
            && matches!(
                synthetic_tensor_layout,
                SyntheticPersistentPromptCacheTensorLayout::RecurrentOnly
                    | SyntheticPersistentPromptCacheTensorLayout::KvAndRecurrent
            )
        {
            let convolution_shape = vec![
                1usize,
                linear_convolution_kernel_dimension.saturating_sub(1),
                linear_convolution_dimension,
            ];
            let recurrent_shape = vec![
                1usize,
                linear_value_head_count,
                linear_value_head_dimension,
                linear_key_head_dimension,
            ];
            for (tensor_suffix, dtype, shape, element_size_bytes) in [
                ("linear.convolution", "BF16", convolution_shape, 2_u64),
                (
                    "linear.gated_delta_recurrent",
                    "F32",
                    recurrent_shape,
                    4_u64,
                ),
            ] {
                let tensor_name = format!("layer_{layer_index}_{tensor_suffix}");
                if should_omit_tensor_name(&tensor_name) {
                    continue;
                }
                tensor_entries.push((
                    tensor_name,
                    dtype,
                    shape.clone(),
                    tensor_byte_count(&shape, element_size_bytes),
                ));
            }
        }
    }
    tensor_entries
}

fn tensor_byte_count(shape: &[usize], element_size_bytes: u64) -> u64 {
    shape
        .iter()
        .map(|dimension| *dimension as u64)
        .product::<u64>()
        * element_size_bytes
}
