use std::collections::HashMap;
use std::fs;

use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT, PersistentPromptCacheBlockKey,
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheDiskStoreError,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::qwen3_5_moe::{certified_ornith_config, persistent_prompt_cache_model_contract};
use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

pub(super) fn open_persistent_prompt_cache_disk_store(
    persistent_prompt_cache_directory: &tempfile::TempDir,
    global_prompt_cache_maximum_size_bytes: u64,
) -> Result<PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreError> {
    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().to_path_buf(),
            persistent_prompt_cache_directory.path().to_path_buf(),
            global_prompt_cache_maximum_size_bytes,
        ),
        persistent_prompt_cache_model_contract(),
    )
}

pub(super) fn runtime_with_shared_limits() -> MlxRuntime {
    let memory_limits = MlxMemoryLimits::new(
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("the test memory limits should be valid");
    MlxRuntime::initialize(memory_limits).expect("the pinned MLX runtime should initialize")
}

pub(super) fn write_format_four_cache_file(format_four_cache_file_path: &std::path::Path) {
    let mut header_bytes = format!(
        r#"{{"__metadata__":{{"format_version":"4","model_id":"{ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID}","model_revision":"{ORNITH_1_0_35B_OPTIQ_4BIT_REVISION}","block_token_count":"{PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT}"}}}}"#
    )
    .into_bytes();
    let header_padding_byte_count = (8 - header_bytes.len() % 8) % 8;
    header_bytes.extend(std::iter::repeat_n(b' ', header_padding_byte_count));
    let mut format_four_file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    format_four_file_bytes.extend(header_bytes);
    fs::write(format_four_cache_file_path, format_four_file_bytes)
        .expect("the test should write the format-four cache file");
}

pub(super) fn persistent_prompt_cache_block_key_for_seed(
    token_seed: u32,
) -> PersistentPromptCacheBlockKey {
    PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &block_tokens_for_seed(token_seed),
    )
    .expect("the test should hash the block tokens")
}

pub(super) fn block_tokens_for_seed(token_seed: u32) -> Vec<u32> {
    (0..PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT)
        .map(|token_offset| token_seed + token_offset as u32)
        .collect()
}

pub(super) fn synthetic_kv_block_tensors(runtime: &MlxRuntime) -> HashMap<String, MlxArray> {
    let ornith_config = certified_ornith_config();
    let key_value_head_count = ornith_config.key_value_head_count() as i32;
    let head_dimension = ornith_config.head_dimension() as i32;
    let full_attention_layer_count = (0..ornith_config.layer_count() as usize)
        .filter(|layer_index| ornith_config.decoder_layer_is_full_attention(*layer_index))
        .count();
    let mut kv_block_tensors = HashMap::with_capacity(full_attention_layer_count * 2);
    for layer_index in 0..ornith_config.layer_count() as usize {
        if ornith_config.decoder_layer_is_full_attention(layer_index) {
            let keys = runtime
                .zeros(
                    &[
                        1,
                        key_value_head_count,
                        PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT as i32,
                        head_dimension,
                    ],
                    MlxDtype::BFloat16,
                )
                .expect("the test should create the keys tensor");
            let values = runtime
                .zeros(
                    &[
                        1,
                        key_value_head_count,
                        PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT as i32,
                        head_dimension,
                    ],
                    MlxDtype::BFloat16,
                )
                .expect("the test should create the values tensor");
            kv_block_tensors.insert(format!("layer_{layer_index}_attention.keys"), keys);
            kv_block_tensors.insert(format!("layer_{layer_index}_attention.values"), values);
        }
    }
    kv_block_tensors
}

pub(super) fn synthetic_recurrent_snapshot_tensors(
    runtime: &MlxRuntime,
) -> HashMap<String, MlxArray> {
    let ornith_config = certified_ornith_config();
    let linear_convolution_kernel_dimension =
        ornith_config.linear_convolution_kernel_dimension() as i32;
    let linear_convolution_dimension = ornith_config.linear_convolution_state_dimension();
    let linear_value_head_count = ornith_config.linear_value_head_count() as i32;
    let linear_value_head_dimension = ornith_config.linear_value_head_dimension() as i32;
    let linear_key_head_dimension = ornith_config.linear_key_head_dimension() as i32;
    let linear_attention_layer_count = (0..ornith_config.layer_count() as usize)
        .filter(|layer_index| !ornith_config.decoder_layer_is_full_attention(*layer_index))
        .count();
    let mut recurrent_snapshot_tensors = HashMap::with_capacity(linear_attention_layer_count * 2);
    for layer_index in 0..ornith_config.layer_count() as usize {
        if !ornith_config.decoder_layer_is_full_attention(layer_index) {
            let convolution = runtime
                .zeros(
                    &[
                        1,
                        linear_convolution_kernel_dimension.saturating_sub(1),
                        linear_convolution_dimension,
                    ],
                    MlxDtype::BFloat16,
                )
                .expect("the test should create the convolution tensor");
            let recurrent = runtime
                .zeros(
                    &[
                        1,
                        linear_value_head_count,
                        linear_value_head_dimension,
                        linear_key_head_dimension,
                    ],
                    MlxDtype::Float32,
                )
                .expect("the test should create the recurrent tensor");
            recurrent_snapshot_tensors.insert(
                format!("layer_{layer_index}_linear.convolution"),
                convolution,
            );
            recurrent_snapshot_tensors.insert(
                format!("layer_{layer_index}_linear.gated_delta_recurrent"),
                recurrent,
            );
        }
    }
    recurrent_snapshot_tensors
}

pub(super) fn assert_split_tensor_shapes_match(
    loaded_tensors: &HashMap<String, MlxArray>,
    expected_tensors: &HashMap<String, MlxArray>,
) {
    assert_eq!(loaded_tensors.len(), expected_tensors.len());
    for (tensor_name, expected_tensor) in expected_tensors {
        let loaded_tensor = loaded_tensors
            .get(tensor_name)
            .expect("the loaded split file should contain every saved tensor");
        assert_eq!(loaded_tensor.shape(), expected_tensor.shape());
        assert_eq!(loaded_tensor.dtype(), expected_tensor.dtype());
    }
}

pub(super) mod hex {
    pub fn encode(block_hash_bytes: [u8; 32]) -> String {
        block_hash_bytes
            .iter()
            .map(|block_hash_byte| format!("{block_hash_byte:02x}"))
            .collect()
    }
}
