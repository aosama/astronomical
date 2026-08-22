use std::collections::HashMap;

use astronomical_model_serving::{
    DecoderCachePersistedTensorLayout, PersistentPromptCacheBlockKey,
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheDiskStoreError, PersistentPromptCacheModelContract,
};
use astronomical_runtime_integration::{MlxArray, MlxMemoryLimits, MlxRuntime};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;
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

pub(super) fn open_persistent_prompt_cache_disk_store_with_contract(
    persistent_prompt_cache_directory: &tempfile::TempDir,
    global_prompt_cache_maximum_size_bytes: u64,
    model_contract: PersistentPromptCacheModelContract,
) -> Result<PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreError> {
    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().to_path_buf(),
            persistent_prompt_cache_directory.path().to_path_buf(),
            global_prompt_cache_maximum_size_bytes,
        ),
        model_contract,
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

pub(super) fn persistent_prompt_cache_block_key_for_seed(
    token_seed: u32,
) -> PersistentPromptCacheBlockKey {
    PersistentPromptCacheBlockKey::for_root_block(
        &persistent_prompt_cache_model_contract(),
        &block_tokens_for_seed(token_seed),
    )
    .expect("the test should hash the block tokens")
}

pub(super) fn block_tokens_for_seed(token_seed: u32) -> Vec<u32> {
    (0..persistent_prompt_cache_model_contract().block_token_count())
        .map(|token_offset| token_seed + token_offset as u32)
        .collect()
}

pub(super) fn synthetic_kv_block_tensors(runtime: &MlxRuntime) -> HashMap<String, MlxArray> {
    synthetic_tensors_for_contract(
        runtime,
        &persistent_prompt_cache_model_contract()
            .decoder_cache_layout()
            .sequence_tensor_layouts(),
        persistent_prompt_cache_model_contract().block_token_count(),
    )
}

pub(super) fn synthetic_recurrent_snapshot_tensors(
    runtime: &MlxRuntime,
) -> HashMap<String, MlxArray> {
    synthetic_tensors_for_contract(
        runtime,
        &persistent_prompt_cache_model_contract()
            .decoder_cache_layout()
            .boundary_tensor_layouts(),
        persistent_prompt_cache_model_contract().block_token_count(),
    )
}

pub(super) fn synthetic_tensors_for_contract(
    runtime: &MlxRuntime,
    persisted_tensor_layouts: &[DecoderCachePersistedTensorLayout],
    block_token_count: usize,
) -> HashMap<String, MlxArray> {
    persisted_tensor_layouts
        .iter()
        .map(|persisted_tensor_layout| {
            let tensor_layout = persisted_tensor_layout.tensor_layout();
            let tensor_dimensions = tensor_layout
                .dimensions()
                .iter()
                .enumerate()
                .map(|(dimension_index, tensor_dimension)| {
                    if Some(dimension_index) == tensor_layout.sequence_axis() {
                        block_token_count as i32
                    } else {
                        *tensor_dimension as i32
                    }
                })
                .collect::<Vec<_>>();
            let tensor = runtime
                .zeros(&tensor_dimensions, tensor_layout.dtype().mlx_dtype())
                .expect("the contract-derived test tensor should allocate");
            (persisted_tensor_layout.persistent_tensor_name(), tensor)
        })
        .collect()
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
