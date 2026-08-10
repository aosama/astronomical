use std::fs;

use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PersistentPromptCacheDiskStoreError, PersistentVisualEmbeddingKey,
};
use astronomical_runtime_integration::MlxDtype;

use super::persistent_prompt_cache_disk_store_support::*;
use crate::common::qwen3_5_moe::persistent_visual_embedding_model_contract;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_save_and_load_one_visual_embedding_file() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store(&cache_directory, LARGE_CACHE_LIMIT_BYTES)
        .expect("the cache should open");
    let visual_embedding_key = visual_embedding_key(7);
    let visual_embeddings = runtime
        .zeros(&[2, 2_048], MlxDtype::BFloat16)
        .expect("the test should create visual embeddings");

    cache
        .save_visual_embedding(&runtime, &visual_embedding_key, &visual_embeddings)
        .expect("the cache should save visual embeddings");

    assert_eq!(cache.visual_embedding_count(), 1);
    assert!(cache.visual_embedding_total_size_bytes() > 0);
    assert!(cache.has_visual_embedding(&visual_embedding_key.visual_embedding_hash()));
    assert_eq!(
        cache.visual_embedding_total_size_bytes(),
        cache.total_size_bytes()
    );
    let loaded_visual_embeddings = cache
        .load_visual_embedding(
            &runtime,
            &visual_embedding_key,
            &persistent_visual_embedding_model_contract(),
        )
        .expect("the cache should load visual embeddings")
        .expect("the saved visual embedding should be present");
    assert_eq!(loaded_visual_embeddings.shape(), vec![2, 2_048]);
    assert_eq!(loaded_visual_embeddings.dtype(), MlxDtype::BFloat16);
}

#[tokio::test]
async fn should_evict_the_oldest_visual_embedding_under_shared_quota_pressure() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let first_visual_embedding_key = visual_embedding_key(7);
    let second_visual_embedding_key = visual_embedding_key(8);
    let first_visual_embeddings = runtime
        .zeros(&[8, 2_048], MlxDtype::BFloat16)
        .expect("the test should create first visual embeddings");
    let second_visual_embeddings = runtime
        .zeros(&[8, 2_048], MlxDtype::BFloat16)
        .expect("the test should create second visual embeddings");
    let one_visual_embedding_quota_bytes = u64::try_from(first_visual_embeddings.byte_count())
        .unwrap_or(0)
        .saturating_add(17 * 1024);
    let cache =
        open_persistent_prompt_cache_disk_store(&cache_directory, one_visual_embedding_quota_bytes)
            .expect("the cache should open");

    cache
        .save_visual_embedding(
            &runtime,
            &first_visual_embedding_key,
            &first_visual_embeddings,
        )
        .expect("the cache should save the first visual embedding");
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    cache
        .save_visual_embedding(
            &runtime,
            &second_visual_embedding_key,
            &second_visual_embeddings,
        )
        .expect("the cache should save the second visual embedding");

    assert_eq!(cache.visual_embedding_count(), 1);
    assert!(!cache.has_visual_embedding(&first_visual_embedding_key.visual_embedding_hash()));
    assert!(cache.has_visual_embedding(&second_visual_embedding_key.visual_embedding_hash()));
    assert!(cache.total_size_bytes() <= one_visual_embedding_quota_bytes);
}

#[tokio::test]
async fn should_untrack_invalid_visual_embedding_load_and_accept_replacement() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store(&cache_directory, LARGE_CACHE_LIMIT_BYTES)
        .expect("the cache should open");
    let visual_embedding_key = visual_embedding_key(9);
    let visual_embeddings = runtime
        .zeros(&[2, 2_048], MlxDtype::BFloat16)
        .expect("the test should create visual embeddings");
    cache
        .save_visual_embedding(&runtime, &visual_embedding_key, &visual_embeddings)
        .expect("the cache should save visual embeddings");
    let visual_embedding_file_path =
        cache_directory
            .path()
            .join("visual_embeddings")
            .join(format!(
                "{}.safetensors",
                hex::encode(visual_embedding_key.visual_embedding_hash())
            ));
    fs::write(&visual_embedding_file_path, b"not a safetensors file")
        .expect("the test should corrupt the visual embedding file");

    let load_result = cache.load_visual_embedding(
        &runtime,
        &visual_embedding_key,
        &persistent_visual_embedding_model_contract(),
    );

    assert!(matches!(
        load_result,
        Err(PersistentPromptCacheDiskStoreError::ValidateModelSpecificArtifact { .. })
    ));
    assert_eq!(cache.visual_embedding_count(), 0);
    cache
        .save_visual_embedding(&runtime, &visual_embedding_key, &visual_embeddings)
        .expect("the cache should replace the invalid visual embedding file");
    let loaded_visual_embeddings = cache
        .load_visual_embedding(
            &runtime,
            &visual_embedding_key,
            &persistent_visual_embedding_model_contract(),
        )
        .expect("the replacement visual embedding should load")
        .expect("the replacement visual embedding should be tracked");
    assert_eq!(loaded_visual_embeddings.shape(), vec![2, 2_048]);
    assert_eq!(loaded_visual_embeddings.dtype(), MlxDtype::BFloat16);
}

fn visual_embedding_key(digest_byte: u8) -> PersistentVisualEmbeddingKey {
    PersistentVisualEmbeddingKey::for_image(
        [digest_byte; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    )
}
