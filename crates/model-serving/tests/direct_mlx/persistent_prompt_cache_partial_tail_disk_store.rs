use astronomical_model_serving::PersistentPromptCachePublicationOutcome;

use super::persistent_prompt_cache_disk_store_support::*;
use crate::common::qwen3_5_moe;

const LARGE_CACHE_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::test]
async fn should_publish_a_partial_tail_block_and_survive_a_rescan() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let model_contract = qwen3_5_moe::persistent_prompt_cache_model_contract();
    let parent_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let tail_token_count = model_contract.block_token_count() / 2;
    let tail_block_key = parent_block_key
        .for_child_block(&tail_tokens_for_count(tail_token_count))
        .expect("the test should hash the tail tokens");
    let tail_kv_block_tensors =
        synthetic_tensors_for_contract(&runtime, &sequence_tensor_layouts(), tail_token_count);
    let tail_snapshot_tensors =
        synthetic_tensors_for_contract(&runtime, &boundary_tensor_layouts(), tail_token_count);

    persistent_prompt_cache
        .publish_block(
            &runtime,
            &parent_block_key,
            None,
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect("the test should publish the parent block first");
    let publication_outcome = persistent_prompt_cache
        .publish_block(
            &runtime,
            &tail_block_key,
            Some(&parent_block_key),
            &tail_kv_block_tensors,
            &tail_snapshot_tensors,
        )
        .expect("the test should durably publish the partial tail block");

    assert_eq!(
        publication_outcome,
        PersistentPromptCachePublicationOutcome::Published
    );
    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 2);
    let tail_block_directory = persistent_prompt_cache_directory
        .path()
        .join("blocks")
        .join(hex::encode(tail_block_key.block_hash()));
    assert!(tail_block_directory.join("sequence.safetensors").is_file());
    assert!(tail_block_directory.join("boundary.safetensors").is_file());

    let rescanned_persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should rescan its directory");
    assert_eq!(
        rescanned_persistent_prompt_cache.sequence_state_block_count(),
        2
    );
}

#[tokio::test]
async fn should_supersede_only_strictly_shorter_tail_siblings() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let model_contract = qwen3_5_moe::persistent_prompt_cache_model_contract();
    let parent_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let full_block_key = parent_block_key
        .for_child_block(&block_tokens_for_seed(1))
        .expect("the test should hash the full child block");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &parent_block_key,
            None,
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect("the test should publish the root block");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &full_block_key,
            Some(&parent_block_key),
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect("the test should publish the full child block");

    let shorter_tail_token_count = model_contract.block_token_count() / 4;
    let shorter_tail_block_key = parent_block_key
        .for_child_block(&tail_tokens_for_count(shorter_tail_token_count))
        .expect("the test should hash the shorter tail");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &shorter_tail_block_key,
            Some(&parent_block_key),
            &synthetic_tensors_for_contract(
                &runtime,
                &sequence_tensor_layouts(),
                shorter_tail_token_count,
            ),
            &synthetic_tensors_for_contract(
                &runtime,
                &boundary_tensor_layouts(),
                shorter_tail_token_count,
            ),
        )
        .expect("the test should publish the shorter tail");

    // A longer tail at the same chain position supersedes the shorter one but
    // must keep the equal-length sibling and the full block.
    let longer_tail_token_count = model_contract.block_token_count() / 2;
    let longer_tail_block_key = parent_block_key
        .for_child_block(&tail_tokens_for_count(longer_tail_token_count))
        .expect("the test should hash the longer tail");
    let equal_length_tail_tokens: Vec<u32> = (0..longer_tail_token_count)
        .map(|token_offset| 80_000_u32 + token_offset as u32)
        .collect();
    let equal_length_tail_block_key = parent_block_key
        .for_child_block(&equal_length_tail_tokens)
        .expect("the test should hash the equal-length tail");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &equal_length_tail_block_key,
            Some(&parent_block_key),
            &synthetic_tensors_for_contract(
                &runtime,
                &sequence_tensor_layouts(),
                longer_tail_token_count,
            ),
            &synthetic_tensors_for_contract(
                &runtime,
                &boundary_tensor_layouts(),
                longer_tail_token_count,
            ),
        )
        .expect("the test should publish the equal-length tail");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &longer_tail_block_key,
            Some(&parent_block_key),
            &synthetic_tensors_for_contract(
                &runtime,
                &sequence_tensor_layouts(),
                longer_tail_token_count,
            ),
            &synthetic_tensors_for_contract(
                &runtime,
                &boundary_tensor_layouts(),
                longer_tail_token_count,
            ),
        )
        .expect("the test should publish the longer tail");

    let blocks_directory = persistent_prompt_cache_directory.path().join("blocks");
    assert!(
        !blocks_directory
            .join(hex::encode(shorter_tail_block_key.block_hash()))
            .exists(),
        "the strictly shorter tail sibling should be removed"
    );
    assert!(
        blocks_directory
            .join(hex::encode(equal_length_tail_block_key.block_hash()))
            .is_dir(),
        "the equal-length tail sibling should survive for divergent conversations"
    );
    assert!(
        blocks_directory
            .join(hex::encode(longer_tail_block_key.block_hash()))
            .is_dir(),
        "the longer tail should survive as its conversation's growth path"
    );
    assert!(
        blocks_directory
            .join(hex::encode(full_block_key.block_hash()))
            .is_dir(),
        "full blocks must never be superseded by tails"
    );
    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 4);
}

#[tokio::test]
async fn should_keep_full_block_publication_free_of_tail_supersede() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let root_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let full_child_block_key = root_block_key
        .for_child_block(&block_tokens_for_seed(1))
        .expect("the test should hash the full child block");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &root_block_key,
            None,
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect("the test should publish the root block");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &full_child_block_key,
            Some(&root_block_key),
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect("the test should publish the full child block");

    assert_eq!(persistent_prompt_cache.sequence_state_block_count(), 2);
    assert!(
        persistent_prompt_cache_directory
            .path()
            .join("blocks")
            .join(hex::encode(root_block_key.block_hash()))
            .is_dir()
    );
    assert!(
        persistent_prompt_cache_directory
            .path()
            .join("blocks")
            .join(hex::encode(full_child_block_key.block_hash()))
            .is_dir()
    );
}

#[tokio::test]
async fn should_load_a_published_partial_tail_block_back_by_its_actual_token_count() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let persistent_prompt_cache = open_persistent_prompt_cache_disk_store(
        &persistent_prompt_cache_directory,
        LARGE_CACHE_LIMIT_BYTES,
    )
    .expect("the persistent prompt cache should open an empty directory");
    let model_contract = qwen3_5_moe::persistent_prompt_cache_model_contract();
    let parent_block_key = persistent_prompt_cache_block_key_for_seed(0);
    let tail_token_count = model_contract.block_token_count() / 2;
    let tail_block_key = parent_block_key
        .for_child_block(&tail_tokens_for_count(tail_token_count))
        .expect("the test should hash the tail tokens");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &parent_block_key,
            None,
            &synthetic_kv_block_tensors(&runtime),
            &synthetic_recurrent_snapshot_tensors(&runtime),
        )
        .expect("the test should publish the parent block first");
    persistent_prompt_cache
        .publish_block(
            &runtime,
            &tail_block_key,
            Some(&parent_block_key),
            &synthetic_tensors_for_contract(&runtime, &sequence_tensor_layouts(), tail_token_count),
            &synthetic_tensors_for_contract(&runtime, &boundary_tensor_layouts(), tail_token_count),
        )
        .expect("the test should durably publish the partial tail block");

    // The load path must validate the stored header against the key's actual
    // token count. Comparing against the contract's full block count here
    // rejected every tail restore with a token-count mismatch.
    let loaded_kv_block = persistent_prompt_cache
        .load_kv_block(&runtime, &tail_block_key, None)
        .expect("the partial tail KV block should load back by its actual token count")
        .expect("the published partial tail KV block should be present");
    assert_split_tensor_shapes_match(
        &loaded_kv_block,
        &synthetic_tensors_for_contract(&runtime, &sequence_tensor_layouts(), tail_token_count),
    );
    let loaded_snapshot = persistent_prompt_cache
        .load_recurrent_snapshot(&runtime, &tail_block_key, None)
        .expect("the partial tail snapshot should load back by its actual token count")
        .expect("the published partial tail snapshot should be present");
    assert_split_tensor_shapes_match(
        &loaded_snapshot,
        &synthetic_tensors_for_contract(&runtime, &boundary_tensor_layouts(), tail_token_count),
    );
}

fn tail_tokens_for_count(tail_token_count: usize) -> Vec<u32> {
    (0..tail_token_count)
        .map(|token_offset| 90_000_u32 + token_offset as u32)
        .collect()
}

fn sequence_tensor_layouts() -> Vec<astronomical_model_serving::DecoderCachePersistedTensorLayout> {
    qwen3_5_moe::persistent_prompt_cache_model_contract()
        .decoder_cache_layout()
        .sequence_tensor_layouts()
}

fn boundary_tensor_layouts() -> Vec<astronomical_model_serving::DecoderCachePersistedTensorLayout> {
    qwen3_5_moe::persistent_prompt_cache_model_contract()
        .decoder_cache_layout()
        .boundary_tensor_layouts()
}
