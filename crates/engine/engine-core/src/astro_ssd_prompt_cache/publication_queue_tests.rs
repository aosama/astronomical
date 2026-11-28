use std::sync::Arc;

use super::{
    AstroSsdPromptCachePublicationJob, AstroSsdPromptCachePublicationQueue,
    MAXIMUM_QUEUED_PUBLICATION_PAYLOAD_BYTES,
};
use crate::astro_ssd_prompt_cache::host_state_tensor::{
    AstroSsdPromptCacheHostTensor, AstroSsdPromptCacheHostTensorElements,
    AstroSsdPromptCacheHostTensors,
};
use crate::astro_ssd_prompt_cache::synthetic_contract_test_support::synthetic_attention_model_contract;
use crate::astro_ssd_prompt_cache::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheDiskStoreConfig,
    AstroSsdPromptCacheModelContract,
};

const BLOCK_TOKEN_COUNT: usize = 8;

/// Eight-token blocks of one attention layer and no recurrent state.
fn attention_only_model_contract() -> AstroSsdPromptCacheModelContract {
    synthetic_attention_model_contract(
        "publication-queue-test-model",
        "revision",
        BLOCK_TOKEN_COUNT,
    )
}

fn opened_store(root_directory: &std::path::Path) -> Arc<AstroSsdPromptCacheDiskStore> {
    let disk_store_config = AstroSsdPromptCacheDiskStoreConfig::new(
        root_directory.join("model").join("revision"),
        root_directory.to_path_buf(),
        1_000_000,
    );
    Arc::new(
        AstroSsdPromptCacheDiskStore::open(disk_store_config, attention_only_model_contract())
            .expect("the store opens"),
    )
}

fn block_tensors(
    model_contract: &AstroSsdPromptCacheModelContract,
) -> AstroSsdPromptCacheHostTensors {
    let mut host_tensors = AstroSsdPromptCacheHostTensors::new();
    for persisted_tensor_layout in model_contract
        .astro_decoder_state_cache_layout()
        .sequence_tensor_layouts()
    {
        host_tensors.insert(
            persisted_tensor_layout.persistent_tensor_name(),
            AstroSsdPromptCacheHostTensor {
                safetensors_dtype_name: "F16",
                shape: vec![1, BLOCK_TOKEN_COUNT, 2],
                elements: AstroSsdPromptCacheHostTensorElements::SixteenBit(vec![
                    0x3C00;
                    BLOCK_TOKEN_COUNT
                        * 2
                ]),
            },
        );
    }
    host_tensors
}

fn publication_job(
    store: &AstroSsdPromptCacheDiskStore,
    block_key: &AstroSsdPromptCacheBlockKey,
    parent_block_key: Option<&AstroSsdPromptCacheBlockKey>,
) -> AstroSsdPromptCachePublicationJob {
    AstroSsdPromptCachePublicationJob {
        block_key: block_key.clone(),
        parent_block_key: parent_block_key.cloned(),
        sequence_state_tensors: block_tensors(&store.model_contract),
        boundary_state_tensors: AstroSsdPromptCacheHostTensors::new(),
    }
}

fn root_and_child_keys(
    store: &AstroSsdPromptCacheDiskStore,
) -> (AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheBlockKey) {
    let root_block_key = AstroSsdPromptCacheBlockKey::for_root_block(
        &store.model_contract,
        &[1, 2, 3, 4, 5, 6, 7, 8],
    )
    .expect("the root key hashes");
    let child_block_key = root_block_key
        .for_child_block(&[9, 10, 11, 12, 13, 14, 15, 16])
        .expect("the child key hashes");
    (root_block_key, child_block_key)
}

#[test]
fn should_let_a_reader_find_a_block_queued_before_it() {
    let root_directory = tempfile::tempdir().expect("a root directory");
    let store = opened_store(root_directory.path());
    let publication_queue = AstroSsdPromptCachePublicationQueue::start(
        Arc::clone(&store),
        MAXIMUM_QUEUED_PUBLICATION_PAYLOAD_BYTES,
    )
    .expect("the writer starts");
    let (root_block_key, _) = root_and_child_keys(&store);
    publication_queue
        .enqueue(publication_job(&store, &root_block_key, None))
        .expect("the block queues");
    assert!(store.has_kv_block(&root_block_key.block_hash()));
    assert_eq!(store.pending_publications().queued_block_count(), 0);
}

#[test]
fn should_write_a_child_after_its_parent_in_queue_order() {
    let root_directory = tempfile::tempdir().expect("a root directory");
    let store = opened_store(root_directory.path());
    let publication_queue = AstroSsdPromptCachePublicationQueue::start(
        Arc::clone(&store),
        MAXIMUM_QUEUED_PUBLICATION_PAYLOAD_BYTES,
    )
    .expect("the writer starts");
    let (root_block_key, child_block_key) = root_and_child_keys(&store);
    publication_queue
        .enqueue(publication_job(&store, &root_block_key, None))
        .expect("the root queues");
    publication_queue
        .enqueue(publication_job(
            &store,
            &child_block_key,
            Some(&root_block_key),
        ))
        .expect("the child queues");
    assert!(store.has_kv_block(&child_block_key.block_hash()));
    assert_eq!(store.sequence_state_block_count(), 2);
}

#[test]
fn should_drop_a_child_whose_parent_was_never_written_and_still_drain() {
    let root_directory = tempfile::tempdir().expect("a root directory");
    let store = opened_store(root_directory.path());
    let publication_queue = AstroSsdPromptCachePublicationQueue::start(
        Arc::clone(&store),
        MAXIMUM_QUEUED_PUBLICATION_PAYLOAD_BYTES,
    )
    .expect("the writer starts");
    let (root_block_key, child_block_key) = root_and_child_keys(&store);
    publication_queue
        .enqueue(publication_job(
            &store,
            &child_block_key,
            Some(&root_block_key),
        ))
        .expect("the orphan child queues");
    publication_queue.wait_until_drained();
    assert!(!store.has_kv_block(&child_block_key.block_hash()));
    assert_eq!(store.pending_publications().queued_payload_bytes(), 0);
}

#[test]
fn should_write_every_queued_block_when_the_queue_is_dropped() {
    let root_directory = tempfile::tempdir().expect("a root directory");
    let store = opened_store(root_directory.path());
    let (root_block_key, child_block_key) = root_and_child_keys(&store);
    {
        let publication_queue = AstroSsdPromptCachePublicationQueue::start(
            Arc::clone(&store),
            MAXIMUM_QUEUED_PUBLICATION_PAYLOAD_BYTES,
        )
        .expect("the writer starts");
        publication_queue
            .enqueue(publication_job(&store, &root_block_key, None))
            .expect("the root queues");
        publication_queue
            .enqueue(publication_job(
                &store,
                &child_block_key,
                Some(&root_block_key),
            ))
            .expect("the child queues");
    }
    assert_eq!(store.pending_publications().queued_block_count(), 0);
    assert!(store.has_kv_block(&root_block_key.block_hash()));
    assert!(store.has_kv_block(&child_block_key.block_hash()));
}

#[test]
fn should_queue_blocks_larger_than_the_bound_one_at_a_time() {
    let root_directory = tempfile::tempdir().expect("a root directory");
    let store = opened_store(root_directory.path());
    let publication_queue = AstroSsdPromptCachePublicationQueue::start(Arc::clone(&store), 1)
        .expect("the writer starts");
    let (root_block_key, child_block_key) = root_and_child_keys(&store);
    publication_queue
        .enqueue(publication_job(&store, &root_block_key, None))
        .expect("the root queues");
    publication_queue
        .enqueue(publication_job(
            &store,
            &child_block_key,
            Some(&root_block_key),
        ))
        .expect("the child queues once the root is written");
    assert!(store.has_kv_block(&child_block_key.block_hash()));
}
