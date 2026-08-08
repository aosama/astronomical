use std::path::Path;

use astronomical_model_serving::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheModelContract, PersistentSpeculativePrefillSelectionContract,
};
use astronomical_runtime_integration::MlxDtype;

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;
use crate::direct_mlx::persistent_prompt_cache_disk_store_support::runtime_with_shared_limits;

#[test]
fn should_reuse_drafter_selection_after_switching_to_another_drafter_and_back() {
    let temporary_cache_root_directory = tempfile::tempdir().expect("the test cache root exists");
    let runtime = runtime_with_shared_limits();
    let prompt_token_ids = vec![11_u32, 12, 13, 14];
    let selected_token_positions_on_gpu = runtime
        .array_from_u32(&[0, 2, 3], &[3])
        .expect("the selection tensor should be created");
    let shared_decoder_cache_layout = persistent_prompt_cache_model_contract()
        .decoder_cache_layout()
        .clone();
    let drafter_a_model_contract = PersistentPromptCacheModelContract::new(
        "drafter-a".to_owned(),
        "revision-a".to_owned(),
        shared_decoder_cache_layout.clone(),
    );
    let drafter_b_model_contract = PersistentPromptCacheModelContract::new(
        "drafter-b".to_owned(),
        "revision-b".to_owned(),
        shared_decoder_cache_layout,
    );
    let drafter_a_selection_contract = selection_contract("drafter-a", "revision-a");
    let drafter_b_selection_contract = selection_contract("drafter-b", "revision-b");

    let drafter_a_cache_store = open_drafter_cache_store(
        temporary_cache_root_directory.path(),
        &drafter_a_model_contract,
    );
    drafter_a_cache_store
        .save_speculative_prefill_selection(
            &runtime,
            &drafter_a_selection_contract,
            &prompt_token_ids,
            &selected_token_positions_on_gpu,
        )
        .expect("drafter A selection should be saved");
    drop(drafter_a_cache_store);

    let drafter_b_cache_store = open_drafter_cache_store(
        temporary_cache_root_directory.path(),
        &drafter_b_model_contract,
    );
    assert!(
        drafter_b_cache_store
            .load_speculative_prefill_selection(
                &runtime,
                &drafter_b_selection_contract,
                &prompt_token_ids,
                None,
            )
            .expect("drafter B lookup should be cache-local")
            .is_none()
    );
    drop(drafter_b_cache_store);

    let drafter_a_cache_store = open_drafter_cache_store(
        temporary_cache_root_directory.path(),
        &drafter_a_model_contract,
    );
    let restored_selection = drafter_a_cache_store
        .load_speculative_prefill_selection(
            &runtime,
            &drafter_a_selection_contract,
            &prompt_token_ids,
            None,
        )
        .expect("drafter A selection should load after switching back")
        .expect("drafter A selection should remain on SSD");

    assert_eq!(restored_selection.dtype(), MlxDtype::UInt32);
    assert_eq!(
        runtime
            .copy_u32_values(&restored_selection)
            .expect("the restored selection should be readable"),
        vec![0, 2, 3]
    );
}

fn open_drafter_cache_store(
    global_prompt_cache_root_directory: &Path,
    drafter_model_contract: &PersistentPromptCacheModelContract,
) -> PersistentPromptCacheDiskStore {
    PersistentPromptCacheDiskStore::open(
        PersistentPromptCacheDiskStoreConfig::new(
            global_prompt_cache_root_directory.join(drafter_model_contract.model_id()),
            global_prompt_cache_root_directory.to_path_buf(),
            100_000_000,
        ),
        drafter_model_contract.clone(),
    )
    .expect("the drafter cache store should open")
}

fn selection_contract(
    drafter_model_id: &str,
    drafter_model_revision: &str,
) -> PersistentSpeculativePrefillSelectionContract {
    PersistentSpeculativePrefillSelectionContract::new(
        drafter_model_id.to_owned(),
        drafter_model_revision.to_owned(),
        [3_u8; 32],
        20,
        32,
        2,
        2,
        3,
        0,
        4,
    )
}
