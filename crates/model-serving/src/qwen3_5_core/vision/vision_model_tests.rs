//! Contracts for releasing the vision tower after an image request: nothing
//! happens before the tower is used, a used tower stops counting once
//! released, and a failed release keeps it resident and counted.

use std::cell::Cell;

use serial_test::serial;

use super::Qwen3_5VisionModel;
use crate::qwen3_5_core::frozen_config_test_support::FROZEN_VISION_CONFIG_JSON;
use crate::qwen3_5_core::vision::Qwen3_5VisionConfig;
use crate::qwen3_5_core::vision::vision_attention::Qwen3_5VisionPaddingZeroCache;
use crate::qwen3_5_core::vision::vision_weights::tests::{
    sidecar_source, test_runtime, vision_weights_over, write_released_tensor_sidecar,
};

const RELEASED_TENSOR_ELEMENT_COUNT: usize = 4 * 1024 * 1024;

/// A vision model over a one-tensor sidecar, marked used or not.
fn vision_model_over(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    shard_path: &std::path::Path,
    weights_have_been_used: bool,
) -> Qwen3_5VisionModel {
    Qwen3_5VisionModel {
        config: Qwen3_5VisionConfig::from_json_bytes(FROZEN_VISION_CONFIG_JSON.as_bytes())
            .expect("the frozen vision config should parse"),
        weights: vision_weights_over(runtime, sidecar_source(shard_path))
            .expect("the vision tensor should bind"),
        weights_have_been_used: Cell::new(weights_have_been_used),
        padding_zero_cache: Qwen3_5VisionPaddingZeroCache::new(),
    }
}

#[test]
#[serial]
fn should_release_nothing_when_no_image_used_the_tower() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_model = vision_model_over(&runtime, &shard_path, false);

    let was_released = vision_model
        .release_used_weights(&runtime)
        .expect("an unused tower should need no release");

    assert!(!was_released);
    assert_eq!(vision_model.resident_payload_bytes(), 0);
}

#[test]
#[serial]
fn should_stop_counting_a_used_tower_once_it_is_released() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_model = vision_model_over(&runtime, &shard_path, true);
    assert_eq!(
        vision_model.resident_payload_bytes(),
        vision_model.tower_payload_bytes(),
        "a used tower counts its whole payload"
    );

    let was_released = vision_model
        .release_used_weights(&runtime)
        .expect("a used tower should release");

    assert!(was_released);
    assert_eq!(vision_model.resident_payload_bytes(), 0);
    assert_eq!(
        vision_model.tower_payload_bytes(),
        (RELEASED_TENSOR_ELEMENT_COUNT * 2) as u64,
        "the tower's size stays known after release"
    );
}

#[test]
#[serial]
fn should_keep_counting_the_tower_when_its_release_fails() {
    let runtime = test_runtime();
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let shard_path = shard_directory.path().join("model-vision.safetensors");
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT);
    let mut vision_model = vision_model_over(&runtime, &shard_path, true);
    write_released_tensor_sidecar(&shard_path, RELEASED_TENSOR_ELEMENT_COUNT / 2);

    vision_model
        .release_used_weights(&runtime)
        .expect_err("a changed tower must not rebind");

    assert_eq!(
        vision_model.resident_payload_bytes(),
        vision_model.tower_payload_bytes(),
        "a tower that stayed resident must stay counted"
    );
}
