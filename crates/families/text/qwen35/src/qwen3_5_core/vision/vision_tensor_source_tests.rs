//! Contracts for the vision tower's reopenable source files: which files are
//! kept, how locations are renumbered, and how unknown tensors are reported.

use std::collections::HashMap;
use std::fs::File;

use super::Qwen3_5VisionTensorSource;
use crate::qwen3_5_core::vision::{Qwen3_5ExecutionError, VisionTensorLocation};

/// Opens `file_count` empty files; the source only keeps their descriptors.
fn empty_source_files(directory: &std::path::Path, file_count: usize) -> Vec<File> {
    let mut source_files = Vec::with_capacity(file_count);
    for file_index in 0..file_count {
        source_files.push(
            File::create(directory.join(format!("model-{file_index}.safetensors")))
                .expect("a source file should be created"),
        );
    }
    source_files
}

fn location_at(source_file_position: usize, stored_tensor_name: &str) -> VisionTensorLocation {
    VisionTensorLocation {
        source_file_position,
        stored_tensor_name: stored_tensor_name.to_owned(),
    }
}

#[test]
fn should_keep_only_the_shards_that_hold_vision_tensors_and_renumber_them() {
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let vision_tensor_source = Qwen3_5VisionTensorSource::keeping_referenced_files(
        empty_source_files(shard_directory.path(), 4),
        HashMap::from([
            (
                "vision_tower.patch_embed.proj.weight".to_owned(),
                location_at(3, "model.visual.patch_embed.proj.weight"),
            ),
            (
                "vision_tower.blocks.0.attn.proj.bias".to_owned(),
                location_at(1, "model.visual.blocks.0.attn.proj.bias"),
            ),
        ]),
    );

    assert_eq!(vision_tensor_source.source_file_count(), 2);
    let renumbered_patch_location = vision_tensor_source
        .location("vision_tower.patch_embed.proj.weight")
        .expect("the patch embedding should be located");
    assert_eq!(renumbered_patch_location.source_file_position, 1);
    assert_eq!(
        renumbered_patch_location.stored_tensor_name, "model.visual.patch_embed.proj.weight",
        "compaction must keep the stored spelling"
    );
    assert_eq!(
        vision_tensor_source
            .location("vision_tower.blocks.0.attn.proj.bias")
            .expect("the block bias should be located")
            .source_file_position,
        0
    );
}

#[test]
fn should_keep_every_file_when_each_holds_a_vision_tensor() {
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let vision_tensor_source = Qwen3_5VisionTensorSource::keeping_referenced_files(
        empty_source_files(shard_directory.path(), 1),
        HashMap::from([(
            "vision_tower.blocks.0.attn.proj.bias".to_owned(),
            location_at(0, "vision_tower.blocks.0.attn.proj.bias"),
        )]),
    );

    assert_eq!(vision_tensor_source.source_file_count(), 1);
    assert_eq!(
        vision_tensor_source
            .location("vision_tower.blocks.0.attn.proj.bias")
            .expect("the block bias should be located")
            .source_file_position,
        0
    );
}

#[test]
fn should_leave_a_location_naming_a_missing_file_out_of_range() {
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let vision_tensor_source = Qwen3_5VisionTensorSource::keeping_referenced_files(
        empty_source_files(shard_directory.path(), 2),
        HashMap::from([(
            "vision_tower.blocks.0.attn.proj.bias".to_owned(),
            location_at(5, "vision_tower.blocks.0.attn.proj.bias"),
        )]),
    );

    assert_eq!(vision_tensor_source.source_file_count(), 0);
    assert_eq!(
        vision_tensor_source
            .location("vision_tower.blocks.0.attn.proj.bias")
            .expect("the location itself should still exist")
            .source_file_position,
        usize::MAX,
        "a location without a file must never point at another file"
    );
}

#[test]
fn should_report_a_tensor_the_source_does_not_locate_as_missing() {
    let vision_tensor_source = Qwen3_5VisionTensorSource::without_files();

    let location_error = vision_tensor_source
        .location("vision_tower.merger.linear_fc1.weight")
        .expect_err("an unknown tensor must not be located");

    assert!(matches!(
        location_error,
        Qwen3_5ExecutionError::MissingTensor { tensor_name }
            if tensor_name == "vision_tower.merger.linear_fc1.weight"
    ));
}
