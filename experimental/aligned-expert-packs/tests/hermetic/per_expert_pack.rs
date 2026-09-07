use std::fs;

use astronomical_experimental_aligned_expert_packs::{
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES, AlignedExpertPackError, PerExpertPackBuildRequest,
    build_per_expert_pack, build_per_expert_pack_metal_io_descriptors,
    per_expert_pack_relative_path, read_per_expert_pack_header, validate_per_expert_pack_header,
};
use astronomical_runtime_integration::MlxMetalExpertPackLoadRange;

use super::aligned_expert_pack::write_synthetic_expert_source_for_layer;

#[test]
fn should_store_one_expert_slice_bit_exactly_in_its_own_file() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source_for_layer(&source_file_path, 0);
    let pack_path = temporary_directory.path().join("0.apack");
    let source_bytes =
        fs::read(&source_file_path).expect("the synthetic source should be readable");

    let pack_header = build_per_expert_pack(
        &pack_path,
        &PerExpertPackBuildRequest {
            model_id: "synthetic-streaming",
            model_revision: "revision-1",
            layer_index: 0,
            expert_id: 2,
            layer_plan: &layer_plan,
        },
    )
    .expect("the per-expert pack should build");
    let pack_bytes = fs::read(&pack_path).expect("the per-expert pack should be readable");

    assert_eq!(pack_header.expert_id, 2);
    assert_eq!(pack_header.format_version, 3);
    for tensor_descriptor in &pack_header.tensor_descriptors {
        assert_eq!(
            tensor_descriptor.pack_segment_offset_bytes
                % ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES,
            0
        );
        let source_start = usize::try_from(tensor_descriptor.source_expert_payload_offset_bytes)
            .expect("the expert slice offset should fit usize");
        let source_end = source_start + tensor_descriptor.logical_byte_count;
        let packed_start = usize::try_from(tensor_descriptor.pack_segment_offset_bytes)
            .expect("the pack offset should fit usize");
        let packed_end = packed_start + tensor_descriptor.logical_byte_count;
        assert_eq!(
            &pack_bytes[packed_start..packed_end],
            &source_bytes[source_start..source_end],
            "{} must preserve expert 2 exactly",
            tensor_descriptor.tensor_name
        );
    }
}

#[test]
fn should_reject_a_pack_built_for_a_foreign_expert_id() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source_for_layer(&source_file_path, 0);
    let pack_path = temporary_directory.path().join("0.apack");
    build_per_expert_pack(
        &pack_path,
        &PerExpertPackBuildRequest {
            model_id: "synthetic-streaming",
            model_revision: "revision-1",
            layer_index: 0,
            expert_id: 1,
            layer_plan: &layer_plan,
        },
    )
    .expect("the per-expert pack should build");
    let pack_header = read_per_expert_pack_header(&pack_path)
        .expect("the freshly built per-expert header should parse");

    let validation_outcome = validate_per_expert_pack_header(
        &pack_path,
        &pack_header,
        &layer_plan,
        "synthetic-streaming",
        "revision-1",
        0,
        2,
    );

    assert!(matches!(
        validation_outcome,
        Err(AlignedExpertPackError::ForeignExpertId {
            expected_expert_id: 2,
            actual_expert_id: 1,
        })
    ));
}

#[test]
fn should_load_one_expert_file_into_the_selected_page_slot() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source_for_layer(&source_file_path, 0);
    let pack_path = temporary_directory.path().join("1.apack");
    let pack_header = build_per_expert_pack(
        &pack_path,
        &PerExpertPackBuildRequest {
            model_id: "synthetic-streaming",
            model_revision: "revision-1",
            layer_index: 0,
            expert_id: 1,
            layer_plan: &layer_plan,
        },
    )
    .expect("the per-expert pack should build");
    let pack_bytes = fs::read(&pack_path).expect("the per-expert pack should be readable");
    let source_bytes =
        fs::read(&source_file_path).expect("the synthetic source should be readable");

    let (_output_tensors, metal_load_ranges) =
        build_per_expert_pack_metal_io_descriptors(&pack_header, 0)
            .expect("one expert file should produce Metal I/O descriptors");
    assert_eq!(
        metal_load_ranges.len(),
        pack_header.tensor_descriptors.len()
    );
    for (load_range, tensor_descriptor) in metal_load_ranges
        .iter()
        .zip(&pack_header.tensor_descriptors)
    {
        let source_start = usize::try_from(load_range.source_file_offset_bytes())
            .expect("the pack offset should fit usize");
        let source_end = source_start + load_range.byte_count();
        let expected_start = usize::try_from(tensor_descriptor.source_expert_payload_offset_bytes)
            .expect("the expert slice offset should fit usize");
        let expected_end = expected_start + tensor_descriptor.logical_byte_count;
        assert_eq!(
            &pack_bytes[source_start..source_end],
            &source_bytes[expected_start..expected_end]
        );
        let _load_range: &MlxMetalExpertPackLoadRange = load_range;
    }
}

#[test]
fn should_name_expert_files_by_layer_and_expert_identity() {
    assert_eq!(
        per_expert_pack_relative_path(7, 3).to_string_lossy(),
        "layers/7/3.apack"
    );
}
