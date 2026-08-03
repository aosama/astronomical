use std::{
    fs::{self, File, FileTimes},
    path::Path,
    time::{Duration, UNIX_EPOCH},
};

use astronomical_model_serving::{
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES, AlignedExpertPackBuildRequest,
    AlignedExpertPackError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource,
    SafetensorsDtype, build_aligned_expert_pack, build_aligned_expert_pack_metal_io_descriptors,
    read_aligned_expert_pack_header, validate_aligned_expert_pack_header,
    validate_aligned_expert_pack_payload,
};

#[test]
fn should_create_byte_identical_aligned_packs_from_identical_sources() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source(&source_file_path);
    let first_pack_path = temporary_directory.path().join("first.expert-pack");
    let second_pack_path = temporary_directory.path().join("second.expert-pack");
    let build_request = aligned_expert_pack_build_request(&layer_plan);

    let first_pack_header = build_aligned_expert_pack(&first_pack_path, &build_request)
        .expect("the first aligned expert pack should build");
    let second_pack_header = build_aligned_expert_pack(&second_pack_path, &build_request)
        .expect("the second aligned expert pack should build");

    assert_eq!(
        fs::read(&first_pack_path).expect("the first pack should be readable"),
        fs::read(&second_pack_path).expect("the second pack should be readable"),
        "identical model sources must produce one deterministic byte sequence"
    );
    assert_eq!(first_pack_header, second_pack_header);
    assert!(
        first_pack_header
            .tensor_descriptors
            .iter()
            .all(
                |tensor_descriptor| tensor_descriptor.pack_segment_offset_bytes
                    % ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES
                    == 0
            ),
        "every packed tensor must begin at an aligned deterministic offset"
    );
}

#[test]
fn should_preserve_each_source_tensor_at_its_aligned_pack_segment() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source(&source_file_path);
    let pack_path = temporary_directory.path().join("layer.expert-pack");
    let pack_header =
        build_aligned_expert_pack(&pack_path, &aligned_expert_pack_build_request(&layer_plan))
            .expect("the aligned expert pack should build");
    let source_bytes =
        fs::read(&source_file_path).expect("the synthetic source should be readable");
    let pack_bytes = fs::read(&pack_path).expect("the packed source should be readable");

    for tensor_descriptor in &pack_header.tensor_descriptors {
        let source_start_offset = usize::try_from(tensor_descriptor.source_payload_offset_bytes)
            .expect("the synthetic source offset should fit usize");
        let source_end_offset = source_start_offset
            .checked_add(tensor_descriptor.logical_byte_count)
            .expect("the synthetic source range should fit usize");
        let packed_start_offset = usize::try_from(tensor_descriptor.pack_segment_offset_bytes)
            .expect("the synthetic pack offset should fit usize");
        let packed_end_offset = packed_start_offset
            .checked_add(tensor_descriptor.logical_byte_count)
            .expect("the synthetic pack range should fit usize");
        assert_eq!(
            &pack_bytes[packed_start_offset..packed_end_offset],
            &source_bytes[source_start_offset..source_end_offset],
            "{} must preserve its source payload exactly",
            tensor_descriptor.tensor_name
        );
    }
}

#[test]
fn should_reject_a_pack_built_for_a_foreign_model_revision() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source(&source_file_path);
    let pack_path = temporary_directory.path().join("layer.expert-pack");
    let build_request = aligned_expert_pack_build_request(&layer_plan);
    build_aligned_expert_pack(&pack_path, &build_request)
        .expect("the aligned expert pack should build before validation");
    let parsed_pack_header = read_aligned_expert_pack_header(&pack_path)
        .expect("the freshly built pack header should parse");

    let validation_outcome = validate_aligned_expert_pack_header(
        &pack_path,
        &parsed_pack_header,
        &layer_plan,
        "synthetic-model",
        "foreign-revision",
        0,
    );

    assert!(matches!(
        validation_outcome,
        Err(AlignedExpertPackError::ForeignModelRevision { .. })
    ));
}

#[test]
fn should_reject_a_corrupted_pack_magic_before_reading_tensor_payloads() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source(&source_file_path);
    let pack_path = temporary_directory.path().join("layer.expert-pack");
    build_aligned_expert_pack(&pack_path, &aligned_expert_pack_build_request(&layer_plan))
        .expect("the aligned expert pack should build before corruption");
    let mut pack_bytes =
        fs::read(&pack_path).expect("the pack should be readable before corruption");
    pack_bytes[0] = b'X';
    fs::write(&pack_path, pack_bytes).expect("the test should corrupt only the pack magic byte");

    let read_outcome = read_aligned_expert_pack_header(&pack_path);

    assert!(matches!(
        read_outcome,
        Err(AlignedExpertPackError::InvalidMagic)
    ));
}

#[test]
fn should_reject_a_pack_after_its_source_file_modification_identity_changes() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source(&source_file_path);
    let pack_path = temporary_directory.path().join("layer.expert-pack");
    let build_request = aligned_expert_pack_build_request(&layer_plan);
    let pack_header = build_aligned_expert_pack(&pack_path, &build_request)
        .expect("the aligned expert pack should build before the source changes");
    File::options()
        .write(true)
        .open(&source_file_path)
        .expect("the synthetic source should reopen")
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(1)))
        .expect("the test should assign a distinct source modification identity");

    let validation_outcome = validate_aligned_expert_pack_header(
        &pack_path,
        &pack_header,
        &layer_plan,
        build_request.model_id,
        build_request.model_revision,
        build_request.layer_index,
    );

    assert!(matches!(
        validation_outcome,
        Err(AlignedExpertPackError::SourceFileModificationIdentityMismatch { .. })
    ));
}

#[test]
fn should_reject_pack_payload_bytes_that_differ_from_the_validated_source_tensor() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source(&source_file_path);
    let pack_path = temporary_directory.path().join("layer.expert-pack");
    let build_request = aligned_expert_pack_build_request(&layer_plan);
    let pack_header = build_aligned_expert_pack(&pack_path, &build_request)
        .expect("the aligned expert pack should build before payload corruption");
    let first_payload_offset = pack_header.tensor_descriptors[0].pack_segment_offset_bytes;
    use std::os::unix::fs::FileExt;
    File::options()
        .write(true)
        .open(&pack_path)
        .expect("the pack should reopen for one-byte corruption")
        .write_at(&[0xFF], first_payload_offset)
        .expect("the test should corrupt one packed payload byte");

    let validation_outcome =
        validate_aligned_expert_pack_payload(&pack_path, &pack_header, &layer_plan);

    assert!(matches!(
        validation_outcome,
        Err(AlignedExpertPackError::PayloadByteMismatch { .. })
    ));
}

#[test]
fn should_coalesce_adjacent_selected_experts_into_one_metal_io_range_per_tensor() {
    let (pack_header, metal_load_ranges) = build_metal_io_ranges_for_selected_experts(&[1, 2, 3]);

    assert_eq!(
        metal_load_ranges.len(),
        pack_header.tensor_descriptors.len()
    );
    for (output_tensor_index, (tensor_descriptor, metal_load_range)) in pack_header
        .tensor_descriptors
        .iter()
        .zip(&metal_load_ranges)
        .enumerate()
    {
        assert_eq!(metal_load_range.output_tensor_index(), output_tensor_index);
        assert_eq!(
            metal_load_range.source_file_offset_bytes(),
            tensor_descriptor.pack_segment_offset_bytes
                + u64::try_from(tensor_descriptor.bytes_per_expert)
                    .expect("the nonzero expert source offset should fit u64")
        );
        assert_eq!(metal_load_range.output_tensor_offset_bytes(), 0);
        assert_eq!(
            metal_load_range.byte_count(),
            tensor_descriptor.bytes_per_expert * 3
        );
    }
}

#[test]
fn should_keep_non_adjacent_selected_experts_in_separate_metal_io_ranges() {
    let (pack_header, metal_load_ranges) = build_metal_io_ranges_for_selected_experts(&[0, 2]);
    let tensor_descriptor_count = pack_header.tensor_descriptors.len();

    assert_eq!(metal_load_ranges.len(), tensor_descriptor_count * 2);
    let first_tensor_descriptor = &pack_header.tensor_descriptors[0];
    assert_eq!(metal_load_ranges[0].output_tensor_index(), 0);
    assert_eq!(
        metal_load_ranges[0].source_file_offset_bytes(),
        first_tensor_descriptor.pack_segment_offset_bytes
    );
    assert_eq!(metal_load_ranges[0].output_tensor_offset_bytes(), 0);
    assert_eq!(
        metal_load_ranges[1].source_file_offset_bytes(),
        first_tensor_descriptor.pack_segment_offset_bytes
            + u64::try_from(first_tensor_descriptor.bytes_per_expert * 2)
                .expect("the synthetic source offset should fit u64")
    );
    assert_eq!(
        metal_load_ranges[1].output_tensor_offset_bytes(),
        first_tensor_descriptor.bytes_per_expert
    );
    assert_eq!(
        metal_load_ranges[0].byte_count(),
        first_tensor_descriptor.bytes_per_expert
    );
    assert_eq!(
        metal_load_ranges[1].byte_count(),
        first_tensor_descriptor.bytes_per_expert
    );
}

#[test]
fn should_preserve_compact_page_slots_when_adjacent_and_scattered_experts_mix() {
    let (pack_header, metal_load_ranges) = build_metal_io_ranges_for_selected_experts(&[0, 1, 3]);
    let tensor_descriptor_count = pack_header.tensor_descriptors.len();

    assert_eq!(metal_load_ranges.len(), tensor_descriptor_count * 2);
    for (output_tensor_index, tensor_descriptor) in
        pack_header.tensor_descriptors.iter().enumerate()
    {
        let first_tensor_range = &metal_load_ranges[output_tensor_index * 2];
        let second_tensor_range = &metal_load_ranges[output_tensor_index * 2 + 1];
        assert_eq!(
            first_tensor_range.output_tensor_index(),
            output_tensor_index
        );
        assert_eq!(
            second_tensor_range.output_tensor_index(),
            output_tensor_index
        );
        assert_eq!(first_tensor_range.output_tensor_offset_bytes(), 0);
        assert_eq!(
            first_tensor_range.byte_count(),
            tensor_descriptor.bytes_per_expert * 2
        );
        assert_eq!(
            second_tensor_range.source_file_offset_bytes(),
            tensor_descriptor.pack_segment_offset_bytes
                + u64::try_from(tensor_descriptor.bytes_per_expert * 3)
                    .expect("the synthetic source offset should fit u64")
        );
        assert_eq!(
            second_tensor_range.output_tensor_offset_bytes(),
            first_tensor_range.byte_count(),
            "the second source run must continue the compact output without a hole"
        );
        assert_eq!(
            second_tensor_range.byte_count(),
            tensor_descriptor.bytes_per_expert
        );
        assert_eq!(
            first_tensor_range.byte_count() + second_tensor_range.byte_count(),
            tensor_descriptor.bytes_per_expert * 3,
            "Metal I/O must write exactly the three selected experts, not the skipped gap"
        );
    }
}

fn aligned_expert_pack_build_request(
    layer_plan: &QuantizedExpertLayerPlan,
) -> AlignedExpertPackBuildRequest<'_> {
    AlignedExpertPackBuildRequest {
        model_id: "synthetic-model",
        model_revision: "revision-1",
        layer_index: 0,
        layer_plan,
    }
}

fn build_metal_io_ranges_for_selected_experts(
    selected_expert_ids: &[usize],
) -> (
    astronomical_model_serving::AlignedExpertPackHeader,
    Vec<astronomical_runtime_integration::MlxMetalExpertPackLoadRange>,
) {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_file_path = temporary_directory.path().join("expert-source.bin");
    let layer_plan = write_synthetic_expert_source(&source_file_path);
    let pack_path = temporary_directory.path().join("layer.expert-pack");
    let pack_header =
        build_aligned_expert_pack(&pack_path, &aligned_expert_pack_build_request(&layer_plan))
            .expect("the aligned expert pack should build");
    let (_metal_output_tensors, metal_load_ranges) =
        build_aligned_expert_pack_metal_io_descriptors(
            &pack_header.tensor_descriptors,
            &layer_plan,
            selected_expert_ids,
        )
        .expect("the selected experts should produce Metal I/O descriptors");
    (pack_header, metal_load_ranges)
}

fn write_synthetic_expert_source(source_file_path: &Path) -> QuantizedExpertLayerPlan {
    write_synthetic_expert_source_for_layer(source_file_path, 0)
}

pub(crate) fn write_synthetic_expert_source_for_layer(
    source_file_path: &Path,
    layer_index: usize,
) -> QuantizedExpertLayerPlan {
    const EXPERT_CAPACITY: usize = 4;
    let tensor_layouts = [
        (
            "gate_proj",
            "weight",
            SafetensorsDtype::Uint32,
            12_usize,
            128_u64,
        ),
        ("gate_proj", "scales", SafetensorsDtype::BFloat16, 4, 256),
        ("gate_proj", "biases", SafetensorsDtype::BFloat16, 4, 384),
        ("up_proj", "weight", SafetensorsDtype::Uint32, 12, 512),
        ("up_proj", "scales", SafetensorsDtype::BFloat16, 4, 640),
        ("up_proj", "biases", SafetensorsDtype::BFloat16, 4, 768),
        ("down_proj", "weight", SafetensorsDtype::Uint32, 12, 896),
        ("down_proj", "scales", SafetensorsDtype::BFloat16, 4, 1_024),
        ("down_proj", "biases", SafetensorsDtype::BFloat16, 4, 1_152),
    ];
    let mut source_bytes = vec![0_u8; 1_280];
    let mut tensor_sources = Vec::with_capacity(tensor_layouts.len());
    for (
        tensor_position,
        (projection_name, parameter_name, dtype, bytes_per_expert, source_payload_offset_bytes),
    ) in tensor_layouts.into_iter().enumerate()
    {
        let tensor_byte_count = bytes_per_expert
            .checked_mul(EXPERT_CAPACITY)
            .expect("the synthetic tensor size should fit usize");
        let source_start_offset = usize::try_from(source_payload_offset_bytes)
            .expect("the synthetic source offset should fit usize");
        for source_byte_position in 0..tensor_byte_count {
            source_bytes[source_start_offset + source_byte_position] =
                u8::try_from(tensor_position * 17 + source_byte_position)
                    .expect("the synthetic byte value should fit u8");
        }
        tensor_sources.push(QuantizedTensorSource {
            tensor_name: format!(
                "layer.{layer_index}.switch_mlp.{projection_name}.{parameter_name}"
            ),
            projection_name: projection_name.to_owned(),
            parameter_name: parameter_name.to_owned(),
            quantization_bits: 8,
            quantization_group_size: 64,
            source_file: source_file_path.to_path_buf(),
            source_file_size_bytes: u64::try_from(source_bytes.len())
                .expect("the synthetic source length should fit u64"),
            dtype,
            full_shape: vec![EXPERT_CAPACITY, bytes_per_expert / dtype.byte_width()],
            tensor_payload_offset: source_payload_offset_bytes,
            bytes_per_expert,
            expert_capacity: EXPERT_CAPACITY,
        });
    }
    fs::write(source_file_path, source_bytes).expect("the test should write its synthetic source");
    QuantizedExpertLayerPlan {
        layer_prefix: format!("layer.{layer_index}"),
        tensor_sources,
        expert_capacity: EXPERT_CAPACITY,
        quantization_bits: 8,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}
