use std::fs::{self, File};

use model_primitives::performance_attribution::PerformanceAttribution;

use super::write_host_safetensors;
use crate::astro_decoder_state_cache::AstroDecoderStateCachePersistedTensorLayout;
use crate::astro_ssd_prompt_cache::AstroSsdPromptCacheModelContract;
use crate::astro_ssd_prompt_cache::block_manifest::SEQUENCE_STATE_FILE_NAME;
use crate::astro_ssd_prompt_cache::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use crate::astro_ssd_prompt_cache::disk_store_file::save_direct_safetensors_file_with_name;
use crate::astro_ssd_prompt_cache::host_state_tensor::{
    AstroSsdPromptCacheHostTensor, AstroSsdPromptCacheHostTensorElements,
    AstroSsdPromptCacheHostTensors,
};
use crate::astro_ssd_prompt_cache::persistent_safetensors_header::read_persistent_safetensors_header;
use crate::astro_ssd_prompt_cache::restore_support_tests_support::test_model_contract;

/// A partial tail's token count: real Ornith geometry, small payloads.
const TAIL_TOKEN_COUNT: usize = 8;

fn host_tensor_for_layout(
    persisted_tensor_layout: &AstroDecoderStateCachePersistedTensorLayout,
    block_token_count: usize,
) -> AstroSsdPromptCacheHostTensor {
    let tensor_layout = persisted_tensor_layout.tensor_layout();
    let mut shape = tensor_layout.dimensions().to_vec();
    if let Some(sequence_axis) = tensor_layout.sequence_axis() {
        shape[sequence_axis] = block_token_count;
    }
    let element_count = shape.iter().product();
    AstroSsdPromptCacheHostTensor {
        safetensors_dtype_name: tensor_layout.dtype().safetensors_dtype_name(),
        shape,
        elements: AstroSsdPromptCacheHostTensorElements::SixteenBit(vec![0x3F80; element_count]),
    }
}

fn ornith_sequence_state_tensors(
    model_contract: &AstroSsdPromptCacheModelContract,
    block_token_count: usize,
) -> AstroSsdPromptCacheHostTensors {
    model_contract
        .astro_decoder_state_cache_layout()
        .sequence_tensor_layouts()
        .iter()
        .map(|persisted_tensor_layout| {
            (
                persisted_tensor_layout.persistent_tensor_name(),
                host_tensor_for_layout(persisted_tensor_layout, block_token_count),
            )
        })
        .collect()
}

fn named_host_tensor(
    elements: AstroSsdPromptCacheHostTensorElements,
    shape: Vec<usize>,
) -> AstroSsdPromptCacheHostTensor {
    AstroSsdPromptCacheHostTensor {
        safetensors_dtype_name: "F32",
        shape,
        elements,
    }
}

#[test]
fn should_stage_a_sequence_state_file_at_the_contract_exact_size() {
    let model_contract = test_model_contract();
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    let staged_file = save_direct_safetensors_file_with_name(
        staging_directory.path(),
        SEQUENCE_STATE_FILE_NAME,
        &ornith_sequence_state_tensors(&model_contract, TAIL_TOKEN_COUNT),
        TAIL_TOKEN_COUNT,
        &model_contract,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("the host-written file passes the header and layout validation");
    assert_eq!(
        staged_file.file_size_bytes,
        model_contract
            .sequence_state_file_bytes_for_block_token_count(TAIL_TOKEN_COUNT)
            .expect("the contract sizes a tail")
    );
}

#[test]
fn should_refuse_a_tensor_of_the_wrong_shape_and_leave_no_temporary_file() {
    let model_contract = test_model_contract();
    let staging_directory = tempfile::tempdir().expect("a staging directory");
    // Tensors shaped for one more token than the header records.
    let staging_outcome = save_direct_safetensors_file_with_name(
        staging_directory.path(),
        SEQUENCE_STATE_FILE_NAME,
        &ornith_sequence_state_tensors(&model_contract, TAIL_TOKEN_COUNT + 1),
        TAIL_TOKEN_COUNT,
        &model_contract,
        &mut PerformanceAttribution::disabled(),
    );
    assert!(matches!(
        staging_outcome,
        Err(AstroSsdPromptCacheDiskStoreError::ValidateBlock { .. })
    ));
    assert_eq!(
        fs::read_dir(staging_directory.path())
            .expect("the staging directory reads")
            .count(),
        0
    );
}

#[test]
fn should_lay_out_payloads_largest_first_then_by_name_in_little_endian_order() {
    let output_directory = tempfile::tempdir().expect("an output directory");
    let output_path = output_directory.path().join("ordered.safetensors");
    let mut host_tensors = AstroSsdPromptCacheHostTensors::new();
    host_tensors.insert(
        "b_small".to_owned(),
        named_host_tensor(
            AstroSsdPromptCacheHostTensorElements::Float32(vec![1.0]),
            vec![1],
        ),
    );
    host_tensors.insert(
        "a_small".to_owned(),
        named_host_tensor(
            AstroSsdPromptCacheHostTensorElements::Float32(vec![2.0]),
            vec![1],
        ),
    );
    host_tensors.insert(
        "large".to_owned(),
        named_host_tensor(
            AstroSsdPromptCacheHostTensorElements::Float32(vec![3.0, 4.0]),
            vec![2],
        ),
    );
    let written_byte_count = write_host_safetensors(
        File::create(&output_path).expect("the output file opens"),
        &host_tensors,
        &[("purpose", "test")],
    )
    .expect("the file writes");
    let header = read_persistent_safetensors_header(
        &File::open(&output_path).expect("the output file reopens"),
        &output_path,
    )
    .expect("the header parses");
    assert_eq!(header.tensor_views["large"].data_offsets, [0, 8]);
    assert_eq!(header.tensor_views["a_small"].data_offsets, [8, 12]);
    assert_eq!(header.tensor_views["b_small"].data_offsets, [12, 16]);
    assert_eq!(header.metadata["purpose"], "test");
    let file_bytes = fs::read(&output_path).expect("the file reads");
    assert_eq!(written_byte_count, file_bytes.len() as u64);
    let payload_start = file_bytes.len() - 16;
    assert_eq!(
        &file_bytes[payload_start..payload_start + 4],
        &3.0_f32.to_le_bytes()
    );
}

#[test]
fn should_write_an_empty_tensor_as_a_zero_length_payload() {
    let output_directory = tempfile::tempdir().expect("an output directory");
    let output_path = output_directory.path().join("empty.safetensors");
    let mut host_tensors = AstroSsdPromptCacheHostTensors::new();
    host_tensors.insert(
        "empty".to_owned(),
        named_host_tensor(
            AstroSsdPromptCacheHostTensorElements::Int32(Vec::new()),
            vec![0],
        ),
    );
    write_host_safetensors(
        File::create(&output_path).expect("the output file opens"),
        &host_tensors,
        &[],
    )
    .expect("the file writes");
    let header = read_persistent_safetensors_header(
        &File::open(&output_path).expect("the output file reopens"),
        &output_path,
    )
    .expect("the header parses");
    assert_eq!(header.tensor_views["empty"].data_offsets, [0, 0]);
    assert_eq!(header.data_section_start_bytes, header.file_size_bytes);
}
