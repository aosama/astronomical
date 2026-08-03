use std::fs;

use astronomical_runtime_integration::{
    MlxDtype, MlxMetalExpertPackLoadRange, MlxMetalExpertPackOutputTensor,
};

use crate::common::runtime_test_support::runtime;

#[test]
fn should_load_a_file_range_into_an_mlx_owned_metal_buffer_before_gpu_use() {
    const PACK_BYTE_COUNT: usize = 64 * 1024;
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_pack_path = temporary_directory.path().join("synthetic.expert-pack");
    let source_values = (0..PACK_BYTE_COUNT / std::mem::size_of::<u32>())
        .map(|value_index| {
            u32::try_from(value_index + 1).expect("the synthetic value should fit u32")
        })
        .collect::<Vec<_>>();
    let source_pack_bytes = source_values
        .iter()
        .flat_map(|source_value| source_value.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(&source_pack_path, source_pack_bytes)
        .expect("the test should write a synthetic expert pack");

    let runtime = runtime();
    let metal_expert_pack_load = runtime
        .load_metal_expert_pack_ranges(
            &source_pack_path,
            &[MlxMetalExpertPackOutputTensor::new(
                vec![
                    i32::try_from(source_values.len()).expect("the synthetic shape should fit i32"),
                ],
                MlxDtype::UInt32,
            )],
            &[MlxMetalExpertPackLoadRange::new(0, 0, 0, PACK_BYTE_COUNT)],
        )
        .expect("Metal I/O should submit the synthetic expert pack range");

    let doubled_values = runtime
        .add(
            metal_expert_pack_load
                .output_array(0)
                .expect("the output array should exist"),
            metal_expert_pack_load
                .output_array(0)
                .expect("the output array should exist"),
        )
        .expect("the GPU stream should wait for Metal I/O before consuming its output")
        .to_vec_u32()
        .expect("the GPU output should copy as uint32 values");
    let metal_io_metrics = metal_expert_pack_load
        .wait_for_completion()
        .expect("the submitted Metal I/O command buffer should complete successfully");

    assert_eq!(
        doubled_values,
        source_values
            .iter()
            .map(|source_value| source_value * 2)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        metal_io_metrics.requested_byte_count,
        PACK_BYTE_COUNT as u64
    );
    assert_eq!(metal_io_metrics.command_count, 1);
    assert!(metal_io_metrics.queue_elapsed_nanoseconds > 0);
}

#[test]
fn should_reject_a_metal_expert_pack_range_that_exceeds_its_source_file() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_pack_path = temporary_directory.path().join("synthetic.expert-pack");
    fs::write(&source_pack_path, vec![0_u8; 64 * 1024])
        .expect("the test should write a synthetic expert pack");

    let load_outcome = runtime().load_metal_expert_pack_ranges(
        &source_pack_path,
        &[MlxMetalExpertPackOutputTensor::new(
            vec![16_384],
            MlxDtype::UInt32,
        )],
        &[MlxMetalExpertPackLoadRange::new(0, 0, 1, 64 * 1024)],
    );

    assert!(load_outcome.is_err());
}

#[test]
fn should_release_an_inflight_metal_load_without_an_explicit_completion_wait() {
    const PACK_BYTE_COUNT: usize = 64 * 1024;
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_pack_path = temporary_directory.path().join("inflight.expert-pack");
    fs::write(&source_pack_path, vec![0x5A_u8; PACK_BYTE_COUNT])
        .expect("the test should write a synthetic expert pack");
    let runtime = runtime();

    for repetition_index in 0..4 {
        let metal_expert_pack_load = runtime
            .load_metal_expert_pack_ranges(
                &source_pack_path,
                &[MlxMetalExpertPackOutputTensor::new(
                    vec![
                        i32::try_from(PACK_BYTE_COUNT / std::mem::size_of::<u32>())
                            .expect("the synthetic output shape should fit i32"),
                    ],
                    MlxDtype::UInt32,
                )],
                &[MlxMetalExpertPackLoadRange::new(0, 0, 0, PACK_BYTE_COUNT)],
            )
            .expect("Metal I/O should submit before immediate owner release");

        drop(metal_expert_pack_load);
        eprintln!(
            "[metal-expert-pack-loader] released inflight repetition {}/4",
            repetition_index + 1
        );
    }
}

#[test]
fn should_assemble_noncontiguous_ranges_into_multiple_mlx_owned_output_buffers() {
    const SOURCE_VALUE_COUNT: usize = 16_384;
    const VALUES_PER_RANGE: usize = 1_024;
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_pack_path = temporary_directory.path().join("noncontiguous.expert-pack");
    let source_values = (0..SOURCE_VALUE_COUNT)
        .map(|source_value_index| {
            u32::try_from(source_value_index + 100)
                .expect("the synthetic source value should fit u32")
        })
        .collect::<Vec<_>>();
    fs::write(
        &source_pack_path,
        source_values
            .iter()
            .flat_map(|source_value| source_value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .expect("the test should write a noncontiguous synthetic expert pack");
    let range_byte_count = VALUES_PER_RANGE * std::mem::size_of::<u32>();
    let runtime = runtime();

    let metal_expert_pack_load = runtime
        .load_metal_expert_pack_ranges(
            &source_pack_path,
            &[
                MlxMetalExpertPackOutputTensor::new(
                    vec![
                        i32::try_from(VALUES_PER_RANGE * 2)
                            .expect("the output shape should fit i32"),
                    ],
                    MlxDtype::UInt32,
                ),
                MlxMetalExpertPackOutputTensor::new(
                    vec![i32::try_from(VALUES_PER_RANGE).expect("the output shape should fit i32")],
                    MlxDtype::UInt32,
                ),
            ],
            &[
                MlxMetalExpertPackLoadRange::new(
                    1,
                    0,
                    (12 * range_byte_count) as u64,
                    range_byte_count,
                ),
                MlxMetalExpertPackLoadRange::new(
                    0,
                    range_byte_count,
                    (7 * range_byte_count) as u64,
                    range_byte_count,
                ),
                MlxMetalExpertPackLoadRange::new(
                    0,
                    0,
                    (2 * range_byte_count) as u64,
                    range_byte_count,
                ),
            ],
        )
        .expect("Metal I/O should submit noncontiguous ranges for multiple outputs");

    let first_output_values = runtime
        .add(
            metal_expert_pack_load
                .output_array(0)
                .expect("the first output should exist"),
            metal_expert_pack_load
                .output_array(0)
                .expect("the first output should exist"),
        )
        .expect("the GPU should consume the first output after Metal I/O")
        .to_vec_u32()
        .expect("the first GPU output should copy as uint32 values");
    let second_output_values = metal_expert_pack_load
        .output_array(1)
        .expect("the second output should exist")
        .to_vec_u32()
        .expect("the second output should copy as uint32 values");
    let completion_metrics = metal_expert_pack_load
        .wait_for_completion()
        .expect("the noncontiguous Metal I/O request should complete");

    let expected_first_output_values = source_values[2 * VALUES_PER_RANGE..3 * VALUES_PER_RANGE]
        .iter()
        .chain(&source_values[7 * VALUES_PER_RANGE..8 * VALUES_PER_RANGE])
        .map(|source_value| source_value * 2)
        .collect::<Vec<_>>();
    assert_eq!(first_output_values, expected_first_output_values);
    assert_eq!(
        second_output_values,
        source_values[12 * VALUES_PER_RANGE..13 * VALUES_PER_RANGE]
    );
    assert_eq!(completion_metrics.command_count, 3);
    assert_eq!(
        completion_metrics.requested_byte_count,
        u64::try_from(range_byte_count * 3).expect("the requested byte count should fit u64")
    );
}
