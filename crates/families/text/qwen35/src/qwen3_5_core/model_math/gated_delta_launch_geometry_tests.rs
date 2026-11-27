//! The pipelined gated-delta kernels hard-code the launch geometry the
//! launchers dispatch with; these tests keep the two in step without a GPU.

use super::{THREADGROUP_THREAD_COUNT, VALUE_ROW_BLOCK_SIZE};
use crate::metal_kernels::{
    ASTRO_FAMILY_QWEN35_GATED_DELTA_REPLAY, ASTRO_FAMILY_QWEN35_GATED_DELTA_SEQUENCE,
    ASTRO_FAMILY_QWEN35_GATED_DELTA_SEQUENCE_CHECKPOINTS, ASTRO_FAMILY_QWEN35_GATED_DELTA_TREE,
};
use metal_kernel_declarations::MetalKernelSource;

fn kernel_source_text(kernel: &MetalKernelSource) -> String {
    kernel
        .source_text(&[])
        .expect("a gated-delta kernel declares no defines")
}

fn declares_constant(source_text: &str, constant_name: &str, constant_value: usize) -> bool {
    source_text.contains(&format!("int {constant_name} = {constant_value};"))
}

#[test]
fn should_declare_the_value_row_block_every_gated_delta_kernel_launches_with() {
    for kernel in [
        &ASTRO_FAMILY_QWEN35_GATED_DELTA_SEQUENCE,
        &ASTRO_FAMILY_QWEN35_GATED_DELTA_SEQUENCE_CHECKPOINTS,
        &ASTRO_FAMILY_QWEN35_GATED_DELTA_TREE,
        &ASTRO_FAMILY_QWEN35_GATED_DELTA_REPLAY,
    ] {
        assert!(
            declares_constant(
                &kernel_source_text(kernel),
                "value_row_block_size",
                VALUE_ROW_BLOCK_SIZE
            ),
            "{} must declare value_row_block_size = {VALUE_ROW_BLOCK_SIZE}",
            kernel.name
        );
    }
}

#[test]
fn should_declare_the_threadgroup_size_the_pipelined_sequence_kernels_launch_with() {
    for kernel in [
        &ASTRO_FAMILY_QWEN35_GATED_DELTA_SEQUENCE,
        &ASTRO_FAMILY_QWEN35_GATED_DELTA_SEQUENCE_CHECKPOINTS,
    ] {
        assert!(
            declares_constant(
                &kernel_source_text(kernel),
                "threadgroup_thread_count",
                THREADGROUP_THREAD_COUNT
            ),
            "{} must declare threadgroup_thread_count = {THREADGROUP_THREAD_COUNT}",
            kernel.name
        );
    }
}

#[test]
fn should_refuse_a_geometry_the_kernel_text_does_not_declare() {
    let sequence_source = kernel_source_text(&ASTRO_FAMILY_QWEN35_GATED_DELTA_SEQUENCE);

    assert!(!declares_constant(
        &sequence_source,
        "value_row_block_size",
        VALUE_ROW_BLOCK_SIZE + 1
    ));
    assert!(!declares_constant(
        &sequence_source,
        "no_such_constant",
        VALUE_ROW_BLOCK_SIZE
    ));
}

#[test]
fn should_unroll_one_step_per_token_of_the_twelve_token_block() {
    let sequence_source = kernel_source_text(&ASTRO_FAMILY_QWEN35_GATED_DELTA_SEQUENCE);

    assert!(sequence_source.contains("int time_block_size = 12;"));
    // Twelve unrolled steps plus one rolled tail step, each rotating its
    // prefetched operands into current ones once.
    assert_eq!(sequence_source.matches("current_keys[4];").count(), 13);
}
