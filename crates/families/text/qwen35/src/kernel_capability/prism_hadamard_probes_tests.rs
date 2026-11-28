//! The signed Hadamard rotation, launched for real on this machine's GPU at
//! every block a folded pack may declare. The worker probe checks one block;
//! Prism's 2B declares 512 and its 9B 4096, and each block runs a different
//! butterfly radix schedule (16·16·2 at 512, 16·16·16 at 4096), so each is
//! checked against the composed route here.

use serial_test::file_serial;

use super::{PrismHadamardSignedRotationProbe, verify_signed_rotation_against_the_composed_route};
use crate::kernel_capability::probe_test_runtime::probe_test_runtime;
use crate::kernel_capability::{CustomMetalKernelProbe, KernelCapabilityError};
use crate::prism_hadamard_qwen35::hadamard::SUPPORTED_BLOCK_SIZES;
use crate::qwen3_5_core::model_math::signed_rotation_kernel::is_signed_rotation_eligible;

#[test]
#[file_serial]
fn should_give_the_composed_bits_at_every_block_a_pack_may_declare() {
    let Some(runtime) =
        probe_test_runtime("should_give_the_composed_bits_at_every_block_a_pack_may_declare")
    else {
        return;
    };
    for block_size in SUPPORTED_BLOCK_SIZES {
        assert!(
            is_signed_rotation_eligible(block_size),
            "block {block_size}"
        );
        match verify_signed_rotation_against_the_composed_route(&runtime, block_size) {
            Ok(()) => {}
            Err(KernelCapabilityError::OutputMismatch { description }) => {
                panic!("block {block_size}: the kernel ran but changed bits: {description}");
            }
            Err(refusal) => {
                println!("block {block_size}: this GPU cannot run the kernel: {refusal:?}");
            }
        }
    }
}

#[test]
#[file_serial]
fn should_pass_the_worker_probe_or_refuse_a_gpu_that_cannot_run_it() {
    let Some(runtime) =
        probe_test_runtime("should_pass_the_worker_probe_or_refuse_a_gpu_that_cannot_run_it")
    else {
        return;
    };
    match PrismHadamardSignedRotationProbe::new(&runtime).probe() {
        Ok(()) => {}
        Err(KernelCapabilityError::OutputMismatch { description }) => {
            panic!("the kernel ran but changed bits: {description}");
        }
        Err(refusal) => println!("this GPU cannot run the kernel: {refusal:?}"),
    }
}

#[test]
fn should_keep_blocks_above_one_single_stage_transform_off_the_kernel() {
    assert!(!is_signed_rotation_eligible(16_384));
    assert!(!is_signed_rotation_eligible(8));
    assert!(!is_signed_rotation_eligible(1_536));
}
