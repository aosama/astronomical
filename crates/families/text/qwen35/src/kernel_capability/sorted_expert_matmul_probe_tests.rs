//! The sorted-expert matmul's probe, launched for real on this machine's GPU.

use serial_test::file_serial;

use super::SortedExpertMatmulProbe;
use crate::kernel_capability::probe_test_runtime::probe_test_runtime;
use crate::kernel_capability::{CustomMetalKernelProbe, KernelCapabilityError};

/// The probe's promise on any GPU: MLX's exact bits, or a refusal because
/// this GPU cannot build or run the kernel. Only an M5's matrix units build
/// it; CI's virtual Macs refuse it while building the pipeline, which is the
/// case the probe exists for. Wrong bits are never acceptable anywhere.
#[test]
#[file_serial]
fn should_give_mlx_bits_or_refuse_a_gpu_that_cannot_run_the_kernel() {
    let Some(runtime) =
        probe_test_runtime("should_give_mlx_bits_or_refuse_a_gpu_that_cannot_run_the_kernel")
    else {
        return;
    };
    match (SortedExpertMatmulProbe { runtime: &runtime }).probe() {
        Ok(()) => {}
        Err(KernelCapabilityError::Compilation { description })
        | Err(KernelCapabilityError::Execution { description }) => {
            println!(
                "the probe refused this GPU, as it must where the kernel cannot run: {description}"
            );
        }
        Err(KernelCapabilityError::OutputMismatch { description }) => {
            panic!("the kernel ran but did not give MLX's bits: {description}");
        }
    }
}
