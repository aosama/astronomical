//! The tiled sorted-expert kernels' probe, launched for real on this
//! machine's GPU.

use serial_test::file_serial;

use super::{SORTED_EXPERT_TILED_MATMUL_FAMILIES, SortedExpertTiledMatmulProbe};
use crate::kernel_capability::probe_test_runtime::probe_test_runtime;
use crate::kernel_capability::{CustomMetalKernelProbe, KernelCapabilityError};

/// MLX's exact bits at every weight width, or a refusal because this GPU
/// cannot build or run the kernels (CI's virtual Macs have no M5 matrix
/// units). Wrong bits are never acceptable anywhere.
#[test]
#[file_serial]
fn should_give_mlx_bits_or_refuse_a_gpu_that_cannot_run_the_tiled_kernels() {
    let Some(runtime) = probe_test_runtime(
        "should_give_mlx_bits_or_refuse_a_gpu_that_cannot_run_the_tiled_kernels",
    ) else {
        return;
    };
    for (family, weight_bits) in SORTED_EXPERT_TILED_MATMUL_FAMILIES {
        let probe = SortedExpertTiledMatmulProbe {
            runtime: &runtime,
            family,
            weight_bits,
        };
        match probe.probe() {
            Ok(()) => {}
            Err(KernelCapabilityError::Compilation { description })
            | Err(KernelCapabilityError::Execution { description }) => {
                println!(
                    "the {weight_bits}-bit probe refused this GPU, as it must where the kernels cannot run: {description}"
                );
            }
            Err(KernelCapabilityError::OutputMismatch { description }) => {
                panic!(
                    "the {weight_bits}-bit kernels ran but did not give MLX's bits: {description}"
                );
            }
        }
    }
}
