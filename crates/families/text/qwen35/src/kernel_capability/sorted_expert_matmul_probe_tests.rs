//! The sorted-expert matmul's probe, launched for real on this machine's GPU.

use serial_test::file_serial;

use super::SortedExpertMatmulProbe;
use crate::kernel_capability::CustomMetalKernelProbe;
use crate::kernel_capability::probe_test_runtime::probe_test_runtime;

#[test]
#[file_serial]
fn should_pass_the_probe_on_this_machines_gpu() {
    let Some(runtime) = probe_test_runtime("should_pass_the_probe_on_this_machines_gpu") else {
        return;
    };
    assert_eq!(
        SortedExpertMatmulProbe { runtime: &runtime }.probe(),
        Ok(())
    );
}
