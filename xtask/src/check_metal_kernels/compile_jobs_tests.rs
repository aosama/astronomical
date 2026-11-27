use std::fs;
use std::path::Path;

use metal_kernel_declarations::kernel_files::read_registered_kernel;

use super::compile_jobs::{MLX_LANGUAGE_VERSIONS, kernel_compile_jobs};

fn registered_kernel_from(
    kernel_text: &str,
    folder: &Path,
) -> metal_kernel_declarations::kernel_files::RegisteredKernel {
    let kernel_file = folder.join("astro_primitive_scaled.metal");
    fs::write(&kernel_file, kernel_text).expect("the kernel is written");
    read_registered_kernel(&kernel_file, folder).expect("the kernel reads")
}

#[test]
fn should_compile_one_baseline_per_language_for_a_plain_kernel() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    let registered_kernel = registered_kernel_from(
        "// @input X float\n// @output Y float\nY[0] = X[0];\n",
        folder.path(),
    );
    let compile_jobs = kernel_compile_jobs(&registered_kernel).expect("jobs are built");
    assert_eq!(compile_jobs.len(), MLX_LANGUAGE_VERSIONS.len());
    assert!(compile_jobs[0].full_source.starts_with("#include \"mlx/backend/metal/kernels/utils.h\"\n[[kernel]] void custom_kernel_astro_primitive_scaled(\n"));
    assert!(
        compile_jobs[0]
            .full_source
            .contains("  const device float* X [[buffer(0)]],\n")
    );
}

#[test]
fn should_add_a_constant_variant_per_small_capable_input_and_a_variant_per_alternative_dtype() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    let registered_kernel = registered_kernel_from(
        "// @input X bfloat16_t|float16_t\n// @input ids int32_t may-be-small\n// @input count int32_t always-small\n// @output Y float\nY[0] = X[ids[count[0]]];\n",
        folder.path(),
    );
    let compile_jobs = kernel_compile_jobs(&registered_kernel).expect("jobs are built");
    assert_eq!(
        compile_jobs.len(),
        3 * MLX_LANGUAGE_VERSIONS.len(),
        "baseline, X as float16_t, ids small"
    );
    assert!(
        compile_jobs[0]
            .full_source
            .contains("const device int32_t* ids")
    );
    assert!(
        compile_jobs[0]
            .full_source
            .contains("const constant int32_t* count")
    );
    assert!(
        compile_jobs[4]
            .description
            .contains("ids in constant memory")
    );
    assert!(
        compile_jobs[4]
            .full_source
            .contains("const constant int32_t* ids")
    );
    assert!(
        compile_jobs[2]
            .full_source
            .contains("const device float16_t* X")
    );
}

#[test]
fn should_write_defines_and_template_values_of_each_instantiation() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    let registered_kernel = registered_kernel_from(
        "// @input X float\n// @output Y float\n// @template N int\n// @template STAGE bool\n// @define SCALE\n// @compile-with N=8 STAGE=true SCALE=2.0f\n// @compile-with N=2 STAGE=false SCALE=0.5f\nY[0] = X[0] * SCALE * N;\n",
        folder.path(),
    );
    let compile_jobs = kernel_compile_jobs(&registered_kernel).expect("jobs are built");
    assert_eq!(compile_jobs.len(), 2 * MLX_LANGUAGE_VERSIONS.len());
    assert!(compile_jobs[0].full_source.contains("#define SCALE 2.0f\n"));
    assert!(
        compile_jobs[0]
            .full_source
            .contains("custom_kernel_astro_primitive_scaled<8, 1>")
    );
    let last_job = compile_jobs.last().expect("a last job");
    assert!(last_job.full_source.contains("#define SCALE 0.5f\n"));
    assert!(
        last_job
            .full_source
            .contains("custom_kernel_astro_primitive_scaled<2, 0>")
    );
}

#[test]
fn should_refuse_a_kernel_whose_body_file_vanished() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    fs::write(
        folder.path().join("astro_primitive_scaled_body.metal"),
        "Y[0] = X[0];",
    )
    .expect("the body is written");
    let registered_kernel = registered_kernel_from(
        "// @body astro_primitive_scaled_body.metal\n// @input X float\n// @output Y float\n",
        folder.path(),
    );
    fs::remove_file(folder.path().join("astro_primitive_scaled_body.metal"))
        .expect("the body is removed");
    assert!(kernel_compile_jobs(&registered_kernel).is_err());
}

#[test]
fn should_give_a_buffer_typed_by_a_template_parameter_that_parameters_value() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    let registered_kernel = registered_kernel_from(
        "// @input X T\n// @output Y T\n// @template T typename\n// @compile-with T=float16_t\nY[0] = X[0];\n",
        folder.path(),
    );
    let compile_jobs = kernel_compile_jobs(&registered_kernel).expect("jobs are built");
    assert!(
        compile_jobs[0]
            .full_source
            .contains("  const device float16_t* X [[buffer(0)]],\n")
    );
    assert!(
        compile_jobs[0]
            .full_source
            .contains("  device float16_t* Y [[buffer(1)]]")
    );
}
