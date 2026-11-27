use std::fs;
use std::path::Path;

use crate::kernel_files::{RegisteredKernel, read_registered_kernel};
use crate::kernel_instantiation::{kernel_instantiation, kernel_instantiation_values};
use crate::mlx_wrapper::ArgumentAddress;

fn registered_kernel_from(kernel_text: &str, folder: &Path) -> RegisteredKernel {
    let kernel_file = folder.join("astro_primitive_scaled.metal");
    fs::write(&kernel_file, kernel_text).expect("the kernel is written");
    read_registered_kernel(&kernel_file, folder).expect("the kernel reads")
}

#[test]
fn should_assemble_defines_template_values_and_resolved_dtypes() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    fs::write(
        folder.path().join("astro_primitive_scaled.h"),
        "// helper\n",
    )
    .expect("the header is written");
    let registered_kernel = registered_kernel_from(
        "// @header astro_primitive_scaled.h\n// @input X T\n// @input count int32_t always-small\n// @input scale float scalar\n// @input ids uint32_t|int32_t may-be-small\n// @output Y T\n// @template T typename\n// @template N int\n// @template STAGE bool\n// @define SHIFT\n// @compile-with T=float16_t N=-3 STAGE=true SHIFT=1.0f\nY[0] = X[0];\n",
        folder.path(),
    );
    let instantiation_values = &kernel_instantiation_values(&registered_kernel)[0];
    let instantiation =
        kernel_instantiation(&registered_kernel, instantiation_values).expect("it assembles");
    assert_eq!(instantiation.header_text, "// helper\n");
    assert!(
        instantiation
            .source_text
            .starts_with("#define SHIFT 1.0f\n// @header")
    );
    let template_values: Vec<&str> = instantiation
        .template_arguments
        .iter()
        .map(|template_argument| template_argument.value.as_str())
        .collect();
    assert_eq!(template_values, ["float16_t", "-3", "1"]);
    assert_eq!(instantiation.baseline_inputs[0].dtype, "float16_t");
    assert_eq!(
        instantiation.baseline_inputs[1].address,
        ArgumentAddress::Constant
    );
    assert_eq!(
        instantiation.baseline_inputs[2].address,
        ArgumentAddress::ConstantReference
    );
    assert_eq!(
        instantiation.baseline_inputs[3].address,
        ArgumentAddress::Device
    );
    assert_eq!(
        instantiation.baseline_inputs[3].dtype, "uint32_t",
        "the first dtype choice is the baseline"
    );
    assert_eq!(
        instantiation.outputs,
        [("Y".to_owned(), "float16_t".to_owned())]
    );
}

#[test]
fn should_give_a_kernel_without_parameters_one_empty_instantiation() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    let registered_kernel =
        registered_kernel_from("// @input X float\n// @output Y float\n", folder.path());
    let instantiation_values = kernel_instantiation_values(&registered_kernel);
    assert_eq!(instantiation_values.len(), 1);
    assert!(instantiation_values[0].is_empty());
    let instantiation =
        kernel_instantiation(&registered_kernel, &instantiation_values[0]).expect("it assembles");
    assert!(instantiation.template_arguments.is_empty());
    assert_eq!(instantiation.header_text, "");
}

#[test]
fn should_write_a_false_bool_template_value_as_zero() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    let registered_kernel = registered_kernel_from(
        "// @input X float\n// @output Y float\n// @template STAGE bool\n// @compile-with STAGE=false\n",
        folder.path(),
    );
    let instantiation = kernel_instantiation(
        &registered_kernel,
        &kernel_instantiation_values(&registered_kernel)[0],
    )
    .expect("it assembles");
    assert_eq!(instantiation.template_arguments[0].value, "0");
}

#[test]
fn should_refuse_a_kernel_whose_source_file_vanished() {
    let folder = tempfile::tempdir().expect("a temporary folder");
    let registered_kernel =
        registered_kernel_from("// @input X float\n// @output Y float\n", folder.path());
    fs::remove_file(folder.path().join("astro_primitive_scaled.metal"))
        .expect("the kernel file is removed");
    assert!(kernel_instantiation(&registered_kernel, &Vec::new()).is_err());
}
