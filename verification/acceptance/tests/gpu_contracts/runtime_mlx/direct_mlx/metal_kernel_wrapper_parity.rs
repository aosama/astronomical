//! Whether `cargo xtask check-metal-kernels` compiles what MLX compiles.
//!
//! The offline compile rebuilds the source MLX writes around each kernel
//! body (`metal_kernel_declarations::mlx_wrapper`, a port of MLX's
//! `write_signature`). This contract asks MLX itself: it declares every
//! registered kernel of the repository, launches it once with MLX's verbose
//! flag on, which prints the generated source when the launch is recorded
//! and before anything is compiled or run, and compares that text with the
//! port's byte for byte. An MLX pin bump that changes the wrapper (the
//! signature, the 8-element constant rule, the attribute list, the template
//! instantiation) fails here instead of silently making the offline compile
//! check a different text.
//!
//! Each input is sized so MLX takes the path its signature block declares:
//! 8 elements for device memory, 1 for constant memory, none for a scalar.

use std::io::Read as _;
use std::os::fd::AsRawFd as _;
use std::path::PathBuf;

use metal_kernel_declarations::kernel_files::{
    RegisteredKernel, kernel_folder_files, read_registered_kernel, repository_root_above,
};
use metal_kernel_declarations::kernel_instantiation::{
    KernelInstantiation, kernel_instantiation, kernel_instantiation_values,
};
use metal_kernel_declarations::kernel_naming::SHARED_HEADER_DIRECTORY;
use metal_kernel_declarations::mlx_wrapper::{
    ArgumentAddress, WrappedTemplateArgument, WrapperRequest, mlx_kernel_source,
};
use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelConfig, MlxMetalKernelOutput,
    MlxMetalKernelTemplateArgument,
};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use crate::common::runtime_test_support;

/// Elements of an input MLX passes in device memory: MLX's constant-memory
/// rule takes inputs under 8.
const DEVICE_INPUT_ELEMENT_COUNT: i32 = 8;
const MLX_SOURCE_FENCE: &str = "```\n";

#[test]
#[serial]
fn should_rebuild_the_source_mlx_generates_for_every_registered_kernel() {
    let runtime = runtime_test_support::runtime();
    let registered_kernels = repository_registered_kernels();
    assert!(
        registered_kernels.len() >= 20,
        "the repository's kernels were found: {}",
        registered_kernels.len()
    );
    let mut mismatched_kernels = Vec::new();
    for registered_kernel in &registered_kernels {
        let instantiation_values = &kernel_instantiation_values(registered_kernel)[0];
        let instantiation = kernel_instantiation(registered_kernel, instantiation_values)
            .expect("the kernel's files read");
        let mlx_source = mlx_generated_source(&runtime, registered_kernel, &instantiation);
        let rebuilt_source = rebuilt_source(registered_kernel, &instantiation, &mlx_source);
        if rebuilt_source != mlx_source {
            mismatched_kernels.push(first_difference(
                &registered_kernel.name,
                &mlx_source,
                &rebuilt_source,
            ));
        }
    }
    assert!(
        mismatched_kernels.is_empty(),
        "the offline wrapper differs from MLX's:\n{}",
        mismatched_kernels.join("\n")
    );
}

#[test]
fn should_read_the_source_between_mlx_fences() {
    assert_eq!(
        fenced_source("Generated source code for `x`:\n```\nline one\nline two\n```\n").as_deref(),
        Some("line one\nline two")
    );
}

#[test]
fn should_find_no_mlx_source_in_output_without_a_fence() {
    assert_eq!(
        fenced_source("Generated source code for `x`:\nno fence"),
        None
    );
}

fn repository_registered_kernels() -> Vec<RegisteredKernel> {
    // Read at run time, not compile time: a test binary run from a nextest
    // archive on another runner gets the workspace's new place there.
    let manifest_directory = std::env::var("CARGO_MANIFEST_DIR")
        .map_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")), PathBuf::from);
    let repository_root = repository_root_above(&manifest_directory)
        .expect("the acceptance crate is in the workspace");
    let shared_header_directory = repository_root.join(SHARED_HEADER_DIRECTORY);
    let mut registered_kernels = Vec::new();
    for kernel_file in kernel_folder_files(&repository_root.join("crates")) {
        let file_name = kernel_file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if file_name.ends_with(".metal") && !file_name.ends_with("_body.metal") {
            registered_kernels.push(
                read_registered_kernel(&kernel_file, &shared_header_directory)
                    .expect("the kernel file reads"),
            );
        }
    }
    registered_kernels
}

/// The source MLX prints for one launch of the kernel.
fn mlx_generated_source(
    runtime: &MlxRuntime,
    registered_kernel: &RegisteredKernel,
    instantiation: &KernelInstantiation,
) -> String {
    let kernel = declared_kernel(registered_kernel, instantiation);
    let input_arrays = input_arrays(runtime, instantiation);
    let input_references: Vec<&MlxArray> = input_arrays.iter().collect();
    let mut output_specs = Vec::new();
    for (_, output_dtype) in &instantiation.outputs {
        output_specs.push(MlxMetalKernelOutput::new(
            vec![DEVICE_INPUT_ELEMENT_COUNT],
            mlx_dtype(output_dtype),
        ));
    }
    let mut kernel_config = MlxMetalKernelConfig::new(&output_specs, [1, 1, 1], [1, 1, 1], None)
        .expect("the launch configures");
    kernel_config
        .add_template_arguments(&template_arguments(&instantiation.template_arguments))
        .expect("the template arguments are accepted");
    kernel_config
        .set_verbose(true)
        .expect("MLX's verbose flag is set");
    let printed_output = captured_standard_output(|| {
        runtime
            .apply_metal_kernel_with_config(&kernel, &input_references, &kernel_config)
            .expect("MLX records the launch");
    });
    fenced_source(&printed_output).unwrap_or_else(|| {
        panic!(
            "MLX printed no source for {}: {printed_output}",
            registered_kernel.name
        )
    })
}

fn declared_kernel(
    registered_kernel: &RegisteredKernel,
    instantiation: &KernelInstantiation,
) -> MlxMetalKernel {
    let mut input_names = Vec::new();
    for input in &registered_kernel.signature.inputs {
        input_names.push(input.name.as_str());
    }
    let mut output_names = Vec::new();
    for output in &registered_kernel.signature.outputs {
        output_names.push(output.name.as_str());
    }
    MlxMetalKernel::new_with_options(
        &registered_kernel.name,
        &input_names,
        &output_names,
        &instantiation.header_text,
        &instantiation.source_text,
        registered_kernel.signature.has_atomic_outputs,
    )
    .expect("the kernel declares")
}

fn input_arrays(runtime: &MlxRuntime, instantiation: &KernelInstantiation) -> Vec<MlxArray> {
    let mut input_arrays = Vec::new();
    for input in &instantiation.baseline_inputs {
        let shape: &[i32] = match input.address {
            ArgumentAddress::Device => &[DEVICE_INPUT_ELEMENT_COUNT],
            ArgumentAddress::Constant => &[1],
            ArgumentAddress::ConstantReference => &[],
        };
        input_arrays.push(
            runtime
                .zeros(shape, mlx_dtype(&input.dtype))
                .expect("the input array is created"),
        );
    }
    input_arrays
}

/// The bridge holds template names for the life of the program.
fn template_arguments(
    wrapped_arguments: &[WrappedTemplateArgument],
) -> Vec<MlxMetalKernelTemplateArgument> {
    let mut template_arguments = Vec::new();
    for wrapped_argument in wrapped_arguments {
        let name: &'static str = Box::leak(wrapped_argument.name.clone().into_boxed_str());
        template_arguments.push(match wrapped_argument.kind_word {
            "typename" => MlxMetalKernelTemplateArgument::Dtype {
                name,
                dtype: mlx_dtype(&wrapped_argument.value),
            },
            "bool" => MlxMetalKernelTemplateArgument::Boolean {
                name,
                boolean_template_argument: wrapped_argument.value == "1",
            },
            _ => MlxMetalKernelTemplateArgument::Integer {
                name,
                integer_template_argument: wrapped_argument
                    .value
                    .parse()
                    .expect("an integer template value"),
            },
        });
    }
    template_arguments
}

/// The Metal type names the signature blocks use, with the MLX dtype of each.
const METAL_TYPE_DTYPES: [(&str, MlxDtype); 5] = [
    ("bfloat16_t", MlxDtype::BFloat16),
    ("float16_t", MlxDtype::Float16),
    ("float", MlxDtype::Float32),
    ("uint32_t", MlxDtype::UInt32),
    ("int32_t", MlxDtype::Int32),
];

fn mlx_dtype(metal_type: &str) -> MlxDtype {
    for (type_name, dtype) in METAL_TYPE_DTYPES {
        if type_name == metal_type {
            return dtype;
        }
    }
    panic!("no MLX dtype for the Metal type {metal_type}; add it to METAL_TYPE_DTYPES");
}

/// The port's text for the same launch, under the function name MLX chose
/// (its name carries a hash of the template values and the dtypes).
fn rebuilt_source(
    registered_kernel: &RegisteredKernel,
    instantiation: &KernelInstantiation,
    mlx_source: &str,
) -> String {
    let function_name = mlx_source
        .split_once("[[kernel]] void ")
        .and_then(|(_, after_keyword)| after_keyword.split_once('('))
        .map(|(function_name, _)| function_name)
        .expect("MLX's source declares a kernel function");
    mlx_kernel_source(&WrapperRequest {
        function_name,
        header: &instantiation.header_text,
        source: &instantiation.source_text,
        inputs: &instantiation.baseline_inputs,
        outputs: &instantiation.outputs,
        template_arguments: &instantiation.template_arguments,
        has_atomic_outputs: registered_kernel.signature.has_atomic_outputs,
    })
}

/// The text between MLX's two code fences: `` ```\n{source}\n```\n ``.
fn fenced_source(printed_output: &str) -> Option<String> {
    let (_, after_opening) = printed_output.split_once(MLX_SOURCE_FENCE)?;
    let (source, _) = after_opening.rsplit_once(&format!("\n{MLX_SOURCE_FENCE}"))?;
    Some(source.to_owned())
}

fn first_difference(kernel_name: &str, mlx_source: &str, rebuilt_source: &str) -> String {
    for (line_index, (mlx_line, rebuilt_line)) in
        mlx_source.lines().zip(rebuilt_source.lines()).enumerate()
    {
        if mlx_line != rebuilt_line {
            return format!(
                "{kernel_name} line {}: MLX `{mlx_line}`, port `{rebuilt_line}`",
                line_index + 1
            );
        }
    }
    format!(
        "{kernel_name}: MLX's source has {} lines, the port's {}",
        mlx_source.lines().count(),
        rebuilt_source.lines().count()
    )
}

/// Everything written to the process's standard output (file descriptor 1)
/// while `print` runs. MLX prints through C++'s `std::cout`, which Rust's
/// test capture does not see, so the descriptor itself is redirected.
fn captured_standard_output(print: impl FnOnce()) -> String {
    let mut capture_file = tempfile::tempfile().expect("a capture file");
    // SAFETY: fflush, dup and dup2 act on valid descriptors this function
    // owns or restores before returning; descriptor 1 is put back even when
    // nothing was written.
    let saved_standard_output = unsafe {
        libc::fflush(std::ptr::null_mut());
        let saved_descriptor = libc::dup(1);
        libc::dup2(capture_file.as_raw_fd(), 1);
        saved_descriptor
    };
    print();
    // SAFETY: as above.
    unsafe {
        libc::fflush(std::ptr::null_mut());
        libc::dup2(saved_standard_output, 1);
        libc::close(saved_standard_output);
    }
    let mut captured_output = String::new();
    std::io::Seek::rewind(&mut capture_file).expect("the capture file rewinds");
    capture_file
        .read_to_string(&mut captured_output)
        .expect("the capture is text");
    captured_output
}
