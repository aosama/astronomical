//! One `@compile-with` instantiation of a registered kernel, assembled as a
//! launcher hands it to MLX: the header and source texts, the template
//! arguments in MLX's order, and every buffer with its dtype resolved.
//!
//! The offline compile and the contract that compares this crate's wrapper
//! with the one MLX prints both start here, so they judge the same text.

use std::fs;
use std::path::PathBuf;

use crate::kernel_files::RegisteredKernel;
use crate::mlx_wrapper::{ArgumentAddress, WrappedInput, WrappedTemplateArgument};
use crate::signature_block::{InputPassing, InstantiationValues, TemplateKind};

/// What MLX receives for one instantiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelInstantiation {
    pub header_text: String,
    /// The defines as `#define` lines, then the source files.
    pub source_text: String,
    pub template_arguments: Vec<WrappedTemplateArgument>,
    /// Every input at its first dtype and the address its passing gives it
    /// in the usual launch.
    pub baseline_inputs: Vec<WrappedInput>,
    /// `(name, dtype)` of every output.
    pub outputs: Vec<(String, String)>,
}

/// The instantiations a kernel is compiled for: its `@compile-with` lines,
/// or one with no values when it has no template parameters or defines.
pub fn kernel_instantiation_values(
    registered_kernel: &RegisteredKernel,
) -> Vec<InstantiationValues> {
    if registered_kernel.signature.instantiations.is_empty() {
        vec![InstantiationValues::new()]
    } else {
        registered_kernel.signature.instantiations.clone()
    }
}

/// Assembles one instantiation of `registered_kernel`.
///
/// # Errors
/// Fails when a header, body or source file cannot be read.
pub fn kernel_instantiation(
    registered_kernel: &RegisteredKernel,
    instantiation: &InstantiationValues,
) -> Result<KernelInstantiation, String> {
    let template_arguments = template_arguments(registered_kernel, instantiation);
    let mut baseline_inputs = Vec::new();
    for input in &registered_kernel.signature.inputs {
        baseline_inputs.push(WrappedInput {
            name: input.name.clone(),
            dtype: resolved_dtype(&input.dtype_choices[0], &template_arguments),
            address: baseline_address(input.passing),
        });
    }
    let mut outputs = Vec::new();
    for output in &registered_kernel.signature.outputs {
        outputs.push((
            output.name.clone(),
            resolved_dtype(&output.dtype, &template_arguments),
        ));
    }
    let source_text = format!(
        "{}{}",
        define_lines(registered_kernel, instantiation),
        concatenated_files(&registered_kernel.source_files)?
    );
    Ok(KernelInstantiation {
        header_text: concatenated_files(&registered_kernel.header_files)?,
        source_text,
        template_arguments,
        baseline_inputs,
        outputs,
    })
}

/// A dtype the block writes as a `typename` template parameter's name
/// (`@output Y OutputT`) is that parameter's value in this instantiation, as
/// the launcher passes the array's own dtype for both.
pub fn resolved_dtype(
    declared_dtype: &str,
    template_arguments: &[WrappedTemplateArgument],
) -> String {
    for template_argument in template_arguments {
        if template_argument.kind_word == "typename" && template_argument.name == declared_dtype {
            return template_argument.value.clone();
        }
    }
    declared_dtype.to_owned()
}

/// The address MLX gives an input in the usual launch: a small-capable input
/// is usually large, so it starts in device memory.
pub fn baseline_address(passing: InputPassing) -> ArgumentAddress {
    match passing {
        InputPassing::Device | InputPassing::MayBeSmall => ArgumentAddress::Device,
        InputPassing::AlwaysSmall => ArgumentAddress::Constant,
        InputPassing::Scalar => ArgumentAddress::ConstantReference,
    }
}

fn concatenated_files(files: &[PathBuf]) -> Result<String, String> {
    let mut concatenated_text = String::new();
    for included_file in files {
        concatenated_text.push_str(
            &fs::read_to_string(included_file)
                .map_err(|read_error| format!("{}: {read_error}", included_file.display()))?,
        );
    }
    Ok(concatenated_text)
}

/// The `#define` lines a launcher's defines become, in declaration order, as
/// `MetalKernelSource::source_text` writes them.
fn define_lines(
    registered_kernel: &RegisteredKernel,
    instantiation: &InstantiationValues,
) -> String {
    let mut define_lines = String::new();
    for define_name in &registered_kernel.signature.define_names {
        if let Some(define_value) = assigned_value(instantiation, define_name) {
            define_lines.push_str(&format!("#define {define_name} {define_value}\n"));
        }
    }
    define_lines
}

fn template_arguments(
    registered_kernel: &RegisteredKernel,
    instantiation: &InstantiationValues,
) -> Vec<WrappedTemplateArgument> {
    let mut template_arguments = Vec::new();
    for template_parameter in &registered_kernel.signature.template_parameters {
        let assigned = assigned_value(instantiation, &template_parameter.name).unwrap_or_default();
        let (kind_word, value) = match template_parameter.kind {
            TemplateKind::Int => ("int", assigned.to_owned()),
            // MLX streams a bool into an ostringstream, which writes 1 or 0.
            TemplateKind::Bool => (
                "bool",
                if assigned == "true" || assigned == "1" {
                    "1"
                } else {
                    "0"
                }
                .to_owned(),
            ),
            TemplateKind::Typename => ("typename", assigned.to_owned()),
        };
        template_arguments.push(WrappedTemplateArgument {
            name: template_parameter.name.clone(),
            kind_word,
            value,
        });
    }
    template_arguments
}

fn assigned_value<'values>(
    instantiation: &'values InstantiationValues,
    name: &str,
) -> Option<&'values str> {
    for (assigned_name, assigned_value) in instantiation {
        if assigned_name == name {
            return Some(assigned_value);
        }
    }
    None
}
