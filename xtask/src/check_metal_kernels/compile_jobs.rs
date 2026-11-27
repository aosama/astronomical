//! The offline compiles each registered kernel needs: every `@compile-with`
//! instantiation, with each input passed every way MLX may pass it, at every
//! Metal language version MLX may compile it with.

use metal_kernel_declarations::kernel_files::RegisteredKernel;
use metal_kernel_declarations::kernel_instantiation::{
    KernelInstantiation, kernel_instantiation, kernel_instantiation_values, resolved_dtype,
};
use metal_kernel_declarations::mlx_wrapper::{
    ArgumentAddress, WrappedInput, WrapperRequest, mlx_kernel_source,
};
use metal_kernel_declarations::signature_block::InputPassing;

/// The Metal language versions MLX v0.32.5 compiles custom kernels with on
/// the macOS versions Astronomical supports: 4.0 on macOS 26 (the pinned
/// deployment target is 26.2) and 4.1 on macOS 27 (`get_metal_version` in
/// MLX's `mlx/backend/metal/device.cpp`).
pub(crate) const MLX_LANGUAGE_VERSIONS: [&str; 2] = ["metal4.0", "metal4.1"];

/// MLX's run-time prelude, `metal::utils()`, is its `utils.h` expanded at
/// MLX's build time; including the provisioned header gives the same text.
const MLX_PRELUDE_INCLUDE: &str = "#include \"mlx/backend/metal/kernels/utils.h\"\n";

/// One offline compile: a complete source and the language it is compiled as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompileJob {
    /// What a failure is reported as: kernel, instantiation and variant.
    pub(crate) description: String,
    pub(crate) language_version: &'static str,
    pub(crate) full_source: String,
}

/// One way the inputs are passed in a launch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct InputVariant {
    description: String,
    inputs: Vec<WrappedInput>,
}

/// Every compile job of `registered_kernel`.
///
/// # Errors
/// Fails when a header, body or source file cannot be read.
pub(crate) fn kernel_compile_jobs(
    registered_kernel: &RegisteredKernel,
) -> Result<Vec<CompileJob>, String> {
    let mut compile_jobs = Vec::new();
    for (instantiation_index, instantiation_values) in
        kernel_instantiation_values(registered_kernel)
            .iter()
            .enumerate()
    {
        let instantiation = kernel_instantiation(registered_kernel, instantiation_values)?;
        for input_variant in input_variants(registered_kernel, &instantiation) {
            let full_source = full_kernel_source(registered_kernel, &instantiation, &input_variant);
            for language_version in MLX_LANGUAGE_VERSIONS {
                compile_jobs.push(CompileJob {
                    description: format!(
                        "{} instantiation {} ({}) {language_version}",
                        registered_kernel.name,
                        instantiation_index + 1,
                        input_variant.description
                    ),
                    language_version,
                    full_source: full_source.clone(),
                });
            }
        }
    }
    Ok(compile_jobs)
}

fn full_kernel_source(
    registered_kernel: &RegisteredKernel,
    instantiation: &KernelInstantiation,
    input_variant: &InputVariant,
) -> String {
    let function_name = format!("custom_kernel_{}", registered_kernel.name);
    let wrapped_source = mlx_kernel_source(&WrapperRequest {
        function_name: &function_name,
        header: &instantiation.header_text,
        source: &instantiation.source_text,
        inputs: &input_variant.inputs,
        outputs: &instantiation.outputs,
        template_arguments: &instantiation.template_arguments,
        has_atomic_outputs: registered_kernel.signature.has_atomic_outputs,
    });
    format!("{MLX_PRELUDE_INCLUDE}{wrapped_source}")
}

/// The baseline, then one variant per small-capable input passed in
/// `constant` memory, then one per alternative dtype: each departure from
/// the baseline is compiled once, which finds a body that cannot take it
/// without compiling every product.
fn input_variants(
    registered_kernel: &RegisteredKernel,
    instantiation: &KernelInstantiation,
) -> Vec<InputVariant> {
    let baseline_inputs = &instantiation.baseline_inputs;
    let mut input_variants = vec![InputVariant {
        description: "baseline".to_owned(),
        inputs: baseline_inputs.clone(),
    }];
    for (input_index, input) in registered_kernel.signature.inputs.iter().enumerate() {
        if input.passing == InputPassing::MayBeSmall {
            let mut small_inputs = baseline_inputs.clone();
            small_inputs[input_index].address = ArgumentAddress::Constant;
            input_variants.push(InputVariant {
                description: format!("{} in constant memory", input.name),
                inputs: small_inputs,
            });
        }
        for alternative_dtype in &input.dtype_choices[1..] {
            let mut retyped_inputs = baseline_inputs.clone();
            retyped_inputs[input_index].dtype =
                resolved_dtype(alternative_dtype, &instantiation.template_arguments);
            input_variants.push(InputVariant {
                description: format!("{} as {alternative_dtype}", input.name),
                inputs: retyped_inputs,
            });
        }
    }
    input_variants
}
