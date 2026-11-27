//! The source MLX compiles for a custom kernel, rebuilt off the GPU.
//!
//! This is a line-for-line port of `write_signature`, the attribute scan and
//! the template instantiation in MLX v0.32.5's
//! `mlx/backend/common/custom_kernel.cpp`, quirks included: attributes and an
//! input's `_shape`, `_strides` and `_ndim` buffers are found by plain
//! substring search over the source only (never the header, and comments
//! count), and the comma after an output is decided by a buffer index that
//! also counts shape buffers. Reproducing MLX exactly is the point: the
//! offline compile proves what MLX will compile, and the acceptance contract
//! `metal_kernel_wrapper_parity` compares this text with the one MLX prints,
//! so an MLX pin bump that changes the wrapper fails loudly.

/// MLX's list of thread attributes, in its order, with the type it declares
/// each with. A body that names one gets it as a parameter.
pub const MLX_THREAD_ATTRIBUTES: [(&str, &str); 20] = [
    ("dispatch_quadgroups_per_threadgroup", "uint"),
    ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"),
    ("grid_origin", "uint3"),
    ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"),
    ("thread_index_in_quadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
];

/// How MLX passes one input array, decided by the array's size at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgumentAddress {
    /// 8 elements or more: `const device T*`.
    Device,
    /// 1 to 7 elements: `const constant T*`.
    Constant,
    /// 0 dimensions: `const constant T&`, with no shape buffers.
    ConstantReference,
}

/// One input as MLX will see it in one launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrappedInput {
    pub name: String,
    pub dtype: String,
    pub address: ArgumentAddress,
}

/// One template argument in MLX's order, its value written as MLX writes it
/// (`1`/`0` for a bool).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrappedTemplateArgument {
    pub name: String,
    pub kind_word: &'static str,
    pub value: String,
}

/// Everything MLX's wrapper is written from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrapperRequest<'text> {
    pub function_name: &'text str,
    pub header: &'text str,
    pub source: &'text str,
    pub inputs: &'text [WrappedInput],
    pub outputs: &'text [(String, String)],
    pub template_arguments: &'text [WrappedTemplateArgument],
    pub has_atomic_outputs: bool,
}

/// The kernel source MLX compiles: header, the generated signature, the
/// body, and the explicit instantiation when there are template arguments.
pub fn mlx_kernel_source(request: &WrapperRequest<'_>) -> String {
    let mut kernel_source = String::from(request.header);
    if !request.template_arguments.is_empty() {
        let mut template_parameters = Vec::new();
        for template_argument in request.template_arguments {
            template_parameters.push(format!(
                "{} {}",
                template_argument.kind_word, template_argument.name
            ));
        }
        kernel_source.push_str(&format!("template <{}>\n", template_parameters.join(", ")));
    }
    kernel_source.push_str(&format!("[[kernel]] void {}(\n", request.function_name));
    let buffer_count_after_inputs = write_input_parameters(&mut kernel_source, request);
    let attribute_parameters = attribute_parameters(request.source);
    write_output_parameters(
        &mut kernel_source,
        request,
        buffer_count_after_inputs,
        attribute_parameters.len(),
    );
    for (attribute_index, attribute_parameter) in attribute_parameters.iter().enumerate() {
        kernel_source.push_str(attribute_parameter);
        kernel_source.push_str(if attribute_index + 1 < attribute_parameters.len() {
            ",\n"
        } else {
            ") {\n"
        });
    }
    kernel_source.push_str(request.source);
    kernel_source.push_str("\n}\n");
    if !request.template_arguments.is_empty() {
        let template_definition =
            template_definition(request.function_name, request.template_arguments);
        kernel_source.push_str(&format!(
            "\ntemplate [[host_name(\"{}\")]] [[kernel]] decltype({template_definition}) {template_definition};\n",
            request.function_name
        ));
    }
    kernel_source
}

/// Writes each input and the shape buffers its name asks for; returns the
/// next buffer index.
fn write_input_parameters(kernel_source: &mut String, request: &WrapperRequest<'_>) -> usize {
    let mut buffer_index = 0;
    for input in request.inputs {
        let location = if input.address == ArgumentAddress::Device {
            "device"
        } else {
            "constant"
        };
        let reference = if input.address == ArgumentAddress::ConstantReference {
            "&"
        } else {
            "*"
        };
        kernel_source.push_str(&format!(
            "  const {location} {}{reference} {} [[buffer({buffer_index})]],\n",
            input.dtype, input.name
        ));
        buffer_index += 1;
        if input.address == ArgumentAddress::ConstantReference {
            continue;
        }
        for (suffix, buffer_type) in [
            ("_shape", "const constant int*"),
            ("_strides", "const constant int64_t*"),
            ("_ndim", "const constant int&"),
        ] {
            if request.source.contains(&format!("{}{suffix}", input.name)) {
                kernel_source.push_str(&format!(
                    "  {buffer_type} {}{suffix} [[buffer({buffer_index})]],\n",
                    input.name
                ));
                buffer_index += 1;
            }
        }
    }
    buffer_index
}

fn write_output_parameters(
    kernel_source: &mut String,
    request: &WrapperRequest<'_>,
    first_output_buffer_index: usize,
    attribute_count: usize,
) {
    let last_plain_buffer_index = request.inputs.len() + request.outputs.len() - 1;
    for (output_index, (output_name, output_dtype)) in request.outputs.iter().enumerate() {
        let buffer_index = first_output_buffer_index + output_index;
        let output_type = if request.has_atomic_outputs {
            format!("atomic<{output_dtype}>")
        } else {
            output_dtype.clone()
        };
        kernel_source.push_str(&format!(
            "  device {output_type}* {output_name} [[buffer({buffer_index})]]"
        ));
        // MLX compares the running buffer index, shape buffers included, with
        // the count of inputs and outputs alone; kept as MLX has it.
        kernel_source.push_str(
            if buffer_index < last_plain_buffer_index || attribute_count > 0 {
                ",\n"
            } else {
                ") {\n"
            },
        );
    }
}

/// The thread attribute parameters the source names, in MLX's order.
pub fn attribute_parameters(source: &str) -> Vec<String> {
    let mut attribute_parameters = Vec::new();
    for (attribute_name, attribute_type) in MLX_THREAD_ATTRIBUTES {
        if source.contains(attribute_name) {
            attribute_parameters.push(format!(
                "  {attribute_type} {attribute_name} [[{attribute_name}]]"
            ));
        }
    }
    attribute_parameters
}

fn template_definition(
    function_name: &str,
    template_arguments: &[WrappedTemplateArgument],
) -> String {
    let mut template_values = Vec::new();
    for template_argument in template_arguments {
        template_values.push(template_argument.value.clone());
    }
    format!("{function_name}<{}>", template_values.join(", "))
}
