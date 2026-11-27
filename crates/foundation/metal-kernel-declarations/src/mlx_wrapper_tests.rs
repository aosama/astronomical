use crate::mlx_wrapper::{
    ArgumentAddress, WrappedInput, WrappedTemplateArgument, WrapperRequest, attribute_parameters,
    mlx_kernel_source,
};

fn wrapped_input(name: &str, dtype: &str, address: ArgumentAddress) -> WrappedInput {
    WrappedInput {
        name: name.to_owned(),
        dtype: dtype.to_owned(),
        address,
    }
}

fn plain_request<'text>(
    source: &'text str,
    inputs: &'text [WrappedInput],
    outputs: &'text [(String, String)],
    template_arguments: &'text [WrappedTemplateArgument],
) -> WrapperRequest<'text> {
    WrapperRequest {
        function_name: "custom_kernel_copy",
        header: "",
        source,
        inputs,
        outputs,
        template_arguments,
        has_atomic_outputs: false,
    }
}

#[test]
fn should_write_the_signature_mlx_writes_for_a_plain_copy() {
    let inputs = [wrapped_input("X", "float", ArgumentAddress::Device)];
    let outputs = [("Y".to_owned(), "float".to_owned())];
    let source = "  Y[thread_position_in_grid.x] = X[thread_position_in_grid.x];";
    let kernel_source = mlx_kernel_source(&plain_request(source, &inputs, &outputs, &[]));
    assert_eq!(
        kernel_source,
        "[[kernel]] void custom_kernel_copy(\n  const device float* X [[buffer(0)]],\n  device float* Y [[buffer(1)]],\n  uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n  Y[thread_position_in_grid.x] = X[thread_position_in_grid.x];\n}\n"
    );
}

#[test]
fn should_pass_small_inputs_in_constant_memory_and_scalars_by_reference() {
    let inputs = [
        wrapped_input("count", "int32_t", ArgumentAddress::Constant),
        wrapped_input("scale", "float", ArgumentAddress::ConstantReference),
    ];
    let outputs = [("Y".to_owned(), "float".to_owned())];
    let kernel_source = mlx_kernel_source(&plain_request(
        "Y[0] = scale * count[0];",
        &inputs,
        &outputs,
        &[],
    ));
    assert!(kernel_source.contains("  const constant int32_t* count [[buffer(0)]],\n"));
    assert!(kernel_source.contains("  const constant float& scale [[buffer(1)]],\n"));
    assert!(
        kernel_source.contains("  device float* Y [[buffer(2)]]) {\n"),
        "{kernel_source}"
    );
}

#[test]
fn should_add_shape_buffers_only_for_names_the_source_mentions() {
    let inputs = [wrapped_input("X", "float", ArgumentAddress::Device)];
    let outputs = [("Y".to_owned(), "float".to_owned())];
    let kernel_source = mlx_kernel_source(&plain_request(
        "Y[0] = X[0] * X_shape[0] * X_ndim; // thread_position_in_grid",
        &inputs,
        &outputs,
        &[],
    ));
    assert!(kernel_source.contains("  const constant int* X_shape [[buffer(1)]],\n"));
    assert!(!kernel_source.contains("X_strides"));
    assert!(kernel_source.contains("  const constant int& X_ndim [[buffer(2)]],\n"));
    assert!(
        kernel_source.contains("  uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n"),
        "an attribute named only in a comment is still declared, as MLX does"
    );
}

#[test]
fn should_write_the_template_line_and_the_explicit_instantiation() {
    let inputs = [wrapped_input("X", "bfloat16_t", ArgumentAddress::Device)];
    let outputs = [("Y".to_owned(), "bfloat16_t".to_owned())];
    let template_arguments = [
        WrappedTemplateArgument {
            name: "N".to_owned(),
            kind_word: "int",
            value: "-4".to_owned(),
        },
        WrappedTemplateArgument {
            name: "STAGE".to_owned(),
            kind_word: "bool",
            value: "1".to_owned(),
        },
        WrappedTemplateArgument {
            name: "T".to_owned(),
            kind_word: "typename",
            value: "bfloat16_t".to_owned(),
        },
    ];
    let kernel_source = mlx_kernel_source(&plain_request(
        "Y[0] = X[0];",
        &inputs,
        &outputs,
        &template_arguments,
    ));
    assert!(kernel_source.starts_with(
        "template <int N, bool STAGE, typename T>\n[[kernel]] void custom_kernel_copy(\n"
    ));
    assert!(kernel_source.ends_with(
        "\ntemplate [[host_name(\"custom_kernel_copy\")]] [[kernel]] decltype(custom_kernel_copy<-4, 1, bfloat16_t>) custom_kernel_copy<-4, 1, bfloat16_t>;\n"
    ));
}

#[test]
fn should_wrap_atomic_outputs() {
    let outputs = [("Y".to_owned(), "float".to_owned())];
    let mut request = plain_request("Y[0] = 0;", &[], &outputs, &[]);
    request.has_atomic_outputs = true;
    assert!(mlx_kernel_source(&request).contains("  device atomic<float>* Y [[buffer(0)]]) {\n"));
}

#[test]
fn should_find_both_attributes_when_one_name_contains_the_other() {
    let attribute_lines = attribute_parameters("dispatch_threads_per_threadgroup.x");
    assert_eq!(
        attribute_lines,
        [
            "  uint3 dispatch_threads_per_threadgroup [[dispatch_threads_per_threadgroup]]",
            "  uint3 threads_per_threadgroup [[threads_per_threadgroup]]"
        ],
        "MLX's substring scan finds threads_per_threadgroup inside the longer name"
    );
}
