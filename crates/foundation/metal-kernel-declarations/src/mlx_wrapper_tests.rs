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

// The expected texts are Metal, so they live in `mlx_wrapper_fixtures/`
// rather than in Rust strings (`cargo xtask check-metal-kernels` refuses
// program text inside Rust). Each was checked against MLX's own output by the
// `metal_kernel_wrapper_parity` contract's rules.

fn float_output() -> [(String, String); 1] {
    [("Y".to_owned(), "float".to_owned())]
}

#[test]
fn should_write_the_signature_mlx_writes_for_a_plain_copy() {
    let inputs = [wrapped_input("X", "float", ArgumentAddress::Device)];
    let source = include_str!("mlx_wrapper_fixtures/plain_copy_body.metal");
    assert_eq!(
        mlx_kernel_source(&plain_request(source, &inputs, &float_output(), &[])),
        include_str!("mlx_wrapper_fixtures/plain_copy_expected.metal")
    );
}

#[test]
fn should_pass_small_inputs_in_constant_memory_and_scalars_by_reference() {
    let inputs = [
        wrapped_input("count", "int32_t", ArgumentAddress::Constant),
        wrapped_input("scale", "float", ArgumentAddress::ConstantReference),
    ];
    let kernel_source = mlx_kernel_source(&plain_request(
        "Y[0] = scale * count[0];",
        &inputs,
        &float_output(),
        &[],
    ));
    assert_eq!(
        kernel_source,
        include_str!("mlx_wrapper_fixtures/small_inputs_expected.metal")
    );
}

#[test]
fn should_add_shape_buffers_only_for_names_the_source_mentions() {
    let inputs = [wrapped_input("X", "float", ArgumentAddress::Device)];
    let source = include_str!("mlx_wrapper_fixtures/shape_buffers_body.metal");
    let kernel_source = mlx_kernel_source(&plain_request(source, &inputs, &float_output(), &[]));
    assert!(!kernel_source.contains("X_strides"));
    assert_eq!(
        kernel_source,
        include_str!("mlx_wrapper_fixtures/shape_buffers_expected.metal"),
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
    assert_eq!(
        kernel_source,
        include_str!("mlx_wrapper_fixtures/template_expected.metal")
    );
}

#[test]
fn should_wrap_atomic_outputs() {
    let outputs = float_output();
    let mut request = plain_request("Y[0] = 0;", &[], &outputs, &[]);
    request.has_atomic_outputs = true;
    assert_eq!(
        mlx_kernel_source(&request),
        include_str!("mlx_wrapper_fixtures/atomic_expected.metal")
    );
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
