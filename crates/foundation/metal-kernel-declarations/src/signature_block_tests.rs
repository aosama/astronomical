use crate::signature_block::{InputPassing, TemplateKind, parse_signature_block};

const SORTED_EXPERT_BLOCK: &str = "// Body of a kernel; this ordinary comment says: anything.
// @header astro_primitive_matrix_units.h astro_primitive_sorted_expert_matmul.h
// @body astro_primitive_sorted_expert_matmul_body.metal
// @input X bfloat16_t
// @input first_rows int32_t may-be-small
// @input row_count int32_t always-small
// @input scale float scalar
// @input activations bfloat16_t|float16_t
// @output Y bfloat16_t
// @template N int
// @template STAGE_ROWS bool
// @template T typename
// @define SCALE
// @compile-with N=512 STAGE_ROWS=true T=bfloat16_t SCALE=0.5f
// @contract direct_mlx::sorted_expert_matmul
// @probe sorted_expert_matmul
#define SOURCE_ROW(row) long(row)
// @input ignored_after_the_block int32_t
";

#[test]
fn should_read_every_fact_of_a_full_block() {
    let block = parse_signature_block(SORTED_EXPERT_BLOCK).expect("the block parses");
    assert_eq!(
        block.header_files,
        [
            "astro_primitive_matrix_units.h",
            "astro_primitive_sorted_expert_matmul.h"
        ]
    );
    assert_eq!(
        block.body_files,
        ["astro_primitive_sorted_expert_matmul_body.metal"]
    );
    assert_eq!(
        block.inputs.len(),
        5,
        "the line after the block is not read"
    );
    assert_eq!(block.inputs[1].passing, InputPassing::MayBeSmall);
    assert_eq!(block.inputs[2].passing, InputPassing::AlwaysSmall);
    assert_eq!(block.inputs[3].passing, InputPassing::Scalar);
    assert_eq!(block.inputs[0].passing, InputPassing::Device);
    assert_eq!(block.inputs[4].dtype_choices, ["bfloat16_t", "float16_t"]);
    assert_eq!(block.outputs[0].name, "Y");
    assert_eq!(block.template_parameters[1].kind, TemplateKind::Bool);
    assert_eq!(block.template_parameters[2].kind, TemplateKind::Typename);
    assert_eq!(block.define_names, ["SCALE"]);
    assert_eq!(
        block.instantiations[0][3],
        ("SCALE".to_owned(), "0.5f".to_owned())
    );
    assert_eq!(
        block.contract.as_deref(),
        Some("direct_mlx::sorted_expert_matmul")
    );
    assert_eq!(block.probe.as_deref(), Some("sorted_expert_matmul"));
}

#[test]
fn should_accept_a_kernel_with_one_input_one_output_and_nothing_else() {
    let block = parse_signature_block("// @input X float\n// @output Y float\n")
        .expect("the smallest block parses");
    assert_eq!(block.inputs.len(), 1);
    assert!(block.instantiations.is_empty());
    assert!(!block.has_atomic_outputs);
}

#[test]
fn should_refuse_a_kernel_without_outputs() {
    let refusal = parse_signature_block("// @input X float\n").expect_err("no output is refused");
    assert!(refusal.description.contains("@output"), "{refusal}");
}

#[test]
fn should_refuse_a_kernel_without_inputs() {
    let refusal = parse_signature_block("// @output Y float\n").expect_err("no input is refused");
    assert!(refusal.description.contains("@input"), "{refusal}");
}

#[test]
fn should_refuse_an_unknown_key_and_name_its_line() {
    let refusal =
        parse_signature_block("// @input A float\n// @output Y float\n// @inptu X float\n")
            .expect_err("a typo is refused");
    assert_eq!(refusal.line_number, 3);
    assert!(refusal.description.contains("@inptu"));
}

#[test]
fn should_refuse_an_unknown_input_passing() {
    let refusal = parse_signature_block("// @input X float tiny\n// @output Y float\n")
        .expect_err("an unknown passing is refused");
    assert!(refusal.description.contains("tiny"));
}

#[test]
fn should_refuse_a_buffer_name_declared_twice() {
    let refusal = parse_signature_block("// @input X float\n// @output X float\n")
        .expect_err("a repeated name is refused");
    assert!(refusal.description.contains("`X`"));
}

#[test]
fn should_refuse_template_parameters_without_an_instantiation() {
    let refusal =
        parse_signature_block("// @input X float\n// @output Y float\n// @template N int\n")
            .expect_err("nothing to compile with");
    assert!(refusal.description.contains("@compile-with"));
}

#[test]
fn should_refuse_an_instantiation_missing_a_value() {
    let refusal = parse_signature_block(
        "// @input X float\n// @output Y float\n// @template N int\n// @template K int\n// @compile-with N=4\n",
    )
    .expect_err("K has no value");
    assert!(refusal.description.contains("`K`"), "{refusal}");
}

#[test]
fn should_refuse_an_instantiation_naming_an_undeclared_parameter() {
    let refusal = parse_signature_block(
        "// @input X float\n// @output Y float\n// @template N int\n// @compile-with N=4 M=2\n",
    )
    .expect_err("M is not declared");
    assert!(refusal.description.contains("`M`"));
}

#[test]
fn should_stop_reading_at_the_first_line_that_is_not_a_comment() {
    let block = parse_signature_block("\n// @input X float\n")
        .expect_err("a blank first line ends the block");
    assert!(block.description.contains("@output"));
}
