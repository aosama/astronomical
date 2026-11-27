use std::path::PathBuf;

use crate::generated_declarations::declarations_source;
use crate::kernel_files::RegisteredKernel;
use crate::signature_block::parse_signature_block;

fn registered_pair() -> RegisteredKernel {
    RegisteredKernel {
        name: "astro_primitive_pair".to_owned(),
        kernel_file: PathBuf::from("/repo/metal/astro_primitive_pair.metal"),
        signature: parse_signature_block(
            "// @input X float\n// @input count int32_t always-small\n// @output Y float\n// @template N int\n// @define SCALE\n// @compile-with N=4 SCALE=1.0f\n",
        )
        .expect("the block parses"),
        header_files: vec![PathBuf::from("/repo/shared/astro_primitive_matrix_units.h")],
        source_files: vec![
            PathBuf::from("/repo/metal/astro_primitive_pair.metal"),
            PathBuf::from("/repo/metal/astro_primitive_pair_body.metal"),
        ],
    }
}

#[test]
fn should_declare_names_in_order_and_include_the_files_themselves() {
    let declarations_text =
        declarations_source(&[registered_pair()]).expect("the declarations are written");
    assert!(declarations_text.contains(
        "pub const ASTRO_PRIMITIVE_PAIR: ::metal_kernel_declarations::MetalKernelSource"
    ));
    assert!(declarations_text.contains("    name: \"astro_primitive_pair\",\n"));
    assert!(declarations_text.contains("    input_names: &[\"X\", \"count\"],\n"));
    assert!(declarations_text.contains("    output_names: &[\"Y\"],\n"));
    assert!(declarations_text.contains("    template_parameter_names: &[\"N\"],\n"));
    assert!(declarations_text.contains("    define_names: &[\"SCALE\"],\n"));
    assert!(declarations_text.contains("    has_atomic_outputs: false,\n"));
    assert!(declarations_text.contains(
        "    source_parts: &[include_str!(\"/repo/metal/astro_primitive_pair.metal\"), include_str!(\"/repo/metal/astro_primitive_pair_body.metal\")],\n"
    ));
}

#[test]
fn should_write_only_the_banner_when_a_crate_has_no_kernels() {
    let declarations_text = declarations_source(&[]).expect("an empty set is written");
    assert_eq!(declarations_text.lines().count(), 1);
}
