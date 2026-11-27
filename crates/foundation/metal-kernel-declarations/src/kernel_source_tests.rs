use crate::kernel_source::{MetalKernelDefine, MetalKernelSource};

const SCALED_COPY: MetalKernelSource = MetalKernelSource {
    name: "astro_primitive_scaled_copy",
    input_names: &["X"],
    output_names: &["Y"],
    template_parameter_names: &[],
    define_names: &["SCALE", "OFFSET"],
    has_atomic_outputs: false,
    header_parts: &["// first header\n", "// second header\n"],
    source_parts: &["// kernel file\n", "Y[0] = X[0] * SCALE + OFFSET;\n"],
};

fn define(name: &'static str, value: &str) -> MetalKernelDefine {
    MetalKernelDefine {
        name,
        value: value.to_owned(),
    }
}

#[test]
fn should_join_the_header_files_in_order() {
    assert_eq!(
        SCALED_COPY.header_text(),
        "// first header\n// second header\n"
    );
}

#[test]
fn should_write_defines_in_declaration_order_ahead_of_the_source() {
    let source_text = SCALED_COPY
        .source_text(&[define("OFFSET", "1.0f"), define("SCALE", "2.0f")])
        .expect("both defines are supplied");
    assert_eq!(
        source_text,
        "#define SCALE 2.0f\n#define OFFSET 1.0f\n// kernel file\nY[0] = X[0] * SCALE + OFFSET;\n"
    );
}

#[test]
fn should_refuse_a_missing_define() {
    let refusal = SCALED_COPY
        .source_text(&[define("SCALE", "2.0f")])
        .expect_err("OFFSET is missing");
    assert!(refusal.description.contains("OFFSET"), "{refusal}");
}

#[test]
fn should_refuse_a_define_supplied_twice() {
    let refusal = SCALED_COPY
        .source_text(&[
            define("SCALE", "2.0f"),
            define("SCALE", "3.0f"),
            define("OFFSET", "0.0f"),
        ])
        .expect_err("SCALE twice is refused");
    assert!(refusal.description.contains("2 times"), "{refusal}");
}

#[test]
fn should_refuse_an_undeclared_define() {
    let refusal = SCALED_COPY
        .source_text(&[
            define("SCALE", "2.0f"),
            define("OFFSET", "0.0f"),
            define("BIAS", "0.0f"),
        ])
        .expect_err("BIAS is not declared");
    assert!(refusal.to_string().contains("astro_primitive_scaled_copy"));
}

#[test]
fn should_give_the_source_unchanged_when_nothing_is_defined() {
    let plain = MetalKernelSource {
        define_names: &[],
        ..SCALED_COPY
    };
    assert_eq!(
        plain.source_text(&[]).expect("no defines needed"),
        "// kernel file\nY[0] = X[0] * SCALE + OFFSET;\n"
    );
}
