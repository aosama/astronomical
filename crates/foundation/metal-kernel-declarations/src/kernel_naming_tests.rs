use crate::kernel_naming::{
    KernelFileKind, KernelScope, crate_kernel_scope, declaration_constant_name, kernel_file_kind,
};

#[test]
fn should_give_model_primitives_the_primitive_scope() {
    let crate_scope =
        crate_kernel_scope("crates/engine/model-primitives").expect("primitives own kernels");
    assert_eq!(crate_scope.scope, KernelScope::Primitive);
    assert_eq!(crate_scope.name_prefix(), "astro_primitive_");
}

#[test]
fn should_give_a_family_crate_its_family_as_owner() {
    let crate_scope =
        crate_kernel_scope("crates/families/text/qwen35/").expect("a family owns kernels");
    assert_eq!(crate_scope.scope, KernelScope::Family);
    assert_eq!(crate_scope.name_prefix(), "astro_family_qwen35_");
}

#[test]
fn should_give_a_format_crate_its_name_up_to_the_first_dash() {
    let crate_scope =
        crate_kernel_scope("crates/weight-formats/jangtq2-qwen35").expect("a format owns kernels");
    assert_eq!(crate_scope.scope, KernelScope::Format);
    assert_eq!(crate_scope.name_prefix(), "astro_format_jangtq2_");
}

#[test]
fn should_give_no_scope_to_crates_that_may_own_no_kernels() {
    assert_eq!(crate_kernel_scope("crates/engine/model-serving"), None);
    assert_eq!(crate_kernel_scope("crates/families/text"), None);
    assert_eq!(
        crate_kernel_scope("crates/weight-formats/jangtq2-qwen35/tests"),
        None
    );
}

#[test]
fn should_tell_a_registered_kernel_a_shared_body_and_a_helper_apart() {
    let prefix = "astro_primitive_";
    assert_eq!(
        kernel_file_kind("astro_primitive_sorted_expert_matmul_4bit.metal", prefix),
        Ok(KernelFileKind::RegisteredKernel)
    );
    assert_eq!(
        kernel_file_kind("astro_primitive_sorted_expert_matmul_body.metal", prefix),
        Ok(KernelFileKind::SharedBody)
    );
    assert_eq!(
        kernel_file_kind("astro_primitive_matrix_units.h", prefix),
        Ok(KernelFileKind::Helper)
    );
}

#[test]
fn should_refuse_a_file_with_another_scope_prefix() {
    let refusal = kernel_file_kind("astro_family_qwen35_gated_delta.metal", "astro_primitive_")
        .expect_err("a family name in the primitive crate is refused");
    assert!(refusal.contains("astro_primitive_"));
}

#[test]
fn should_refuse_names_without_an_operation_or_with_capitals() {
    assert!(kernel_file_kind("astro_primitive_.metal", "astro_primitive_").is_err());
    assert!(kernel_file_kind("astro_primitive_body.metal", "astro_primitive_").is_err());
    assert!(kernel_file_kind("astro_primitive_TopK.metal", "astro_primitive_").is_err());
}

#[test]
fn should_refuse_other_extensions_in_a_metal_folder() {
    assert!(kernel_file_kind("astro_primitive_notes.txt", "astro_primitive_").is_err());
}

#[test]
fn should_name_the_declaration_constant_in_capitals() {
    assert_eq!(
        declaration_constant_name("astro_primitive_vocabulary_top_k"),
        "ASTRO_PRIMITIVE_VOCABULARY_TOP_K"
    );
}
