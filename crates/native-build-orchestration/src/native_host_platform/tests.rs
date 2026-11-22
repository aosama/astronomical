use super::{
    MLX_LINK_LIBRARIES_RELATIVE_PATH, NativeTargetPlatform, linux_compiler_configure_arguments,
    native_build_platform_for_host, platform_native_products,
};

#[test]
fn should_accept_a_target_for_the_host_operating_system() {
    assert_eq!(
        native_build_platform_for_host(
            "x86_64-unknown-linux-gnu",
            "x86_64-unknown-linux-gnu",
            Some(NativeTargetPlatform::Linux)
        )
        .expect("a Linux host builds for Linux"),
        NativeTargetPlatform::Linux
    );
    assert_eq!(
        native_build_platform_for_host(
            "aarch64-apple-darwin",
            "aarch64-apple-darwin",
            Some(NativeTargetPlatform::MacOs)
        )
        .expect("a macOS host builds for macOS"),
        NativeTargetPlatform::MacOs
    );
}

#[test]
fn should_refuse_a_target_for_another_operating_system_than_the_host() {
    let refusal = native_build_platform_for_host(
        "aarch64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        Some(NativeTargetPlatform::Linux),
    )
    .expect_err("a Linux host cannot build the Metal runtime");
    assert!(refusal.to_string().contains("host's operating system"));
}

#[test]
fn should_refuse_every_target_on_a_host_without_a_native_recipe() {
    assert!(
        native_build_platform_for_host("x86_64-unknown-linux-gnu", "x86_64-unknown-freebsd", None)
            .is_err()
    );
}

#[test]
fn should_refuse_a_target_operating_system_without_a_native_recipe() {
    let refusal = native_build_platform_for_host(
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
        Some(NativeTargetPlatform::Linux),
    )
    .expect_err("Windows has no native recipe yet");
    assert!(refusal.to_string().contains("macOS or Linux"));
}

#[test]
fn should_refuse_a_linux_target_for_another_architecture_or_c_library() {
    for foreign_linux_triple in ["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-musl"] {
        let refusal = native_build_platform_for_host(
            foreign_linux_triple,
            "x86_64-unknown-linux-gnu",
            Some(NativeTargetPlatform::Linux),
        )
        .expect_err("CMake would build host archives under the foreign identity");
        assert!(refusal.to_string().contains("host's own triple"));
    }
}

#[test]
fn should_keep_accepting_an_intel_macos_triple_on_an_apple_silicon_host_as_before() {
    assert_eq!(
        native_build_platform_for_host(
            "x86_64-apple-darwin",
            "aarch64-apple-darwin",
            Some(NativeTargetPlatform::MacOs)
        )
        .expect("macOS accepts the triples it accepted before"),
        NativeTargetPlatform::MacOs
    );
}

#[test]
fn should_publish_the_metallib_on_macos_and_the_link_library_list_on_linux() {
    let macos_entry_paths: Vec<&str> = platform_native_products(NativeTargetPlatform::MacOs)
        .iter()
        .map(|platform_product| platform_product.entry_relative_path)
        .collect();
    let linux_entry_paths: Vec<&str> = platform_native_products(NativeTargetPlatform::Linux)
        .iter()
        .map(|platform_product| platform_product.entry_relative_path)
        .collect();
    assert_eq!(macos_entry_paths, ["share/mlx/mlx.metallib"]);
    assert_eq!(linux_entry_paths, [MLX_LINK_LIBRARIES_RELATIVE_PATH]);
}

#[test]
fn should_configure_linux_with_the_compilers_the_identity_probes() {
    assert_eq!(
        linux_compiler_configure_arguments(),
        ["-DCMAKE_C_COMPILER=cc", "-DCMAKE_CXX_COMPILER=c++"]
    );
}
