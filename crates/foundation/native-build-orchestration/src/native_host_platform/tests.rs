use super::{
    MLX_PACKAGE_INSTALL_RELATIVE_DIRECTORY, MLX_TARGETS_RELATIVE_PATH, NativeTargetPlatform,
    identity_environment_variables, installs_mlx_package, linux_compiler_configure_arguments,
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
fn should_publish_the_metallib_on_macos_and_mlx_targets_on_linux() {
    let macos_entry_paths: Vec<&str> = platform_native_products(NativeTargetPlatform::MacOs)
        .iter()
        .map(|platform_product| platform_product.entry_relative_path)
        .collect();
    let linux_entry_paths: Vec<&str> = platform_native_products(NativeTargetPlatform::Linux)
        .iter()
        .map(|platform_product| platform_product.entry_relative_path)
        .collect();
    assert_eq!(macos_entry_paths, ["share/mlx/mlx.metallib"]);
    assert_eq!(linux_entry_paths, [MLX_TARGETS_RELATIVE_PATH]);
}

#[test]
fn should_configure_linux_with_the_compilers_the_identity_probes() {
    assert_eq!(
        linux_compiler_configure_arguments(),
        ["-DCMAKE_C_COMPILER=cc", "-DCMAKE_CXX_COMPILER=c++"]
    );
}

#[test]
fn should_install_mlx_package_only_where_the_link_reads_it() {
    assert!(installs_mlx_package(NativeTargetPlatform::Linux));
    assert!(
        !installs_mlx_package(NativeTargetPlatform::MacOs),
        "the Metal build links a fixed framework set and installs nothing"
    );
}

#[test]
fn should_publish_mlx_targets_from_where_the_package_install_puts_it() {
    let linux_build_paths: Vec<&str> = platform_native_products(NativeTargetPlatform::Linux)
        .iter()
        .map(|platform_product| platform_product.build_relative_path)
        .collect();
    assert_eq!(
        linux_build_paths,
        [format!(
            "{MLX_PACKAGE_INSTALL_RELATIVE_DIRECTORY}/{MLX_TARGETS_RELATIVE_PATH}"
        )]
    );
}

#[test]
fn should_rerun_on_the_linux_identity_inputs_and_nothing_new_on_macos() {
    let linux_variables = identity_environment_variables(NativeTargetPlatform::Linux);
    for identity_input in ["CMAKE_PREFIX_PATH", "BLA_VENDOR", "BLAS_ROOT", "MKLROOT"] {
        assert!(
            linux_variables.contains(&identity_input),
            "{identity_input}"
        );
    }
    assert_eq!(
        linux_variables,
        native_identity::probes::BLAS_SEARCH_VARIABLES
    );
    for process_varying_variable in ["PATH", "LD_LIBRARY_PATH"] {
        assert!(
            !linux_variables.contains(&process_varying_variable),
            "{process_varying_variable} differs between Cargo processes and must not re-run the build"
        );
    }
    assert!(identity_environment_variables(NativeTargetPlatform::MacOs).is_empty());
}

fn cuda_build_target_for_tests() -> native_identity::linux_gpu_backend::CudaBuildTarget {
    native_identity::linux_gpu_backend::CudaBuildTarget {
        cuda_compiler_path: std::path::PathBuf::from("/toolkit/bin/nvcc"),
        cuda_architectures: "86".to_owned(),
    }
}

// --- positive ---

#[test]
fn should_configure_the_cuda_recipe_with_the_detected_compiler_and_architectures() {
    let configure_arguments =
        super::linux_cuda_compiler_configure_arguments(&cuda_build_target_for_tests());
    assert_eq!(
        configure_arguments,
        [
            "-DCMAKE_C_COMPILER=cc",
            "-DCMAKE_CXX_COMPILER=c++",
            "-DASTRONOMICAL_MLX_BUILD_CUDA=ON",
            "-DCMAKE_CUDA_COMPILER=/toolkit/bin/nvcc",
            "-DCMAKE_CUDA_HOST_COMPILER=c++",
            "-DMLX_CUDA_ARCHITECTURES=86",
        ]
    );
}

#[test]
fn should_publish_the_cuda_link_list_and_run_time_compiler_headers_for_the_cuda_recipe() {
    let cuda_products = platform_native_products(NativeTargetPlatform::LinuxCuda);
    let entry_paths: Vec<&str> = cuda_products
        .iter()
        .map(|platform_product| platform_product.entry_relative_path)
        .collect();
    assert!(entry_paths.contains(&MLX_TARGETS_RELATIVE_PATH));
    assert!(entry_paths.contains(&super::CUDA_LINK_LIBRARIES_RELATIVE_PATH));
    assert!(entry_paths.contains(&super::CUDA_TOOLKIT_INCLUDE_DIRECTORY_RELATIVE_PATH));
    for header_folder in ["cute", "cutlass", "cccl"] {
        let header_product = cuda_products
            .iter()
            .find(|platform_product| {
                platform_product
                    .entry_relative_path
                    .ends_with(header_folder)
            })
            .expect(header_folder);
        assert_eq!(header_product.kind, super::NativeProductKind::Directory);
        assert!(
            header_product
                .entry_relative_path
                .starts_with(super::CUDA_JIT_HEADERS_RELATIVE_DIRECTORY)
        );
    }
}

// --- edge / boundary ---

#[test]
fn should_rerun_the_cuda_recipe_on_the_blas_and_cuda_search_settings() {
    let cuda_variables = identity_environment_variables(NativeTargetPlatform::LinuxCuda);
    for identity_input in native_identity::probes::BLAS_SEARCH_VARIABLES
        .iter()
        .chain(native_identity::probes::CUDA_SEARCH_VARIABLES.iter())
    {
        assert!(cuda_variables.contains(identity_input), "{identity_input}");
    }
    assert_eq!(
        cuda_variables.len(),
        native_identity::probes::BLAS_SEARCH_VARIABLES.len()
            + native_identity::probes::CUDA_SEARCH_VARIABLES.len()
    );
    assert!(installs_mlx_package(NativeTargetPlatform::LinuxCuda));
}

#[test]
fn should_keep_the_cpu_recipe_products_unchanged() {
    assert_eq!(
        platform_native_products(NativeTargetPlatform::Linux).len(),
        1
    );
}

// --- negative ---

#[test]
fn should_never_promote_a_macos_target_to_cuda() {
    let macos_platform = native_build_platform_for_host(
        "aarch64-apple-darwin",
        "aarch64-apple-darwin",
        Some(NativeTargetPlatform::MacOs),
    )
    .unwrap();
    assert_eq!(
        macos_platform.with_gpu_backend(true),
        NativeTargetPlatform::MacOs
    );
}
