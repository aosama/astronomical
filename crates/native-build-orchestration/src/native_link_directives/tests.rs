use std::path::Path;

use super::{
    NativeLinkDirectives, linux_link_directives, macos_link_directive_lines,
    mlx_exported_link_libraries,
};

const NATIVE_LIBRARY_DIRECTORY: &str = "/store/entries/identity/lib";

/// The `mlx` target block of the `MLXTargets.cmake` MLX v0.32.4 installs on a
/// Linux host with OpenBLAS outside the system library directories, exactly as
/// CMake's export writes it, escapes included.
const OPENBLAS_MLX_TARGETS: &str = r#"# Create imported target mlx
add_library(mlx STATIC IMPORTED)

set_target_properties(mlx PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "MLX_STATIC"
  INTERFACE_COMPILE_FEATURES "cxx_std_20"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "\$<LINK_ONLY:dl>;\$<LINK_ONLY:pthread>;/opt/openblas/lib/libopenblas.so;\$<LINK_ONLY:-lm>;\$<LINK_ONLY:-ldl>;/opt/openblas/lib/libopenblas.so;\$<LINK_ONLY:>;\$<LINK_ONLY:>;\$<LINK_ONLY:>"
)
"#;

/// An `MLXTargets.cmake` whose `mlx` target links exactly these entries.
fn mlx_targets_linking(link_library_entries: &[&str]) -> String {
    let escaped_entries = link_library_entries.join(";").replace('$', "\\$");
    format!(
        "add_library(mlx STATIC IMPORTED)\n\nset_target_properties(mlx PROPERTIES\n  INTERFACE_LINK_LIBRARIES \"{escaped_entries}\"\n)\n"
    )
}

fn linux_directives(link_library_entries: &[&str]) -> NativeLinkDirectives {
    linux_link_directives(
        Path::new(NATIVE_LIBRARY_DIRECTORY),
        &mlx_targets_linking(link_library_entries),
    )
}

#[test]
fn should_keep_the_macos_link_line_exactly_as_every_mac_build_links_today() {
    let cargo_directive_lines = macos_link_directive_lines(
        Path::new(NATIVE_LIBRARY_DIRECTORY),
        Path::new("/toolchain/clang/lib/darwin"),
    );
    assert_eq!(
        cargo_directive_lines,
        [
            "cargo:rustc-link-search=native=/store/entries/identity/lib",
            "cargo:rustc-link-lib=static=mlxc",
            "cargo:rustc-link-lib=static=mlx",
            "cargo:rustc-link-lib=dylib=c++",
            "cargo:rustc-link-lib=framework=Metal",
            "cargo:rustc-link-lib=framework=Foundation",
            "cargo:rustc-link-lib=framework=IOKit",
            "cargo:rustc-link-lib=framework=CoreFoundation",
            "cargo:rustc-link-lib=framework=QuartzCore",
            "cargo:rustc-link-lib=framework=Accelerate",
            "cargo:rustc-link-search=native=/toolchain/clang/lib/darwin",
            "cargo:rustc-link-lib=static=clang_rt.osx",
        ]
    );
}

#[test]
fn should_link_the_archives_then_mlx_libraries_then_the_gnu_cxx_runtime_on_linux() {
    let link_directives =
        linux_link_directives(Path::new(NATIVE_LIBRARY_DIRECTORY), OPENBLAS_MLX_TARGETS);
    assert_eq!(
        link_directives.cargo_directive_lines,
        [
            "cargo:rustc-link-search=native=/store/entries/identity/lib",
            "cargo:rustc-link-lib=static=mlxc",
            "cargo:rustc-link-lib=static=mlx",
            "cargo:rustc-link-lib=dylib=dl",
            "cargo:rustc-link-lib=dylib=pthread",
            "cargo:rustc-link-search=native=/opt/openblas/lib",
            "cargo:rustc-link-lib=dylib:+verbatim=libopenblas.so",
            "cargo:rustc-link-arg=-Wl,-rpath,/opt/openblas/lib",
            "cargo:rustc-link-lib=dylib=m",
            "cargo:rustc-link-lib=dylib=stdc++",
        ]
    );
    assert!(
        link_directives.warning_messages.is_empty(),
        "a shared OpenBLAS links without warnings: {:?}",
        link_directives.warning_messages
    );
    assert_eq!(
        link_directives.linked_library_files,
        [Path::new("/opt/openblas/lib/libopenblas.so")],
        "the one recorded file is checked once, however often MLX lists it"
    );
}

#[test]
fn should_skip_the_run_path_for_a_library_in_a_system_directory() {
    let link_directives = linux_directives(&["/usr/lib/libopenblas.so"]);
    assert!(
        link_directives
            .cargo_directive_lines
            .contains(&"cargo:rustc-link-lib=dylib:+verbatim=libopenblas.so".to_owned())
    );
    assert!(
        !link_directives
            .cargo_directive_lines
            .iter()
            .any(|cargo_directive_line| cargo_directive_line.contains("rpath")),
        "the loader already searches /usr/lib"
    );
}

#[test]
fn should_link_a_versioned_shared_library_by_its_exact_file_name() {
    let link_directives = linux_directives(&["/opt/blas/lib/libopenblas.so.0"]);
    assert!(
        link_directives
            .cargo_directive_lines
            .contains(&"cargo:rustc-link-lib=dylib:+verbatim=libopenblas.so.0".to_owned()),
        "a versioned file without an unversioned link must still link"
    );
}

#[test]
fn should_recognize_a_shared_library_whose_name_contains_so() {
    let link_directives = linux_directives(&["/opt/solver/lib/libfoo.solver.so.3.27"]);
    assert!(
        link_directives
            .cargo_directive_lines
            .contains(&"cargo:rustc-link-lib=dylib:+verbatim=libfoo.solver.so.3.27".to_owned())
    );
    assert!(link_directives.warning_messages.is_empty());
}

#[test]
fn should_link_only_the_archives_and_runtime_when_mlx_records_no_libraries() {
    let link_directives = linux_directives(&[]);
    assert_eq!(
        link_directives.cargo_directive_lines,
        [
            "cargo:rustc-link-search=native=/store/entries/identity/lib",
            "cargo:rustc-link-lib=static=mlxc",
            "cargo:rustc-link-lib=static=mlx",
            "cargo:rustc-link-lib=dylib=stdc++",
        ]
    );
}

#[test]
fn should_warn_about_a_static_blas_whose_own_dependencies_are_unrecorded() {
    let link_directives = linux_directives(&["/opt/blas/lib/libopenblas.a"]);
    assert!(
        link_directives
            .cargo_directive_lines
            .contains(&"cargo:rustc-link-lib=static:+verbatim=libopenblas.a".to_owned())
    );
    assert_eq!(link_directives.warning_messages.len(), 1);
    assert!(link_directives.warning_messages[0].contains("static library"));
}

#[test]
fn should_warn_instead_of_linking_entries_a_build_script_cannot_express() {
    for unexpressible_entry in [
        "$<LINK_ONLY:-pthread>",
        "-l:",
        "$<LINK_ONLY:$<$<PLATFORM_ID:Linux>:atomic>>",
        "MLX::OpenBLAS",
        "/opt/blas/lib/openblas.dll",
        "/opt/blas/lib/lib.so",
        "/opt/solver/lib/libfoo.solver",
    ] {
        let link_directives = linux_directives(&[unexpressible_entry]);
        assert_eq!(
            link_directives.cargo_directive_lines.len(),
            4,
            "{unexpressible_entry:?} must add no link directive"
        );
        assert_eq!(
            link_directives.warning_messages.len(),
            1,
            "{unexpressible_entry:?} must be reported"
        );
    }
}

#[test]
fn should_read_the_mlx_targets_link_libraries_with_the_export_escapes_undone() {
    let link_library_entries = mlx_exported_link_libraries(OPENBLAS_MLX_TARGETS).unwrap();

    assert_eq!(link_library_entries.len(), 9);
    assert_eq!(link_library_entries[0], "$<LINK_ONLY:dl>");
    assert_eq!(link_library_entries[2], "/opt/openblas/lib/libopenblas.so");
}

#[test]
fn should_read_no_link_libraries_from_a_package_without_the_property() {
    let package_without_link_libraries = "add_library(mlx STATIC IMPORTED)\n\nset_target_properties(mlx PROPERTIES\n  INTERFACE_COMPILE_FEATURES \"cxx_std_20\"\n)\n";

    assert_eq!(
        mlx_exported_link_libraries(package_without_link_libraries),
        Some(Vec::new())
    );
}

#[test]
fn should_ignore_the_link_libraries_of_any_target_but_mlx() {
    let another_target = "set_target_properties(gguflib PROPERTIES\n  INTERFACE_LINK_LIBRARIES \"/opt/other/libother.so\"\n)\nset_target_properties(mlx PROPERTIES\n  INTERFACE_LINK_LIBRARIES \"dl\"\n)\n";

    assert_eq!(mlx_exported_link_libraries(another_target).unwrap(), ["dl"]);
}

#[test]
fn should_warn_instead_of_linking_a_path_relative_to_the_install_prefix() {
    let link_directives = linux_directives(&["${_IMPORT_PREFIX}/lib/libmlxextra.a"]);

    assert_eq!(link_directives.cargo_directive_lines.len(), 4);
    assert_eq!(link_directives.warning_messages.len(), 1);
}

#[test]
fn should_link_a_named_library_file_exactly() {
    let link_directives = linux_directives(&["-l:libopenblas.so.0", "-l:libblas.a"]);

    assert!(
        link_directives
            .cargo_directive_lines
            .contains(&"cargo:rustc-link-lib=dylib:+verbatim=libopenblas.so.0".to_owned())
    );
    assert!(
        link_directives
            .cargo_directive_lines
            .contains(&"cargo:rustc-link-lib=static:+verbatim=libblas.a".to_owned())
    );
    assert!(link_directives.warning_messages.is_empty());
}

#[test]
fn should_warn_about_a_conditional_entry_instead_of_dropping_it() {
    let link_directives = linux_directives(&["$<LINK_ONLY:$<$<PLATFORM_ID:Linux>:atomic>>"]);

    assert_eq!(link_directives.warning_messages.len(), 1);
    assert!(link_directives.warning_messages[0].contains("conditional"));
}

#[test]
fn should_skip_the_run_path_for_a_debian_multiarch_library_folder() {
    let link_directives = linux_directives(&["/usr/lib/x86_64-linux-gnu/libopenblas.so"]);

    assert!(
        !link_directives
            .cargo_directive_lines
            .iter()
            .any(|cargo_directive_line| cargo_directive_line.contains("rpath"))
    );
}

#[test]
fn should_warn_when_mlx_targets_defines_no_mlx_properties() {
    let namespaced_export =
        "set_target_properties(mlx::mlx PROPERTIES\n  INTERFACE_LINK_LIBRARIES \"dl\"\n)\n";

    assert_eq!(mlx_exported_link_libraries(namespaced_export), None);
    assert_eq!(mlx_exported_link_libraries(""), None);
    let link_directives =
        linux_link_directives(Path::new(NATIVE_LIBRARY_DIRECTORY), namespaced_export);
    assert_eq!(link_directives.warning_messages.len(), 1);
    assert_eq!(
        link_directives
            .cargo_directive_lines
            .last()
            .map(String::as_str),
        Some("cargo:rustc-link-lib=dylib=stdc++")
    );
}
