use std::path::Path;

use super::{NativeLinkDirectives, linux_link_directives, macos_link_directive_lines};

const NATIVE_LIBRARY_DIRECTORY: &str = "/store/entries/identity/lib";

/// The `INTERFACE_LINK_LIBRARIES` MLX v0.32.4 records on a Linux host with
/// OpenBLAS outside the system library directories, exactly as CMake writes
/// it, internal markers included.
const OPENBLAS_LINK_LIBRARY_LIST: &str = "$<LINK_ONLY:dl>
$<LINK_ONLY:pthread>
/opt/openblas/lib/libopenblas.so
$<LINK_ONLY:-lm>
$<LINK_ONLY:-ldl>
/opt/openblas/lib/libopenblas.so
::@(0x5566ba818860)
$<LINK_ONLY:$<BUILD_INTERFACE:mlx_version>>
::@
$<LINK_ONLY:$<BUILD_INTERFACE:fmt::fmt-header-only>>
";

fn linux_directives(mlx_link_library_list: &str) -> NativeLinkDirectives {
    linux_link_directives(Path::new(NATIVE_LIBRARY_DIRECTORY), mlx_link_library_list)
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
    let link_directives = linux_directives(OPENBLAS_LINK_LIBRARY_LIST);
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
    let link_directives = linux_directives("/usr/lib/libopenblas.so\n");
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
    let link_directives = linux_directives("/opt/blas/lib/libopenblas.so.0\n");
    assert!(
        link_directives
            .cargo_directive_lines
            .contains(&"cargo:rustc-link-lib=dylib:+verbatim=libopenblas.so.0".to_owned()),
        "a versioned file without an unversioned link must still link"
    );
}

#[test]
fn should_recognize_a_shared_library_whose_name_contains_so() {
    let link_directives = linux_directives("/opt/solver/lib/libfoo.solver.so.3.27\n");
    assert!(
        link_directives
            .cargo_directive_lines
            .contains(&"cargo:rustc-link-lib=dylib:+verbatim=libfoo.solver.so.3.27".to_owned())
    );
    assert!(link_directives.warning_messages.is_empty());
}

#[test]
fn should_link_only_the_archives_and_runtime_when_mlx_records_no_libraries() {
    let link_directives = linux_directives("");
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
    let link_directives = linux_directives("/opt/blas/lib/libopenblas.a\n");
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
        "-l:libopenblas.so.0",
        "MLX::OpenBLAS",
        "/opt/blas/lib/openblas.dll",
        "/opt/blas/lib/lib.so",
        "/opt/solver/lib/libfoo.solver",
    ] {
        let link_directives = linux_directives(unexpressible_entry);
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
