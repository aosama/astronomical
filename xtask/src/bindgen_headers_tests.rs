use std::path::{Path, PathBuf};
use std::process::Command;

use astronomical_native_identity::{NativeBuildProfile, digest_bytes};

use super::command::run;
use super::hash_manifest::{manifest_differences, tree_hash_manifest};
use super::header_locations::{
    built_entry_directory, built_identity, native_build_store_directory,
    native_build_store_schema_version, native_dependency_cache_directory,
};
use super::options::{ProvisionMode, parse_provision_options};
use super::patch_pipeline::{
    PatchStep, parse_cmake_patch_pipeline, parse_patch_list, render_patch_list,
};
use super::pinned_archives::select_pinned_header_archives;
use super::provision_flow::completion_marker_text;

const FIXTURE_PATCH_MARKER: &str = "astronomical fixture patch applied";

fn arguments(argument_words: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_word in argument_words {
        owned_arguments.push((*argument_word).to_owned());
    }
    owned_arguments
}

fn repository_root() -> PathBuf {
    crate::workspace_root_from_manifest_directory()
}

fn manifest_line(
    archive_file_name: &str,
    archive_sha256: &str,
    dependency_description: &str,
) -> String {
    format!(
        "{archive_file_name}|https://example.invalid/{archive_file_name}|{archive_sha256}|{dependency_description}\n"
    )
}

#[test]
fn should_default_to_provisioning_the_core_profile() {
    let provision_options = parse_provision_options(&arguments(&[])).expect("no flags is valid");

    assert_eq!(provision_options.mode, ProvisionMode::Provision);
    assert_eq!(
        provision_options.native_build_profile,
        NativeBuildProfile::Core
    );
}

#[test]
fn should_parse_every_flag_the_shell_entry_point_accepted() {
    let provision_options = parse_provision_options(&arguments(&[
        "--cache-dir",
        "/cache",
        "--native-build-store-dir",
        "/store",
        "--manifest",
        "/m",
        "--patch-list",
        "/p",
        "--profile",
        "core+memory-contract",
        "--target",
        "aarch64-apple-darwin",
        "--verify-headers",
    ]))
    .expect("every flag is valid");

    assert_eq!(provision_options.cache_directory.as_deref(), Some("/cache"));
    assert_eq!(
        provision_options.native_build_store_directory.as_deref(),
        Some("/store")
    );
    assert_eq!(
        provision_options.explicit_manifest_path.as_deref(),
        Some("/m")
    );
    assert_eq!(
        provision_options.explicit_patch_list_path.as_deref(),
        Some("/p")
    );
    assert_eq!(
        provision_options.native_build_profile,
        NativeBuildProfile::CoreMemoryContract
    );
    assert_eq!(
        provision_options.target_triple.as_deref(),
        Some("aarch64-apple-darwin")
    );
    assert_eq!(provision_options.mode, ProvisionMode::VerifyHeaders);
}

#[test]
fn should_refuse_unknown_dangling_or_conflicting_flags() {
    assert!(parse_provision_options(&arguments(&["--bogus"])).is_err());
    assert!(parse_provision_options(&arguments(&["--cache-dir"])).is_err());
    assert!(parse_provision_options(&arguments(&["--target", ""])).is_err());
    assert!(parse_provision_options(&arguments(&["--profile", "full"])).is_err());
    assert!(
        parse_provision_options(&arguments(&["--verify-headers", "--print-patch-list"])).is_err()
    );
}

#[test]
fn should_select_mlx_and_mlx_c_even_though_one_name_prefixes_the_other() {
    let manifest_text = format!(
        "{}{}{}",
        manifest_line("mlx-c.tar.gz", &"b".repeat(64), "MLX C 0.7.0"),
        manifest_line("fmt.tar.gz", &"c".repeat(64), "fmt 12.1.0"),
        manifest_line("mlx.tar.gz", &"a".repeat(64), "MLX 0.32.4"),
    );

    let pinned_archives =
        select_pinned_header_archives(&manifest_text).expect("both archives are declared");

    assert_eq!(pinned_archives.mlx.archive_file_name, "mlx.tar.gz");
    assert_eq!(pinned_archives.mlx_c.archive_file_name, "mlx-c.tar.gz");
}

#[test]
fn should_refuse_a_manifest_missing_duplicating_or_corrupting_an_archive() {
    let mlx_only = manifest_line("mlx.tar.gz", &"a".repeat(64), "MLX 0.32.4");
    let duplicated_mlx = format!(
        "{mlx_only}{mlx_only}{}",
        manifest_line("c.tar.gz", &"b".repeat(64), "MLX C 0.7.0")
    );
    let uppercase_digest = manifest_line("mlx.tar.gz", &"A".repeat(64), "MLX 0.32.4");

    assert!(select_pinned_header_archives(&mlx_only).is_err());
    assert!(select_pinned_header_archives(&duplicated_mlx).is_err());
    assert!(select_pinned_header_archives(&uppercase_digest).is_err());
    assert!(select_pinned_header_archives("").is_err());
}

/// Pins the parser against the real native CMakeLists: if a restructure
/// stops it finding the patches, this fails instead of provisioning
/// unpatched headers.
#[test]
fn should_parse_the_real_native_cmake_patch_pipeline() {
    let cmake_lists_text = std::fs::read_to_string(
        repository_root().join("crates/runtime-integration/native/CMakeLists.txt"),
    )
    .expect("the native CMakeLists should be readable");

    let patch_steps = parse_cmake_patch_pipeline(&cmake_lists_text);

    let mut mlx_patch_count = 0;
    let mut mlx_c_patch_count = 0;
    for patch_step in &patch_steps {
        assert!(
            patch_step.patch_path.starts_with("third-party/patches/"),
            "{patch_step:?}"
        );
        assert!(
            repository_root().join(&patch_step.patch_path).is_file(),
            "{patch_step:?}"
        );
        match patch_step.dependency_name.as_str() {
            "mlx" => mlx_patch_count += 1,
            "mlx_c" => mlx_c_patch_count += 1,
            other_dependency => panic!("unexpected dependency {other_dependency}"),
        }
    }
    assert!(
        mlx_patch_count >= 1 && mlx_c_patch_count >= 1,
        "{patch_steps:?}"
    );
}

#[test]
fn should_ignore_patch_files_of_other_fetchcontent_targets() {
    let cmake_lists_text = "FetchContent_Declare(\n    fmt\n    PATCH_COMMAND \"-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../third-party/patches/fmt.patch\"\n)\n\
FetchContent_Declare(\n    mlx\n    \"-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../third-party/patches/mlx.patch\"\n)\n";

    assert_eq!(
        parse_cmake_patch_pipeline(cmake_lists_text),
        vec![PatchStep {
            dependency_name: "mlx".to_owned(),
            patch_path: "third-party/patches/mlx.patch".to_owned()
        }]
    );
    assert!(parse_cmake_patch_pipeline("").is_empty());
}

#[test]
fn should_round_trip_an_explicit_patch_list_and_refuse_a_malformed_line() {
    let patch_list_text = "mlx\t/fixture/mlx.patch\nmlx_c\t/fixture/mlx_c.patch\n";

    let patch_steps = parse_patch_list(patch_list_text).expect("well-formed list");

    assert_eq!(render_patch_list(&patch_steps), patch_list_text);
    assert!(parse_patch_list("mlx /no-tab.patch\n").is_err());
}

#[test]
fn should_report_modified_built_only_and_extracted_only_paths() {
    let extracted_manifest = "aaaa  ./c/mlx.h\nbbbb  ./c/only-extracted.h\ncccc  ./c/same.h\n";
    let built_manifest = "ffff  ./c/mlx.h\ndddd  ./c/only-built.h\ncccc  ./c/same.h\n";

    let differing_paths = manifest_differences(extracted_manifest, built_manifest);

    assert_eq!(
        differing_paths,
        vec![
            "modified ./c/mlx.h",
            "built-only ./c/only-built.h",
            "extracted-only ./c/only-extracted.h"
        ]
    );
    assert!(manifest_differences(built_manifest, built_manifest).is_empty());
}

/// Byte-for-byte parity with the shell's `find | sort | shasum` manifest, so
/// an extraction the shell published still verifies as cached.
#[cfg(target_os = "macos")]
#[test]
fn should_hash_a_tree_exactly_like_find_sort_and_shasum() {
    let tree_directory = tempfile::tempdir().expect("a temporary directory");
    write_fixture_file(&tree_directory.path().join("b/z.h"), "zed\n");
    write_fixture_file(&tree_directory.path().join("a.h"), "a\n");
    write_fixture_file(&tree_directory.path().join("a/b.h"), "ab\n");
    write_fixture_file(&tree_directory.path().join(".hidden"), "");

    let shell_manifest = Command::new("sh")
        .args([
            "-c",
            "find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256",
        ])
        .current_dir(tree_directory.path())
        .output()
        .expect("the shell pipeline should run");

    assert_eq!(
        tree_hash_manifest(tree_directory.path()).expect("the tree should hash"),
        String::from_utf8_lossy(&shell_manifest.stdout)
    );
}

#[test]
fn should_write_the_completion_marker_in_the_shell_layout() {
    let pinned_archives = select_pinned_header_archives(&format!(
        "{}{}",
        manifest_line("mlx.tar.gz", &"a".repeat(64), "MLX 0.32.4"),
        manifest_line("mlx-c.tar.gz", &"b".repeat(64), "MLX C 0.7.0")
    ))
    .expect("both archives are declared");

    let marker_text = completion_marker_text("feed", &pinned_archives, 12, 3);

    assert_eq!(
        marker_text,
        format!(
            "source_identity=feed\nmlx_archive=mlx.tar.gz\nmlx_archive_sha256={}\nmlx_c_archive=mlx-c.tar.gz\nmlx_c_archive_sha256={}\nmlx_file_count=12\nmlx_c_file_count=3\n",
            "a".repeat(64),
            "b".repeat(64)
        )
    );
}

fn write_fixture_file(file_path: &Path, file_text: &str) {
    std::fs::create_dir_all(file_path.parent().expect("a parent directory"))
        .expect("fixture directory");
    std::fs::write(file_path, file_text).expect("fixture file");
}

/// Fixture archives, patches, manifest, and patch list in a sandbox, so the
/// provisioning journeys never touch the real pins, cache, or network.
#[cfg(target_os = "macos")]
struct ProvisionSandbox {
    sandbox_directory: tempfile::TempDir,
}

#[cfg(target_os = "macos")]
impl ProvisionSandbox {
    fn install() -> Self {
        let sandbox = Self {
            sandbox_directory: tempfile::tempdir().expect("a sandbox directory"),
        };
        let mlx_sha256 =
            sandbox.write_fixture_archive("mlx-fixture.tar.gz", "mlx-fixture-1.0", "mlx/array.h");
        let mlx_c_sha256 = sandbox.write_fixture_archive(
            "mlx-c-fixture.tar.gz",
            "mlx-c-fixture-1.0",
            "mlx/c/mlx.h",
        );
        sandbox.write_patch("mlx.patch", "mlx/array.h");
        sandbox.write_patch("mlx_c.patch", "mlx/c/mlx.h");
        let manifest_text = format!(
            "{}{}",
            manifest_line("mlx-fixture.tar.gz", &mlx_sha256, "MLX fixture headers"),
            manifest_line(
                "mlx-c-fixture.tar.gz",
                &mlx_c_sha256,
                "MLX C fixture headers"
            ),
        );
        std::fs::write(sandbox.path("manifest"), manifest_text).expect("fixture manifest");
        let patch_list_text = format!(
            "mlx\t{}\nmlx_c\t{}\n",
            sandbox.path("patches/mlx.patch").display(),
            sandbox.path("patches/mlx_c.patch").display()
        );
        std::fs::write(sandbox.path("patch-list"), patch_list_text).expect("fixture patch list");
        sandbox
    }

    fn path(&self, relative_path: &str) -> PathBuf {
        self.sandbox_directory.path().join(relative_path)
    }

    fn write_fixture_archive(
        &self,
        archive_file_name: &str,
        top_level_name: &str,
        header_path: &str,
    ) -> String {
        write_fixture_file(
            &self.path(&format!("archive-sources/{top_level_name}/{header_path}")),
            "original\n",
        );
        std::fs::create_dir_all(self.path("cache")).expect("fixture cache");
        let tar_status = Command::new("tar")
            .arg("-czf")
            .arg(self.path(&format!("cache/{archive_file_name}")))
            .arg("-C")
            .arg(self.path("archive-sources"))
            .arg(top_level_name)
            .status()
            .expect("tar should run");
        assert!(tar_status.success());
        digest_bytes(
            &std::fs::read(self.path(&format!("cache/{archive_file_name}")))
                .expect("fixture archive"),
        )
    }

    fn write_patch(&self, patch_file_name: &str, header_path: &str) {
        let patch_text = format!(
            "--- a/{header_path}\n+++ b/{header_path}\n@@ -1 +1,2 @@\n original\n+{FIXTURE_PATCH_MARKER}\n"
        );
        write_fixture_file(
            &self.path(&format!("patches/{patch_file_name}")),
            &patch_text,
        );
    }

    fn provision_arguments(&self, extra_flags: &[&str]) -> Vec<String> {
        let mut provision_arguments = Vec::new();
        for (flag, sandbox_path) in [
            ("--cache-dir", "cache"),
            ("--native-build-store-dir", "store"),
            ("--manifest", "manifest"),
            ("--patch-list", "patch-list"),
        ] {
            provision_arguments.push(flag.to_owned());
            provision_arguments.push(self.path(sandbox_path).display().to_string());
        }
        provision_arguments.extend(arguments(extra_flags));
        provision_arguments
    }

    fn extraction_directory(&self) -> PathBuf {
        let source_identity =
            super::header_locations::source_identity(&repository_root()).expect("source identity");
        self.path("cache/bindgen-headers").join(source_identity)
    }
}

#[cfg(target_os = "macos")]
#[test]
fn should_provision_patched_headers_then_reuse_and_self_heal_them() {
    let sandbox = ProvisionSandbox::install();

    run(&repository_root(), &sandbox.provision_arguments(&[]))
        .expect("the fresh provision should succeed");
    let extraction_directory = sandbox.extraction_directory();
    let patched_header = extraction_directory.join("mlx_c-src/mlx/c/mlx.h");
    assert!(
        std::fs::read_to_string(&patched_header)
            .expect("patched header")
            .contains(FIXTURE_PATCH_MARKER)
    );
    assert!(
        std::fs::read_to_string(extraction_directory.join("mlx-src/mlx/array.h"))
            .expect("header")
            .contains(FIXTURE_PATCH_MARKER)
    );
    let marker_text =
        std::fs::read_to_string(extraction_directory.join("complete")).expect("completion marker");
    assert!(marker_text.contains("mlx_archive=mlx-fixture.tar.gz"));

    run(&repository_root(), &sandbox.provision_arguments(&[]))
        .expect("the cached run should succeed");
    std::fs::write(&patched_header, "damaged\n").expect("damage the extraction");
    run(&repository_root(), &sandbox.provision_arguments(&[]))
        .expect("a damaged extraction should self-heal");

    assert!(
        std::fs::read_to_string(&patched_header)
            .expect("healed header")
            .contains(FIXTURE_PATCH_MARKER)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn should_refuse_a_tampered_or_missing_archive_without_downloading() {
    let sandbox = ProvisionSandbox::install();
    std::fs::write(sandbox.path("cache/mlx-fixture.tar.gz"), "tampered")
        .expect("tamper with the archive");

    let tampered_error = run(&repository_root(), &sandbox.provision_arguments(&[]))
        .expect_err("a tampered archive must be refused");
    assert!(
        tampered_error.contains("fails its pinned SHA-256"),
        "{tampered_error}"
    );
    assert!(
        tampered_error.contains("cargo xtask bootstrap-native-dependencies"),
        "{tampered_error}"
    );

    std::fs::remove_file(sandbox.path("cache/mlx-fixture.tar.gz")).expect("remove the archive");
    let missing_error = run(&repository_root(), &sandbox.provision_arguments(&[]))
        .expect_err("a missing archive must be refused");
    assert!(
        missing_error.contains("cargo xtask bootstrap-native-dependencies"),
        "{missing_error}"
    );
    assert!(
        !sandbox.extraction_directory().exists(),
        "no partial extraction may be published"
    );
}

/// Builds a fixture native build store entry under the identity the real
/// toolchain resolves, holding the given MLX-C header tree.
#[cfg(target_os = "macos")]
fn install_fixture_store_entry(sandbox: &ProvisionSandbox, built_header_text: &str) {
    let host_target = String::from_utf8(
        Command::new("rustc")
            .args(["--print", "host-tuple"])
            .output()
            .expect("rustc should run")
            .stdout,
    )
    .expect("utf-8 host target");
    let built_identity = built_identity(
        &repository_root(),
        NativeBuildProfile::Core,
        host_target.trim(),
    )
    .expect("built identity");
    let schema_version =
        native_build_store_schema_version(&repository_root()).expect("schema version");
    let built_entry =
        built_entry_directory(&sandbox.path("store"), schema_version, &built_identity);
    write_fixture_file(&built_entry.join("complete"), "");
    write_fixture_file(&built_entry.join("include/mlx/c/mlx.h"), built_header_text);
}

#[cfg(target_os = "macos")]
#[test]
fn should_verify_headers_against_a_completed_build_and_catch_a_one_byte_difference() {
    let sandbox = ProvisionSandbox::install();
    let missing_build_error = run(
        &repository_root(),
        &sandbox.provision_arguments(&["--verify-headers"]),
    )
    .expect_err("verification without a completed build must fail");
    assert!(
        missing_build_error.contains("no completed native build"),
        "{missing_build_error}"
    );

    run(&repository_root(), &sandbox.provision_arguments(&[])).expect("provision first");
    install_fixture_store_entry(&sandbox, &format!("original\n{FIXTURE_PATCH_MARKER}\n"));
    run(
        &repository_root(),
        &sandbox.provision_arguments(&["--verify-headers"]),
    )
    .expect("identical headers should verify");

    install_fixture_store_entry(&sandbox, &format!("original\n{FIXTURE_PATCH_MARKER}!\n"));
    let mismatch_error = run(
        &repository_root(),
        &sandbox.provision_arguments(&["--verify-headers"]),
    )
    .expect_err("a one-byte difference must fail verification");
    assert!(mismatch_error.contains("differ"), "{mismatch_error}");
}

#[test]
fn should_refuse_relative_cache_and_store_directories() {
    let relative_cache_error = native_dependency_cache_directory(Some("relative/cache"))
        .expect_err("a relative cache must be refused");
    let relative_store_error = native_build_store_directory(Some("relative/store"))
        .expect_err("a relative store must be refused");

    assert!(
        relative_cache_error.contains("absolute"),
        "{relative_cache_error}"
    );
    assert!(
        relative_store_error.contains("absolute"),
        "{relative_store_error}"
    );
    assert_eq!(
        native_build_store_directory(Some("/store")).expect("an absolute store is accepted"),
        PathBuf::from("/store")
    );
}
