use crate::staged_package_scope::{ClippyScope, WorkspaceMember, clippy_scope};

fn member(name: &str, directory: &str) -> WorkspaceMember {
    WorkspaceMember {
        name: name.to_owned(),
        directory: directory.to_owned(),
    }
}

fn members() -> Vec<WorkspaceMember> {
    vec![
        member("model-primitives", "crates/engine/model-primitives"),
        member("qwen35", "crates/families/text/qwen35"),
        member(
            "qwen35-recognition",
            "crates/families/text/qwen35/recognition",
        ),
        member("xtask", "xtask"),
    ]
}

fn paths(staged_paths: &[&str]) -> Vec<String> {
    staged_paths.iter().map(|path| (*path).to_owned()).collect()
}

// --- positive ---

#[test]
fn should_lint_only_the_packages_that_own_the_staged_rust_files() {
    let scope = clippy_scope(
        &paths(&[
            "crates/families/text/qwen35/src/lib.rs",
            "crates/engine/model-primitives/src/sorted_expert_matmul/mod.rs",
            "crates/families/text/qwen35/src/kernel_capability/mod.rs",
        ]),
        &members(),
    );
    assert_eq!(
        scope,
        ClippyScope::Packages(vec!["model-primitives".to_owned(), "qwen35".to_owned()])
    );
}

#[test]
fn should_give_a_nested_crate_its_own_files() {
    let scope = clippy_scope(
        &paths(&["crates/families/text/qwen35/recognition/src/lib.rs"]),
        &members(),
    );
    assert_eq!(
        scope,
        ClippyScope::Packages(vec!["qwen35-recognition".to_owned()])
    );
}

#[test]
fn should_ignore_staged_files_that_are_not_rust() {
    let scope = clippy_scope(
        &paths(&[
            "crates/engine/model-primitives/src/sorted_expert_matmul/metal/kernel.metal",
            "crates/engine/model-primitives/src/lib.rs",
            "docs/glossary.md",
        ]),
        &members(),
    );
    assert_eq!(
        scope,
        ClippyScope::Packages(vec!["model-primitives".to_owned()])
    );
}

// --- edge ---

#[test]
fn should_not_take_a_folder_whose_name_only_starts_like_a_package() {
    let scope = clippy_scope(&paths(&["xtask-helpers/src/main.rs"]), &members());
    assert_eq!(scope, ClippyScope::Workspace);
}

#[test]
fn should_lint_the_whole_workspace_when_no_rust_file_is_staged() {
    assert_eq!(clippy_scope(&[], &members()), ClippyScope::Workspace);
    assert_eq!(
        clippy_scope(&paths(&["README.md"]), &members()),
        ClippyScope::Workspace
    );
}

// --- negative ---

#[test]
fn should_lint_the_whole_workspace_when_a_manifest_or_lint_setting_changes() {
    for workspace_wide_path in [
        "crates/engine/model-primitives/Cargo.toml",
        "Cargo.lock",
        "clippy.toml",
        ".cargo/config.toml",
        "rust-toolchain.toml",
    ] {
        let scope = clippy_scope(
            &paths(&[
                "crates/engine/model-primitives/src/lib.rs",
                workspace_wide_path,
            ]),
            &members(),
        );
        assert_eq!(scope, ClippyScope::Workspace, "{workspace_wide_path}");
    }
}

#[test]
fn should_lint_the_whole_workspace_for_a_rust_file_outside_every_package() {
    let scope = clippy_scope(&paths(&["scripts/release/tool.rs"]), &members());
    assert_eq!(scope, ClippyScope::Workspace);
}
