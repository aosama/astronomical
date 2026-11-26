//! No workflow and no composite action calls a shell script file: repository
//! tooling is typed, tested Rust in xtask (issue #2635).

mod contract_support;

/// Repository tooling is xtask (issue #2635): no workflow and no composite
/// action may call a shell script file. The CI scripts that once had to stay
/// shell, because they ran where no Rust toolchain exists, were replaced by
/// the run timeline and by an expression in the `astronomical-cache` action.
#[test]
fn should_call_no_shell_script_from_any_workflow_or_action() {
    let github_directory = contract_support::repository_root().join(".github");
    let mut definition_paths = Vec::new();
    for definition_folder in ["workflows", "actions"] {
        collect_yaml_files(
            &github_directory.join(definition_folder),
            &mut definition_paths,
        );
    }
    assert!(
        definition_paths.len() > 3,
        "the workflow and action definitions were not found"
    );
    for definition_path in definition_paths {
        let definition_text =
            std::fs::read_to_string(&definition_path).expect("the definition should be readable");
        assert!(
            !calls_a_shell_script(&definition_text),
            "{} calls a shell script; put the tooling in xtask",
            definition_path.display()
        );
    }
}

/// A `.sh` file name followed by the end of a word: a space, a line end or a
/// closing quote.
fn calls_a_shell_script(definition_text: &str) -> bool {
    for script_ending in [".sh ", ".sh\n", ".sh'", ".sh\""] {
        if definition_text.contains(script_ending) {
            return true;
        }
    }
    false
}

#[test]
fn should_recognize_a_shell_script_call_and_nothing_else() {
    assert!(calls_a_shell_script("run: scripts/ci/x.sh begin y\n"));
    assert!(calls_a_shell_script("run: tools/report.sh\n"));
    assert!(calls_a_shell_script("trap 'tools/end.sh' EXIT\n"));
    assert!(!calls_a_shell_script(
        "shell: bash\nrun: xtask classify-scope\n"
    ));
    assert!(!calls_a_shell_script("path: ~/.ssh/known_hosts\n"));
}

fn collect_yaml_files(folder: &std::path::Path, found_paths: &mut Vec<std::path::PathBuf>) {
    let Ok(folder_entries) = std::fs::read_dir(folder) else {
        return;
    };
    for folder_entry in folder_entries.flatten() {
        let entry_path = folder_entry.path();
        if entry_path.is_dir() {
            collect_yaml_files(&entry_path, found_paths);
        } else if entry_path.extension().is_some_and(is_yaml_extension) {
            found_paths.push(entry_path);
        }
    }
}

fn is_yaml_extension(extension: &std::ffi::OsStr) -> bool {
    extension == "yml" || extension == "yaml"
}
