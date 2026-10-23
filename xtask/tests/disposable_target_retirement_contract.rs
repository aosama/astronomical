//! The retirement contract for `cargo xtask disposable-target`: pins the
//! workspace profile discipline, the string-only cargo alias form, the GPU
//! serial and optimized-build rules baked into the acceptance aliases, and
//! the permanent absence of the retired shell runner and its two shell
//! contracts. Absorbs the ownership block of the retired
//! `test-cargo-artifact-lifecycle-contract.sh`.

use std::path::Path;

fn file_text(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()))
}

#[test]
fn should_pin_profiles_alias_rules_and_the_retired_runner_absence() {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let workspace_manifest_text = file_text(&repository_root.join("Cargo.toml"));
    for profile_name in ["dev", "test"] {
        assert!(
            workspace_manifest_text
                .contains(&format!("[profile.{profile_name}]")),
            "the workspace manifest should keep its {profile_name} profile"
        );
    }
    assert!(
        workspace_manifest_text.contains("debug = \"line-tables-only\""),
        "the workspace profiles should keep bounded diagnostics"
    );
    assert!(
        workspace_manifest_text.contains("debug = true"),
        "the full-debug profile should keep complete debug information"
    );
    assert!(
        workspace_manifest_text.contains("split-debuginfo = \"packed\""),
        "the full-debug profile should keep packed split debug information"
    );

    let cargo_configuration_text = file_text(&repository_root.join(".cargo/config.toml"));
    // The alias section runs until the next header line; plain bracket
    // splitting would cut at the `[--update-baseline]` inside the section's
    // own usage comments.
    let alias_section_text: String = cargo_configuration_text
        .lines()
        .skip_while(|configuration_line| configuration_line.trim() != "[alias]")
        .skip(1)
        .take_while(|configuration_line| !configuration_line.starts_with('['))
        .collect::<Vec<&str>>()
        .join("\n");
    assert!(
        !alias_section_text.is_empty(),
        "the cargo configuration should have an alias section"
    );
    assert!(
        !alias_section_text.contains("= ["),
        "every cargo alias must stay a single string, never an array"
    );
    assert!(
        alias_section_text.contains("xtask ="),
        "the xtask alias should stay documented in the cargo configuration"
    );
    for acceptance_alias in [
        "test-direct-mlx",
        "test-mlx-memory",
        "test-acceptance",
        "test-quantization-direct-mlx",
        "test-quantization-acceptance",
        "test-jangtq2-decode-throughput",
        "test-jangtq2-serving-throughput",
    ] {
        assert!(
            alias_section_text.contains(&format!("{acceptance_alias} =")),
            "the {acceptance_alias} alias should stay pinned"
        );
    }
    for throughput_alias in ["test-jangtq2-decode-throughput", "test-jangtq2-serving-throughput"] {
        let alias_command_text = alias_section_text
            .lines()
            .find(|alias_line| alias_line.starts_with(&format!("{throughput_alias} =")))
            .unwrap_or_else(|| panic!("the {throughput_alias} alias should exist"));
        assert!(
            alias_command_text.contains("--release"),
            "the {throughput_alias} alias must carry the optimized build"
        );
        assert!(
            alias_command_text.contains("--test-threads=1"),
            "the {throughput_alias} alias must carry the GPU serial rule"
        );
    }

    // The retirement itself is pinned: the shell runner and its two shell
    // contracts must never come back.
    for retired_relative_path in [
        "scripts/run-in-disposable-cargo-target.sh",
        "verification/tooling_contracts/test-cargo-artifact-lifecycle-contract.sh",
        "verification/tooling_contracts/test-cargo-artifact-cleanup-signal-contract.sh",
    ] {
        assert!(
            !repository_root.join(retired_relative_path).exists(),
            "{retired_relative_path} was retired in favor of cargo xtask disposable-target"
        );
    }
}
