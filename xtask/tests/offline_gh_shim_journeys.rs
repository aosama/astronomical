//! The prune and sccache-save journeys with a shimmed `gh`: the policy's
//! process boundary is the subject, so these are integration journeys, not
//! unit tests. The shim never touches real GitHub state, mirroring the shell
//! contract tests this file replaces (issue #1250).

use astronomical_test_harness::offline_test;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

struct GhShim {
    shim_directory: PathBuf,
    script_path: PathBuf,
}

impl GhShim {
    /// Writes a `gh` shim that replays canned responses: a plain `api` call
    /// prints the default branch, a `--paginate` cache list prints the TSV
    /// rows, and `cache delete` exits per the map (default 0).
    fn install(
        root_directory: &std::path::Path,
        cache_list_tsv: &str,
        delete_exit_codes: &HashMap<u64, i32>,
    ) -> Self {
        let shim_directory = root_directory.join("shim-bin");
        std::fs::create_dir_all(&shim_directory).unwrap();
        let delete_lines: Vec<String> = delete_exit_codes
            .iter()
            .map(|(cache_id, exit_code)| format!("        {cache_id}) exit {exit_code} ;;"))
            .collect();
        // The shim's PATH is the shim directory alone (that is the
        // isolation guarantee), so every command here must be a shell
        // builtin: `printf`, not `cat` or external tools.
        let tsv_emit_lines: Vec<String> = cache_list_tsv
            .lines()
            .map(|tsv_line| format!("    printf '%s\\n' '{tsv_line}'"))
            .collect();
        let shim_script = format!(
            "#!/bin/sh\nif [ \"$1\" = \"api\" ] && [ \"$2\" = \"--paginate\" ]; then\n{}\n    exit 0\nfi\nif [ \"$1\" = \"api\" ]; then\n    printf 'main\\n'\n    exit 0\nfi\nif [ \"$1\" = \"cache\" ] && [ \"$2\" = \"delete\" ]; then\n    case \"$3\" in\n{}\n        *) exit 0 ;;\n    esac\nfi\nexit 1\n",
            tsv_emit_lines.join("\n"),
            delete_lines.join("\n")
        );
        let script_path = shim_directory.join("gh");
        std::fs::write(&script_path, shim_script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        Self {
            shim_directory,
            script_path,
        }
    }

    fn command_for_xtask(&self, subcommand_arguments: &[&str]) -> Command {
        // PATH is REPLACED, not extended: the shim directory is the only
        // lookup root, so the real authenticated gh can never run, and the
        // denied-network harness below fails any attempt loudly. The xtask
        // binary itself is addressed by cargo's absolute CARGO_BIN_EXE path.
        let mut xtask_command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        xtask_command.args(subcommand_arguments);
        xtask_command.env("PATH", &self.shim_directory);
        xtask_command
    }
}

impl Drop for GhShim {
    fn drop(&mut self) {
        if let Some(sandbox_directory) = self.script_path.parent().and_then(std::path::Path::parent)
        {
            let _ = std::fs::remove_dir_all(sandbox_directory);
        }
    }
}

const THREE_FAMILY_CACHE_LIST: &str = "1\tastronomical-v2-native-build-macOS-a\trefs/heads/main\t2026-10-01T00:00:00Z\t100\n\
2\tastronomical-v2-native-build-macOS-b\trefs/heads/main\t2026-10-02T00:00:00Z\t100\n\
3\tastronomical-v2-native-build-macOS-c\trefs/heads/main\t2026-10-03T00:00:00Z\t100\n\
4\tastronomical-v2-sccache-macOS-oversize\trefs/heads/main\t2026-10-09T00:00:00Z\t500000000\n\
5\tastronomical-v2-sccache-macOS-good\trefs/heads/main\t2026-10-01T00:00:00Z\t100000000\n";

fn output_path(root_directory: &std::path::Path) -> PathBuf {
    root_directory.join("github-output")
}

fn read_output_map(output_path: &std::path::Path) -> HashMap<String, String> {
    std::fs::read_to_string(output_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            line.split_once('=')
                .map(|(name, value)| (name.to_owned(), value.to_owned()))
        })
        .collect()
}

offline_test!(
    prune_deletes_only_surplus_and_oversize_entries_through_gh,
    {
        let sandbox_directory =
            std::env::temp_dir().join(format!("astronomical-prune-journey-{}", std::process::id()));
        std::fs::create_dir_all(&sandbox_directory).unwrap();
        let shim = GhShim::install(&sandbox_directory, THREE_FAMILY_CACHE_LIST, &HashMap::new());

        let mut prune_command =
            shim.command_for_xtask(&["prune-caches", "--repo", "aosama/astronomical"]);
        let prune_run = prune_command.output().expect("the xtask binary should run");
        assert!(prune_run.status.success());

        let prune_log = String::from_utf8_lossy(&prune_run.stderr);
        // Oversize sccache is deleted with its own reason, and the native-build
        // family keeps its two newest; the within-budget sccache entry survives.
        assert!(prune_log.contains("action=deleted reason=oversize cache_id=4"));
        assert!(prune_log.contains("action=deleted reason=surplus cache_id=1"));
        assert!(
            !prune_log.contains("cache_id=5"),
            "the within-budget sccache entry must stay"
        );
        assert!(
            !prune_log.contains("cache_id=2"),
            "second-newest native-build entry must stay"
        );
        assert!(
            !prune_log.contains("cache_id=3"),
            "newest native-build entry must stay"
        );
        assert!(prune_log.contains("outcome=pruned deleted=2"));
    }
);

offline_test!(prune_dry_run_reports_deletions_without_calling_delete, {
    let sandbox_directory =
        std::env::temp_dir().join(format!("astronomical-prune-dry-{}", std::process::id()));
    std::fs::create_dir_all(&sandbox_directory).unwrap();
    // Any delete invocation would exit non-zero under this shim, so a
    // dry-run must never reach it.
    let mut blocked_deletions = HashMap::new();
    blocked_deletions.insert(1u64, 1i32);
    let shim = GhShim::install(
        &sandbox_directory,
        THREE_FAMILY_CACHE_LIST,
        &blocked_deletions,
    );

    let mut prune_command =
        shim.command_for_xtask(&["prune-caches", "--dry-run", "--repo", "aosama/astronomical"]);
    let prune_run = prune_command.output().expect("the xtask binary should run");
    assert!(prune_run.status.success());
    let prune_log = String::from_utf8_lossy(&prune_run.stderr);
    assert!(prune_log.contains("action=would-delete"));
    assert!(prune_log.contains("outcome=dry-run deleted=2"));
});

offline_test!(sccache_save_replaces_a_stored_oversize_entry_and_saves, {
    let sandbox_directory = std::env::temp_dir().join(format!(
        "astronomical-sccache-save-journey-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&sandbox_directory).unwrap();
    let shim = GhShim::install(&sandbox_directory, THREE_FAMILY_CACHE_LIST, &HashMap::new());
    let sccache_directory = sandbox_directory.join("sccache-dir");
    std::fs::create_dir_all(sccache_directory.join("nested")).unwrap();
    std::fs::write(sccache_directory.join("top"), vec![0u8; 1000]).unwrap();
    std::fs::write(sccache_directory.join("nested/leaf"), vec![0u8; 500]).unwrap();
    let github_output_path = output_path(&sandbox_directory);
    std::fs::write(&github_output_path, "").unwrap();

    let mut save_command =
        shim.command_for_xtask(&["save-sccache", "astronomical-v2-sccache-macOS-oversize"]);
    save_command
        .env("GITHUB_REPOSITORY", "aosama/astronomical")
        .env("SCCACHE_DIR", &sccache_directory)
        .env("GITHUB_OUTPUT", &github_output_path);
    let save_run = save_command.output().expect("the xtask binary should run");
    assert!(save_run.status.success());

    let decisions = read_output_map(&github_output_path);
    assert_eq!(
        decisions.get("should_save").map(String::as_str),
        Some("true")
    );
    assert_eq!(
        decisions.get("decision").map(String::as_str),
        Some("replace-oversize-entry")
    );
    assert_eq!(
        decisions.get("directory_bytes").map(String::as_str),
        Some("1500")
    );
    let save_log = String::from_utf8_lossy(&save_run.stderr);
    assert!(save_log.contains("action=deleted-oversize cache_id=4"));
});

offline_test!(
    sccache_save_skips_when_a_stored_entry_is_already_within_budget,
    {
        let sandbox_directory = std::env::temp_dir().join(format!(
            "astronomical-sccache-save-skip-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&sandbox_directory).unwrap();
        let shim = GhShim::install(&sandbox_directory, THREE_FAMILY_CACHE_LIST, &HashMap::new());
        let sccache_directory = sandbox_directory.join("sccache-dir");
        std::fs::create_dir_all(&sccache_directory).unwrap();
        std::fs::write(sccache_directory.join("small"), vec![0u8; 10]).unwrap();
        let github_output_path = output_path(&sandbox_directory);
        std::fs::write(&github_output_path, "").unwrap();

        // The stored oversize entry belongs to a different primary key, so this
        // key's stored entry (id 5) is within budget and the save is skipped.
        let mut save_command =
            shim.command_for_xtask(&["save-sccache", "astronomical-v2-sccache-macOS-good"]);
        save_command
            .env("GITHUB_REPOSITORY", "aosama/astronomical")
            .env("SCCACHE_DIR", &sccache_directory)
            .env("GITHUB_OUTPUT", &github_output_path);
        let save_run = save_command.output().expect("the xtask binary should run");
        assert!(save_run.status.success());
        let decisions = read_output_map(&github_output_path);
        assert_eq!(
            decisions.get("should_save").map(String::as_str),
            Some("false")
        );
        assert_eq!(
            decisions.get("decision").map(String::as_str),
            Some("fresh-entry-within-budget")
        );
    }
);

offline_test!(sccache_save_without_a_listable_cache_degrades_to_skip, {
    let sandbox_directory = std::env::temp_dir().join(format!(
        "astronomical-sccache-save-nolist-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&sandbox_directory).unwrap();
    // No `gh` on PATH: listing fails, and the read-only-token path must skip
    // the save instead of failing the run.
    let shim = GhShim::install(&sandbox_directory, "", &HashMap::new());
    std::fs::remove_file(&shim.script_path).unwrap();
    let github_output_path = output_path(&sandbox_directory);
    std::fs::write(&github_output_path, "").unwrap();

    let mut save_command = shim.command_for_xtask(&["save-sccache", "some-key"]);
    save_command
        .env("GITHUB_REPOSITORY", "aosama/astronomical")
        .env("SCCACHE_DIR", sandbox_directory.join("absent-dir"))
        .env("GITHUB_OUTPUT", &github_output_path);
    let save_run = save_command.output().expect("the xtask binary should run");
    assert!(save_run.status.success());
    let decisions = read_output_map(&github_output_path);
    assert_eq!(
        decisions.get("decision").map(String::as_str),
        Some("cache-list-unavailable")
    );
});
