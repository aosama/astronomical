//! Unit tests for the pure logic of `cargo xtask disposable-target`, the
//! typed port of the retired `scripts/run-in-disposable-cargo-target.sh`.

use crate::disposable_target::{
    UsageError, allocated_bytes_from_du_output, exit_code_from_status, is_valid_lane_name,
    lane_root_string_is_acceptable, marker_identity, parse_arguments, select_rustc_wrapper,
};

fn lane_arguments(lane_name: &str, command: &[&str]) -> Vec<String> {
    let mut arguments = vec!["--lane".to_owned(), lane_name.to_owned(), "--".to_owned()];
    arguments.extend(
        command
            .iter()
            .map(|command_part| (*command_part).to_owned()),
    );
    arguments
}

fn should_accept_lane_name(lane_name: &str) -> bool {
    is_valid_lane_name(lane_name)
}

mod should_validate_lane_names {
    use super::*;

    #[test]
    fn should_accept_lowercase_letters_digits_and_internal_hyphens() {
        for valid_lane_name in ["abc", "a-b-c", "lane123", "a1-b2", "direct-mlx", "x"] {
            assert!(
                should_accept_lane_name(valid_lane_name),
                "{valid_lane_name} should be a valid lane name"
            );
        }
    }

    #[test]
    fn should_reject_empty_lane_name() {
        assert!(!should_accept_lane_name(""));
    }

    #[test]
    fn should_reject_uppercase_letters() {
        assert!(!should_accept_lane_name("Direct-Mlx"));
    }

    #[test]
    fn should_reject_underscores_and_spaces() {
        assert!(!should_accept_lane_name("direct_mlx"));
        assert!(!should_accept_lane_name("direct mlx"));
    }

    #[test]
    fn should_reject_leading_and_trailing_hyphens() {
        assert!(!should_accept_lane_name("-escape"));
        assert!(!should_accept_lane_name("escape-"));
    }

    #[test]
    fn should_reject_path_traversal_lane_names() {
        assert!(!should_accept_lane_name("../escape"));
        assert!(!should_accept_lane_name("a/../b"));
    }
}

mod should_parse_arguments {
    use super::*;

    #[test]
    fn should_accept_lane_separator_and_command() {
        let journey = parse_arguments(&lane_arguments(
            "app-release",
            &["scripts/internal/x.sh", "--flag"],
        ))
        .expect("a well-formed invocation should parse");
        assert_eq!(journey.lane_name, "app-release");
        assert_eq!(journey.command, ["scripts/internal/x.sh", "--flag"]);
    }

    #[test]
    fn should_reject_a_missing_lane_flag() {
        assert!(matches!(
            parse_arguments(&[
                "build".to_owned(),
                "--".to_owned(),
                "cargo".to_owned(),
                "build".to_owned()
            ]),
            Err(UsageError::PrintUsage)
        ));
    }

    #[test]
    fn should_reject_too_few_arguments() {
        assert!(matches!(
            parse_arguments(&lane_arguments("lane", &[])),
            Err(UsageError::PrintUsage)
        ));
    }

    #[test]
    fn should_reject_an_invalid_lane_name_with_a_specific_error() {
        assert!(matches!(
            parse_arguments(&lane_arguments("../escape", &["cmd"])),
            Err(UsageError::LaneName(lane_name)) if lane_name == "../escape"
        ));
    }

    #[test]
    fn should_reject_a_missing_command_separator() {
        assert!(matches!(
            parse_arguments(&[
                "--lane".to_owned(),
                "lane".to_owned(),
                "cmd".to_owned(),
                "arg".to_owned()
            ]),
            Err(UsageError::PrintUsage)
        ));
    }
}

mod should_build_marker_identity {
    use super::*;

    #[test]
    fn should_embed_the_marker_version_and_lane_name() {
        assert_eq!(
            marker_identity("app-release"),
            "astronomical-disposable-cargo-target-v1:app-release"
        );
    }
}

mod should_account_allocated_bytes_from_du_output {
    use super::*;

    #[test]
    fn should_parse_tab_separated_kibibytes() {
        assert_eq!(
            allocated_bytes_from_du_output("4\t/private/tmp/astronomical-cargo-lane.abc123\n"),
            Ok(4 * 1024)
        );
    }

    #[test]
    fn should_parse_space_separated_kibibytes() {
        assert_eq!(
            allocated_bytes_from_du_output("2048 /private/tmp/target\n"),
            Ok(2048 * 1024)
        );
    }

    #[test]
    fn should_reject_non_numeric_output() {
        assert_eq!(allocated_bytes_from_du_output("n/a\t/path\n"), Err(()));
    }

    #[test]
    fn should_reject_empty_output() {
        assert_eq!(allocated_bytes_from_du_output(""), Err(()));
    }

    #[test]
    fn should_reject_overflowing_kibibyte_counts() {
        assert_eq!(
            allocated_bytes_from_du_output(&format!("{}\t/path\n", u64::MAX)),
            Err(())
        );
    }
}

mod should_validate_lane_root_strings {
    use super::*;

    #[test]
    fn should_accept_absolute_non_root_directories() {
        assert!(lane_root_string_is_acceptable("/private/tmp"));
        assert!(lane_root_string_is_acceptable("/var/folders/lane-root"));
    }

    #[test]
    fn should_reject_empty_relative_dot_and_root_paths() {
        assert!(!lane_root_string_is_acceptable(""));
        assert!(!lane_root_string_is_acceptable("."));
        assert!(!lane_root_string_is_acceptable(".."));
        assert!(!lane_root_string_is_acceptable("/"));
    }

    #[test]
    fn should_reject_relative_paths() {
        assert!(!lane_root_string_is_acceptable("tmp/cargo-lanes"));
    }
}

mod should_select_rustc_wrapper {
    use super::*;

    #[test]
    fn should_prefer_a_configured_wrapper() {
        assert_eq!(
            select_rustc_wrapper(Some("sccache"), true),
            Some("sccache".to_owned())
        );
        assert_eq!(
            select_rustc_wrapper(Some("custom-wrapper"), true),
            Some("custom-wrapper".to_owned())
        );
    }

    #[test]
    fn should_treat_an_empty_wrapper_configuration_as_unset() {
        assert_eq!(
            select_rustc_wrapper(Some(""), true),
            Some("sccache".to_owned())
        );
        assert_eq!(select_rustc_wrapper(Some(""), false), None);
    }

    #[test]
    fn should_fall_back_to_sccache_only_when_available() {
        assert_eq!(select_rustc_wrapper(None, true), Some("sccache".to_owned()));
        assert_eq!(select_rustc_wrapper(None, false), None);
    }
}

mod should_convert_exit_status_to_shell_status {
    use super::*;

    #[test]
    fn should_read_a_plain_exit_code() {
        let exit_status = run_status(23);
        assert_eq!(exit_code_from_status(&exit_status), 23);
    }

    #[test]
    fn should_read_zero_as_success() {
        let exit_status = run_status(0);
        assert_eq!(exit_code_from_status(&exit_status), 0);
    }

    fn run_status(code: i32) -> std::process::ExitStatus {
        let status_output = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("exit {code}"))
            .status()
            .expect("the exit-status fixture should run");
        status_output
    }
}
