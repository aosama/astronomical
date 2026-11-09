use std::path::PathBuf;

use super::{
    ReleaseCommand, parse_app_store_package_command, parse_build_and_install_stable_command,
    parse_distribution_validation_command, parse_install_stable_command, parse_release_command,
};

fn arguments(argument_texts: &[&str]) -> Vec<String> {
    argument_texts.iter().map(ToString::to_string).collect()
}

#[test]
fn should_route_every_subcommand_to_its_own_parser() {
    let routed_command = parse_release_command(&arguments(&[
        "validate-distribution-app",
        "--app-bundle",
        "/Astronomical.app",
        "--team-id",
        "TEAM123456",
    ]));
    assert_eq!(
        routed_command,
        Ok(ReleaseCommand::ValidateDistributionApp {
            app_bundle: PathBuf::from("/Astronomical.app"),
            team_identifier: "TEAM123456".to_owned(),
        })
    );
    assert_eq!(
        parse_release_command(&arguments(&["accept-sparkle-update"])),
        Ok(ReleaseCommand::AcceptSparkleUpdate)
    );
}

#[test]
fn should_refuse_arguments_after_accept_sparkle_update() {
    assert!(parse_release_command(&arguments(&["accept-sparkle-update", "--dry-run"])).is_err());
}

#[test]
fn should_name_the_missing_team_identifier_for_distribution_validation() {
    let refusal =
        parse_distribution_validation_command(&arguments(&["--app-bundle", "/Astronomical.app"]))
            .expect_err("a missing --team-id must be refused");
    assert!(refusal.starts_with("--team-id is required"), "{refusal}");
}

#[test]
fn should_leave_the_app_store_output_package_optional() {
    let parsed_command = parse_app_store_package_command(&arguments(&[
        "--app-bundle",
        "/Astronomical.app",
        "--installer-identity",
        "3rd Party Mac Developer Installer",
    ]));
    let Ok(ReleaseCommand::AppStorePackage(package_request)) = parsed_command else {
        panic!("expected an app store package command, got {parsed_command:?}");
    };
    assert_eq!(package_request.output_package, None);
    assert_eq!(
        package_request.installer_identity,
        "3rd Party Mac Developer Installer"
    );
}

#[test]
fn should_refuse_an_app_store_package_without_an_installer_identity() {
    assert!(
        parse_app_store_package_command(&arguments(&["--app-bundle", "/Astronomical.app"]))
            .is_err()
    );
}

#[test]
fn should_accept_the_dry_run_switch_on_either_side_of_the_install_bundle() {
    for argument_order in [
        ["--dry-run", "--app-bundle", "/Astronomical.app"],
        ["--app-bundle", "/Astronomical.app", "--dry-run"],
    ] {
        assert_eq!(
            parse_install_stable_command(&arguments(&argument_order)),
            Ok(ReleaseCommand::InstallStable {
                app_bundle: Some(PathBuf::from("/Astronomical.app")),
                is_dry_run: true,
            })
        );
    }
}

#[test]
fn should_install_the_default_bundle_when_no_install_argument_is_given() {
    assert_eq!(
        parse_install_stable_command(&[]),
        Ok(ReleaseCommand::InstallStable {
            app_bundle: None,
            is_dry_run: false,
        })
    );
}

#[test]
fn should_refuse_an_install_bundle_flag_without_its_path() {
    assert!(parse_install_stable_command(&arguments(&["--app-bundle", "--dry-run"])).is_err());
}

#[test]
fn should_accept_only_an_empty_or_dry_run_build_and_install_argument_list() {
    assert_eq!(
        parse_build_and_install_stable_command(&[]),
        Ok(ReleaseCommand::BuildAndInstallStable { is_dry_run: false })
    );
    assert_eq!(
        parse_build_and_install_stable_command(&arguments(&["--dry-run"])),
        Ok(ReleaseCommand::BuildAndInstallStable { is_dry_run: true })
    );
    assert!(
        parse_build_and_install_stable_command(&arguments(&["--dry-run", "--dry-run"])).is_err()
    );
    assert!(parse_build_and_install_stable_command(&arguments(&["--publish"])).is_err());
}
