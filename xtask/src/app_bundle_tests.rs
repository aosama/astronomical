use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::build_identity::{BuildIdentity, daemon_package_identity, github_pages_feed_url};
use super::build_steps::{npm_install_is_current, swift_product_build_command};
use super::bundle_assembly::{BundleInputs, missing_inputs};
use super::bundle_check::daemon_version_matches;
use super::channel_profile::{Channel, ChannelProfile, SandboxPolicy};
use super::code_signing::{SigningRequest, signing_plan};
use super::command::{AppCommand, DevOpenPlacement, PlistSubject, parse_app_command};
use super::dev_pipeline::DevPipelinePaths;
use super::icon_cache::cached_icon_file_name;
use super::info_plist::{escape_xml_text, render_agent_info_plist, render_info_plist};
use super::instance_control::{InstanceAction, status_report_lines};
use super::native_metallib::metallib_store_path;
use super::running_instance::open_bundle_arguments;
use super::step_reporter::estimate_remaining_seconds;

fn sample_build_identity(is_dirty: bool) -> BuildIdentity {
    BuildIdentity {
        application_version: "0.2.88".to_owned(),
        repository_url: "https://github.com/example-owner/astronomical".to_owned(),
        build_commit: "49af41bd9bba".to_owned(),
        build_number: "4242".to_owned(),
        build_date_utc: "20261102".to_owned(),
        is_dirty,
    }
}

fn arguments(argument_words: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_word in argument_words {
        owned_arguments.push((*argument_word).to_owned());
    }
    owned_arguments
}

#[test]
fn should_round_trip_every_channel_name() {
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        assert_eq!(
            Channel::from_channel_name(channel.as_channel_name()),
            Ok(channel)
        );
    }
}

#[test]
fn should_reject_an_unknown_channel_name() {
    assert!(Channel::from_channel_name("beta").is_err());
    assert!(Channel::from_channel_name("").is_err());
}

#[test]
fn should_keep_development_isolated_from_stable() {
    let development_profile = ChannelProfile::for_channel(Channel::Development);
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);

    assert_ne!(development_profile.daemon_port, stable_profile.daemon_port);
    assert_ne!(
        development_profile.bundle_identifier,
        stable_profile.bundle_identifier
    );
    assert_ne!(
        development_profile.state_directory_name,
        stable_profile.state_directory_name
    );
    assert!(!development_profile.requires_clean_worktree);
    assert!(stable_profile.requires_clean_worktree);
}

#[test]
fn should_strip_sparkle_and_state_directory_from_the_app_store_profile() {
    let app_store_profile = ChannelProfile::for_channel(Channel::AppStore);

    assert!(!app_store_profile.ships_sparkle);
    assert_eq!(app_store_profile.state_directory_name, None);
    assert_eq!(
        app_store_profile.sandbox_policy,
        SandboxPolicy::AppStoreSandbox
    );
    assert_eq!(
        app_store_profile.daemon_cargo_features,
        Some("astronomical-daemon/app-store-state-root")
    );
}

#[test]
fn should_render_development_plist_with_sparkle_state_directory_and_dirty_marker() {
    let plist_text = render_info_plist(
        &ChannelProfile::for_channel(Channel::Development),
        &sample_build_identity(true),
        "https://example-owner.github.io/astronomical/appcast.xml",
    );

    assert!(plist_text.contains(
        "<key>CFBundleIdentifier</key><string>dev.astronomical.app.development</string>"
    ));
    assert!(plist_text.contains("<key>SUFeedURL</key><string>https://example-owner.github.io/astronomical/appcast.xml</string>"));
    assert!(
        plist_text.contains(
            "<key>AstronomicalStateDirectoryName</key><string>.astronomical-dev</string>"
        )
    );
    assert!(plist_text.contains("<key>AstronomicalDaemonPort</key><integer>6733</integer>"));
    assert!(plist_text.contains("<key>AstronomicalBuildDirty</key><true/>"));
    assert!(plist_text.ends_with("</dict>\n</plist>\n"));
}

#[test]
fn should_render_app_store_plist_without_sparkle_or_state_directory() {
    let plist_text = render_info_plist(
        &ChannelProfile::for_channel(Channel::AppStore),
        &sample_build_identity(false),
        "https://example-owner.github.io/astronomical/appcast.xml",
    );

    assert!(!plist_text.contains("SUFeedURL"));
    assert!(!plist_text.contains("SUPublicEDKey"));
    assert!(!plist_text.contains("AstronomicalStateDirectoryName"));
    assert!(plist_text.contains("<key>AstronomicalBuildDirty</key><false/>"));
}

/// App Store Connect refuses a Mac app with no category, so every channel's
/// main app names the decided one; the agent is a helper and names none.
#[test]
fn should_name_the_productivity_category_in_every_main_app_plist_only() {
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        let channel_profile = ChannelProfile::for_channel(channel);
        let application_plist_text = render_info_plist(
            &channel_profile,
            &sample_build_identity(false),
            "https://example-owner.github.io/astronomical/appcast.xml",
        );
        let agent_plist_text =
            render_agent_info_plist(&channel_profile, &sample_build_identity(false));

        assert!(
            application_plist_text.contains(
                "<key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>"
            ),
            "{channel:?}"
        );
        assert!(
            !agent_plist_text.contains("LSApplicationCategoryType"),
            "{channel:?}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn should_render_a_plist_that_plutil_accepts_for_every_channel() {
    let plist_directory = tempfile::tempdir().expect("a temporary directory should be available");
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        let plist_path = plist_directory
            .path()
            .join(format!("{}.plist", channel.as_channel_name()));
        let plist_text = render_info_plist(
            &ChannelProfile::for_channel(channel),
            &sample_build_identity(false),
            "https://example-owner.github.io/astronomical/appcast.xml?a=1&b=2",
        );
        std::fs::write(&plist_path, plist_text).expect("the plist should be writable");

        let lint_status = std::process::Command::new("plutil")
            .arg("-lint")
            .arg(&plist_path)
            .output()
            .expect("plutil should run on macOS");

        assert!(
            lint_status.status.success(),
            "plutil rejected the {channel:?} plist"
        );
    }
}

#[test]
fn should_escape_xml_special_characters() {
    assert_eq!(escape_xml_text("a&b<c>\"d"), "a&amp;b&lt;c&gt;&quot;d");
    assert_eq!(escape_xml_text(""), "");
    assert_eq!(escape_xml_text("plain"), "plain");
}

#[test]
fn should_derive_the_github_pages_feed_url() {
    assert_eq!(
        github_pages_feed_url("https://github.com/example-owner/astronomical.git"),
        Ok("https://example-owner.github.io/astronomical/appcast.xml".to_owned())
    );
    assert_eq!(
        github_pages_feed_url("https://github.com/example-owner/astronomical"),
        Ok("https://example-owner.github.io/astronomical/appcast.xml".to_owned())
    );
}

#[test]
fn should_reject_a_repository_url_that_is_not_an_https_github_repository() {
    assert!(github_pages_feed_url("git@github.com:example-owner/astronomical.git").is_err());
    assert!(github_pages_feed_url("https://github.com/example-owner").is_err());
    assert!(github_pages_feed_url("https://github.com/example-owner/astronomical/extra").is_err());
}

#[test]
fn should_read_the_daemon_version_and_repository_from_cargo_metadata() {
    let cargo_metadata_text = r#"{"packages":[
        {"name":"astronomical-cli","version":"9.9.9","repository":"https://github.com/other/cli"},
        {"name":"astronomical-daemon","version":"0.2.88","repository":"https://github.com/example-owner/astronomical"}
    ]}"#;

    assert_eq!(
        daemon_package_identity(cargo_metadata_text),
        Ok((
            "0.2.88".to_owned(),
            "https://github.com/example-owner/astronomical".to_owned()
        ))
    );
}

#[test]
fn should_fail_when_cargo_metadata_lacks_the_daemon_package() {
    assert!(daemon_package_identity(r#"{"packages":[]}"#).is_err());
    assert!(daemon_package_identity("not json").is_err());
    assert!(
        daemon_package_identity(
            r#"{"packages":[{"name":"astronomical-daemon","version":"1.0.0"}]}"#
        )
        .is_err()
    );
}

#[test]
fn should_skip_npm_ci_only_when_the_lockfile_stamp_matches_installed_modules() {
    assert!(npm_install_is_current("abc123", "abc123\n", true));
    assert!(!npm_install_is_current("abc123", "abc123", false));
    assert!(!npm_install_is_current("abc123", "def456", true));
    assert!(!npm_install_is_current("", "", true));
}

#[test]
fn should_estimate_no_eta_before_every_step_has_a_previous_timing() {
    let mut previous_step_seconds = BTreeMap::new();
    previous_step_seconds.insert("compile".to_owned(), 10.0);

    assert_eq!(
        estimate_remaining_seconds(&["compile", "sign"], 0, 0.0, &previous_step_seconds),
        None
    );
}

#[test]
fn should_estimate_eta_from_the_rest_of_the_current_step_and_later_steps() {
    let mut previous_step_seconds = BTreeMap::new();
    previous_step_seconds.insert("compile".to_owned(), 10.0);
    previous_step_seconds.insert("sign".to_owned(), 2.0);

    assert_eq!(
        estimate_remaining_seconds(&["compile", "sign"], 0, 4.0, &previous_step_seconds),
        Some(8.0)
    );
    // A step running longer than last time contributes zero, never negative.
    assert_eq!(
        estimate_remaining_seconds(&["compile", "sign"], 0, 30.0, &previous_step_seconds),
        Some(2.0)
    );
    assert_eq!(
        estimate_remaining_seconds(&["compile", "sign"], 2, 0.0, &previous_step_seconds),
        Some(0.0)
    );
}

#[test]
fn should_key_the_cached_icon_on_channel_and_renderer_digest() {
    assert_eq!(
        cached_icon_file_name("development", "0123456789abcdef0123"),
        "development-0123456789abcdef.icns"
    );
    assert_eq!(cached_icon_file_name("stable", "abc"), "stable-abc.icns");
}

#[test]
fn should_place_the_metallib_inside_the_versioned_native_store_entry() {
    assert_eq!(
        metallib_store_path(Path::new("/home-fixture"), 3, "feedface"),
        PathBuf::from(
            "/home-fixture/Library/Caches/Astronomical/native-builds/v3/entries/feedface/share/mlx/mlx.metallib"
        )
    );
}

#[test]
fn should_sign_sparkle_first_and_the_bundle_last() {
    let development_profile = ChannelProfile::for_channel(Channel::Development);
    let signing_invocations = signing_plan(&SigningRequest {
        bundle_path: Path::new("/fixture/Dev.app"),
        channel_profile: &development_profile,
        signing_identity: None,
        repository_root: Path::new("/repository"),
    })
    .expect("an ad-hoc Development plan is valid");

    assert_eq!(signing_invocations.len(), 12);
    assert!(
        signing_invocations[0]
            .code_object_path
            .ends_with("Sparkle.framework/Versions/B/Autoupdate")
    );
    assert!(
        signing_invocations[4]
            .code_object_path
            .ends_with("Frameworks/Sparkle.framework")
    );
    // A seal covers what is beneath it: the agent's executable, then the
    // agent's bundle, both before the main bundle that contains them.
    assert!(signing_invocations[5].code_object_path.ends_with(
        "Contents/Library/LoginItems/Astronomical Development Agent.app/Contents/MacOS/astronomical-menu"
    ));
    assert!(
        signing_invocations[6]
            .code_object_path
            .ends_with("Contents/Library/LoginItems/Astronomical Development Agent.app")
    );
    assert!(
        signing_invocations[7]
            .code_object_path
            .ends_with("Contents/MacOS/astronomical-app")
    );
    assert_eq!(
        signing_invocations[11].code_object_path,
        PathBuf::from("/fixture/Dev.app")
    );
    for signing_invocation in &signing_invocations {
        assert_eq!(
            signing_invocation.codesign_arguments,
            vec!["--force", "--sign", "-", "--preserve-metadata=entitlements"]
        );
    }
}

#[test]
fn should_match_the_daemon_version_only_with_this_builds_version_and_commit() {
    let build_identity = sample_build_identity(false);

    assert!(daemon_version_matches(
        "astronomicald 0.2.88 (49af41bd9bba)",
        &build_identity
    ));
    assert!(!daemon_version_matches(
        "astronomicald 0.2.87 (49af41bd9bba)",
        &build_identity
    ));
    assert!(!daemon_version_matches(
        "astronomicald 0.2.88 (000000000000)",
        &build_identity
    ));
    assert!(!daemon_version_matches("", &build_identity));
}

#[test]
fn should_parse_the_app_subcommands() {
    assert_eq!(
        parse_app_command(&arguments(&["dev"])),
        Ok(AppCommand::Dev(DevOpenPlacement::InFront))
    );
    assert_eq!(
        parse_app_command(&arguments(&["dev", "--background"])),
        Ok(AppCommand::Dev(DevOpenPlacement::InBackground))
    );
    assert!(parse_app_command(&arguments(&["dev", "--background", "--background"])).is_err());
    assert!(parse_app_command(&arguments(&["dev", "--front"])).is_err());
    assert_eq!(
        parse_app_command(&arguments(&["info-plist", "--channel", "stable"])),
        Ok(AppCommand::InfoPlist(
            Channel::Stable,
            PlistSubject::MainApplication
        ))
    );
    assert_eq!(
        parse_app_command(&arguments(&[
            "info-plist",
            "--channel",
            "development",
            "--agent"
        ])),
        Ok(AppCommand::InfoPlist(
            Channel::Development,
            PlistSubject::Agent
        ))
    );
    assert!(
        parse_app_command(&arguments(&[
            "info-plist",
            "--channel",
            "stable",
            "--helper"
        ]))
        .is_err(),
        "only --agent may follow the channel"
    );
}

#[test]
fn should_reject_unknown_or_incomplete_app_subcommands() {
    assert!(parse_app_command(&arguments(&[])).is_err());
    assert!(parse_app_command(&arguments(&["dev", "--release"])).is_err());
    assert!(parse_app_command(&arguments(&["info-plist", "--channel"])).is_err());
    assert!(parse_app_command(&arguments(&["info-plist", "--channel", "beta"])).is_err());
}

#[test]
fn should_stage_and_keep_caches_outside_the_cargo_profile_directory() {
    let pipeline_paths = DevPipelinePaths::for_profile(
        Path::new("/repository"),
        &ChannelProfile::for_channel(Channel::Development),
    );

    assert_eq!(
        pipeline_paths.bundle_path,
        PathBuf::from(
            "/repository/target/astronomical-macos-development.noindex/Astronomical Development.app"
        )
    );
    assert!(
        pipeline_paths
            .staging_bundle_path
            .starts_with("/repository/target/astronomical-macos-development.noindex/.staging")
    );
    assert_eq!(
        pipeline_paths.cache_directory,
        PathBuf::from("/repository/target/app-dev-cache")
    );
}

#[test]
fn should_name_every_missing_bundle_input() {
    let bundle_inputs = BundleInputs::for_repository(
        Path::new("/absent-repository"),
        &ChannelProfile::for_channel(Channel::Development),
        Path::new("/absent-repository/target/app-dev"),
        PathBuf::from("/absent-repository/mlx.metallib"),
        PathBuf::from("/absent-repository/icon.icns"),
    );

    let missing_paths = missing_inputs(&bundle_inputs);

    assert_eq!(missing_paths.len(), bundle_inputs.every_source_path().len());
    assert!(missing_paths.contains(&PathBuf::from("/absent-repository/mlx.metallib")));
}

/// `.build/release` is the only output path both SwiftPM build systems
/// provide; `out/Products` exists only under swiftbuild, so a native-build
/// runner (the CI image's Swift) would miss any input read from there.
#[test]
fn should_read_every_swift_build_product_from_the_release_directory() {
    let swift_build_directory = Path::new("/repository/apps/astronomical-menu/.build");
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        let bundle_inputs = BundleInputs::for_repository(
            Path::new("/repository"),
            &ChannelProfile::for_channel(channel),
            Path::new("/repository/target/release"),
            PathBuf::from("/repository/mlx.metallib"),
            PathBuf::from("/repository/icon.icns"),
        );
        for source_path in bundle_inputs.every_source_path() {
            assert!(
                !source_path.to_string_lossy().contains("out/Products"),
                "{} reads the swiftbuild-only layout",
                source_path.display()
            );
            let Ok(path_inside_swift_build) = source_path.strip_prefix(swift_build_directory)
            else {
                continue;
            };
            assert!(
                path_inside_swift_build.starts_with("release")
                    || path_inside_swift_build.starts_with("checkouts"),
                "{} is a Swift input outside .build/release",
                source_path.display()
            );
        }
    }
}

#[test]
fn should_parse_every_instance_action() {
    assert_eq!(
        parse_app_command(&arguments(&["instance", "status"])),
        Ok(AppCommand::Instance(InstanceAction::Status))
    );
    assert_eq!(
        parse_app_command(&arguments(&["instance", "restart"])),
        Ok(AppCommand::Instance(InstanceAction::Restart))
    );
    assert_eq!(
        InstanceAction::from_action_name("start"),
        Ok(InstanceAction::Start)
    );
    assert_eq!(
        InstanceAction::from_action_name("stop"),
        Ok(InstanceAction::Stop)
    );
}

#[test]
fn should_reject_unknown_instance_actions() {
    assert!(parse_app_command(&arguments(&["instance", "kill"])).is_err());
    assert!(parse_app_command(&arguments(&["instance"])).is_err());
}

/// The retired shell guard proved the Development builder refuses a Stable
/// channel selection; `app dev` takes no channel at all.
#[test]
fn should_refuse_a_channel_selection_on_the_development_builder() {
    assert!(parse_app_command(&arguments(&["dev", "--channel", "stable"])).is_err());
}

fn fixture_status_document(daemon_commit: &str) -> serde_json::Value {
    serde_json::json!({
        "status": "ready",
        "activity": "idle",
        "ready_model_id": "qwen-fixture",
        "expert_memory_mode": null,
        "mlx_memory_ceiling_bytes": 36_507_222_016_u64,
        "machine_mlx_memory_ceiling_bytes": 42_949_672_960_u64,
        "configured_maximum_mlx_memory_gb": 34,
        "serving_session": {
            "completed_request_count": 2,
            "average_prefill_tok_per_second": 812.4,
            "average_generation_tok_per_second": 41.6
        },
        "application": { "commit": daemon_commit, "is_dirty": true }
    })
}

#[test]
fn should_report_status_with_gigabytes_and_a_matching_build() {
    let status_lines =
        status_report_lines(&fixture_status_document("49af41bd9bba"), "49af41bd9bba0000");

    assert_eq!(status_lines[0], "Development instance: ready (idle)");
    assert_eq!(status_lines[1], "  model:             qwen-fixture  [-]");
    assert_eq!(
        status_lines[2],
        "  memory ceiling:    34.00 GB effective of 40.00 GB (configured 34, pending -)"
    );
    assert_eq!(
        status_lines[3],
        "  session:           2 requests · 812 tok/s prefill · 42 tok/s decode"
    );
    assert_eq!(
        status_lines[4],
        "  build: 49af41bd9bba (matches HEAD; dirty true)"
    );
}

#[test]
fn should_warn_when_the_running_daemon_was_built_from_another_commit() {
    let status_lines =
        status_report_lines(&fixture_status_document("0123456789ab"), "49af41bd9bba0000");

    assert!(status_lines[4].starts_with("  WARNING: built from 0123456789ab"));
    assert!(status_lines[4].ends_with("rebuild with: cargo xtask app dev"));
}

#[test]
fn should_report_unknown_fields_as_dashes_for_a_sparse_status_document() {
    let status_lines = status_report_lines(&serde_json::json!({}), "49af41bd9bba");

    assert_eq!(status_lines[0], "Development instance: - (-)");
    assert_eq!(
        status_lines[2],
        "  memory ceiling:    - effective of - (configured -, pending -)"
    );
    assert_eq!(status_lines[4], "  build: unknown");
}

#[test]
fn should_render_each_channels_own_chat_scheme_into_the_info_plist() {
    for (channel, chat_scheme_entry) in [
        (Channel::Development, "<string>astronomical-dev</string>"),
        (Channel::Stable, "<string>astronomical</string>"),
        (Channel::AppStore, "<string>astronomical</string>"),
    ] {
        let plist_text = render_info_plist(
            &ChannelProfile::for_channel(channel),
            &sample_build_identity(false),
            "https://example-owner.github.io/astronomical/appcast.xml",
        );

        assert!(
            plist_text.contains(chat_scheme_entry),
            "{chat_scheme_entry} missing for {channel:?}"
        );
    }
}

#[test]
fn should_not_render_the_stable_chat_scheme_into_a_development_info_plist() {
    let plist_text = render_info_plist(
        &ChannelProfile::for_channel(Channel::Development),
        &sample_build_identity(true),
        "https://example-owner.github.io/astronomical/appcast.xml",
    );

    assert!(!plist_text.contains("<string>astronomical</string>"));
}

fn command_arguments(tool_command: &std::process::Command) -> Vec<String> {
    let mut arguments = Vec::new();
    for argument in tool_command.get_args() {
        arguments.push(argument.to_string_lossy().into_owned());
    }
    arguments
}

#[test]
fn should_hand_the_macos_sdk_to_the_menu_link_step_so_the_app_gets_the_current_design() {
    let swift_command = swift_product_build_command(
        Path::new("/repository"),
        "astronomical-menu",
        "/sdks/MacOSX.sdk",
    );
    assert_eq!(swift_command.get_program(), "swift");
    assert_eq!(
        command_arguments(&swift_command),
        vec![
            "build",
            "--configuration",
            "release",
            "--package-path",
            "/repository/apps/astronomical-menu",
            "--product",
            "astronomical-menu",
            "-Xswiftc",
            "-warnings-as-errors",
            "-Xswiftc",
            "-Xclang-linker",
            "-Xswiftc",
            "-isysroot",
            "-Xswiftc",
            "-Xclang-linker",
            "-Xswiftc",
            "/sdks/MacOSX.sdk",
        ]
    );
}

#[test]
fn should_pass_an_sdk_path_with_spaces_as_one_argument() {
    let swift_command = swift_product_build_command(
        Path::new("/repository"),
        "astronomical-app-app-store",
        "/Applications/Xcode Beta.app/SDKs/MacOSX.sdk",
    );
    let arguments = command_arguments(&swift_command);
    assert_eq!(
        arguments.last().map(String::as_str),
        Some("/Applications/Xcode Beta.app/SDKs/MacOSX.sdk")
    );
    assert!(arguments.contains(&"astronomical-app-app-store".to_string()));
}

/// `open -g` launches the app inactive, but the app raises its own window
/// unless the launch argument tells it not to, so a background open needs
/// both; a normal open needs neither.
#[test]
fn should_open_the_dev_app_behind_other_windows_only_when_asked() {
    let bundle_path = Path::new("/builds/Astronomical Development.app");

    assert_eq!(
        open_bundle_arguments(bundle_path, DevOpenPlacement::InFront),
        vec![std::ffi::OsString::from(bundle_path)]
    );
    assert_eq!(
        open_bundle_arguments(bundle_path, DevOpenPlacement::InBackground),
        vec![
            std::ffi::OsString::from("-g"),
            std::ffi::OsString::from(bundle_path),
            std::ffi::OsString::from("--args"),
            std::ffi::OsString::from("--launch-in-background"),
        ]
    );
}
