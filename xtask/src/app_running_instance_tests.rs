//! Contracts for finding, quitting and reopening the running applications of
//! a channel: the command-line patterns `pgrep -f` matches, whether an
//! application was opened by path or started by macOS as a login item, and
//! the agent a restart brings back.

use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};

use super::channel_profile::{Channel, ChannelProfile};
use super::instance_control::reopen_agent_if_it_was_running;
use super::running_instance::{
    agent_process_pattern, escape_extended_regular_expression, login_or_path_command_line_pattern,
    running_process_count,
};

/// A command line as `pgrep -f` sees it, from a channel's bundle built in
/// `worktree_name`'s target directory.
fn bundle_command_line(worktree_name: &str, channel: Channel, executable_name: &str) -> String {
    let channel_profile = ChannelProfile::for_channel(channel);
    format!(
        "/repository/.claude/worktrees/{worktree_name}/target/{}/{}.app/{}",
        channel_profile.output_directory_name,
        channel_profile.bundle_name,
        executable_relative_path(&channel_profile, executable_name)
    )
}

/// The agent runs from its nested bundle; everything else from Contents/MacOS.
fn executable_relative_path(channel_profile: &ChannelProfile, executable_name: &str) -> String {
    if executable_name == "astronomical-menu" {
        return channel_profile.agent_executable_relative_path();
    }
    format!("Contents/MacOS/{executable_name}")
}

#[test]
fn should_match_the_development_menu_app_from_any_worktree() {
    let development_pattern =
        agent_process_pattern(&ChannelProfile::for_channel(Channel::Development));

    for worktree_name in ["first-worktree", "second-worktree"] {
        let command_line =
            bundle_command_line(worktree_name, Channel::Development, "astronomical-menu");
        assert!(
            command_line.contains(&development_pattern),
            "{command_line}"
        );
    }
}

#[test]
fn should_not_match_the_development_menu_app_with_the_stable_pattern() {
    let stable_pattern = agent_process_pattern(&ChannelProfile::for_channel(Channel::Stable));
    let development_command_line =
        bundle_command_line("any-worktree", Channel::Development, "astronomical-menu");

    assert!(!development_command_line.contains(&stable_pattern));
}

#[test]
fn should_keep_stable_and_app_store_menu_apps_apart_despite_one_bundle_name() {
    let stable_pattern = agent_process_pattern(&ChannelProfile::for_channel(Channel::Stable));
    let stable_command_line =
        bundle_command_line("any-worktree", Channel::Stable, "astronomical-menu");
    let app_store_command_line =
        bundle_command_line("any-worktree", Channel::AppStore, "astronomical-menu");

    assert!(stable_command_line.contains(&stable_pattern));
    assert!(!app_store_command_line.contains(&stable_pattern));
}

#[test]
fn should_not_match_the_daemon_or_worker_inside_the_same_bundle() {
    let development_pattern =
        agent_process_pattern(&ChannelProfile::for_channel(Channel::Development));

    for executable_name in ["astronomicald", "astronomical-inference-worker"] {
        let command_line =
            bundle_command_line("any-worktree", Channel::Development, executable_name);
        assert!(
            !command_line.contains(&development_pattern),
            "{command_line}"
        );
    }
}

#[test]
fn should_escape_every_character_a_regular_expression_treats_specially() {
    assert_eq!(
        escape_extended_regular_expression(r"a.b[c](d)*e+f?g{h}i|j^k$l\m"),
        r"a\.b\[c\]\(d\)\*e\+f\?g\{h\}i\|j\^k\$l\\m"
    );
}

#[test]
fn should_leave_plain_text_and_spaces_as_they_are_when_escaping() {
    for plain_text in [
        "",
        "Astronomical Development",
        "/Contents/MacOS/astronomical-menu",
    ] {
        assert_eq!(escape_extended_regular_expression(plain_text), plain_text);
    }
}

#[test]
fn should_match_a_path_anywhere_or_a_command_line_that_is_only_the_label() {
    assert_eq!(
        login_or_path_command_line_pattern(
            "/out.noindex/Some App.app/Contents/MacOS/agent",
            "dev.example.app.agent"
        ),
        r"/out\.noindex/Some App\.app/Contents/MacOS/agent|^dev\.example\.app\.agent$"
    );
}

/// A stand-in for an application: `/bin/cat` reached through a link, so its
/// path can be bundle-shaped, kept alive by an open input it never receives.
fn start_stand_in(stand_in_executable: &Path, first_launch_argument: Option<&str>) -> Child {
    let mut stand_in_command = Command::new(stand_in_executable);
    if let Some(first_launch_argument) = first_launch_argument {
        stand_in_command.arg0(first_launch_argument);
    }
    stand_in_command
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("the stand-in should start")
}

/// Runs real processes in both launch shapes and asks `pgrep` itself. The
/// label is unique to this test run, so an agent really running on the Mac
/// is never counted. Matched by path alone, the label-launched process was
/// invisible, which is how a stop left a login-launched agent running.
#[test]
fn should_count_processes_opened_by_path_and_started_under_the_label_and_no_others() {
    let scratch_directory = tempfile::tempdir().expect("a temporary directory");
    let stand_in_executable = scratch_directory
        .path()
        .join("Some Agent.app/Contents/MacOS/astronomical-menu");
    std::fs::create_dir_all(stand_in_executable.parent().expect("a parent directory"))
        .expect("the bundle directories should be creatable");
    std::os::unix::fs::symlink("/bin/cat", &stand_in_executable).expect("the stand-in should link");
    let service_label = format!("dev.example.run{}.agent", std::process::id());
    let command_line_pattern =
        login_or_path_command_line_pattern(&stand_in_executable.to_string_lossy(), &service_label);
    let longer_label = format!("{service_label}.helper");
    let dotless_label = service_label.replace('.', "x");
    let mut stand_in_processes = [
        start_stand_in(&stand_in_executable, None),
        start_stand_in(Path::new("/bin/cat"), Some(&service_label)),
        start_stand_in(Path::new("/bin/cat"), Some(&longer_label)),
        start_stand_in(Path::new("/bin/cat"), Some(&dotless_label)),
    ];

    let path_only_count = running_process_count(&escape_extended_regular_expression(
        &stand_in_executable.to_string_lossy(),
    ));
    let path_or_label_count = running_process_count(&command_line_pattern);
    for stand_in_process in &mut stand_in_processes {
        stand_in_process.kill().expect("the stand-in should stop");
        stand_in_process
            .wait()
            .expect("the stand-in should be reaped");
    }
    let count_after_exit = running_process_count(&command_line_pattern);

    assert_eq!(
        path_only_count, 1,
        "the path alone sees only the path launch"
    );
    assert_eq!(path_or_label_count, 2, "{command_line_pattern}");
    assert_eq!(count_after_exit, 0);
}

fn open_agent_succeeds(_bundle_path: &Path) -> Result<(), String> {
    Ok(())
}

fn open_agent_fails(_bundle_path: &Path) -> Result<(), String> {
    Err("open refused".to_owned())
}

#[test]
fn should_reopen_a_menu_app_that_was_running() {
    let reopen_outcome = reopen_agent_if_it_was_running(
        true,
        Path::new("/fictional/Fictional.app"),
        open_agent_succeeds,
    );

    assert_eq!(reopen_outcome, Ok(()));
}

#[test]
fn should_not_open_a_menu_app_that_was_not_running() {
    // The opener always fails, so an Ok here proves it was never called.
    let reopen_outcome = reopen_agent_if_it_was_running(
        false,
        Path::new("/fictional/Fictional.app"),
        open_agent_fails,
    );

    assert_eq!(reopen_outcome, Ok(()));
}

#[test]
fn should_say_the_daemon_is_running_when_the_menu_app_cannot_reopen() {
    let reopen_outcome = reopen_agent_if_it_was_running(
        true,
        Path::new("/fictional/Fictional.app"),
        open_agent_fails,
    );

    assert_eq!(
        reopen_outcome,
        Err(
            "the daemon is running, but the menu-bar agent did not reopen: open refused".to_owned()
        )
    );
}
