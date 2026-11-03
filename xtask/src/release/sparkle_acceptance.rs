//! `cargo xtask release accept-sparkle-update`: proves, through Sparkle's
//! own command-line update driver, that a signed update installs and
//! relaunches, that a corrupted archive is rejected, and that external user
//! state is untouched either way. Typed port of the retired
//! `accept-sparkle-update.sh`.
//!
//! The shell bounded the whole run, cold Swift and Xcode builds included, at
//! 120 seconds through a `perl` alarm re-exec; a cold build alone can exceed
//! that. Here tool resolution reports its own progress, and the 120-second
//! budget covers the acceptance itself, enforced by deadline-aware waits on
//! every spawned process.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use crate::app_bundle::{SPARKLE_PUBLIC_ED25519_KEY, escape_xml_text};
use crate::step_log::{log_line, start_step};

const ACCEPTANCE_BUDGET: Duration = Duration::from_secs(120);
const FILE_WAIT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(250);
const ACCEPTANCE_BUNDLE_IDENTIFIER: &str = "dev.astronomical.update-acceptance";
const ACCEPTANCE_APP_NAME: &str = "Astronomical Update Acceptance.app";
const UPDATE_IMAGE_NAME: &str = "Astronomical-Update-Acceptance-2.0.0.dmg";
const STATE_MARKER_PATHS: [&str; 4] = [
    "config/config.json",
    "models/model-marker",
    "cache/cache-marker",
    "logs/log-marker",
];

pub(crate) struct SparkleTools {
    sparkle_framework: PathBuf,
    appcast_tool: PathBuf,
    sparkle_cli: PathBuf,
}

/// Everything the acceptance spawned or created, torn down on every exit.
struct AcceptanceRun {
    acceptance_root: PathBuf,
    deadline: Instant,
    server_process: Option<Child>,
    app_process: Option<Child>,
}

impl Drop for AcceptanceRun {
    fn drop(&mut self) {
        stop_relaunched_app(&self.acceptance_root);
        for spawned_process in [self.app_process.as_mut(), self.server_process.as_mut()]
            .into_iter()
            .flatten()
        {
            let _ = spawned_process.kill();
            let _ = spawned_process.wait();
        }
        let _ = std::fs::remove_dir_all(&self.acceptance_root);
    }
}

pub(crate) fn accept_sparkle_update(repository_root: &Path) -> Result<(), String> {
    let tools_step = start_step("resolve-sparkle-tools");
    let sparkle_tools = resolve_sparkle_tools(repository_root)?;
    tools_step.finish("success");
    let acceptance_root = repository_root.join("target/sparkle-update-acceptance.noindex");
    let _ = std::fs::remove_dir_all(&acceptance_root);
    create_directory(&acceptance_root.join("served"))?;
    let mut acceptance_run = AcceptanceRun {
        acceptance_root,
        deadline: Instant::now() + ACCEPTANCE_BUDGET,
        server_process: None,
        app_process: None,
    };
    let feed_url = start_loopback_server(repository_root, &mut acceptance_run)?;
    let (old_app, _new_app) =
        assemble_acceptance_apps(repository_root, &acceptance_run, &sparkle_tools, &feed_url)?;
    generate_signed_appcast(&acceptance_run, &sparkle_tools, &feed_url)?;
    let state_before = seed_external_state(&acceptance_run.acceptance_root.join("state"))?;
    install_and_relaunch(
        &mut acceptance_run,
        &sparkle_tools,
        &old_app,
        &feed_url,
        &state_before,
    )?;
    reject_corrupted_update(
        &acceptance_run,
        &sparkle_tools,
        &old_app,
        &feed_url,
        &state_before,
    )?;
    println!(
        "Sparkle acceptance passed: signed update installed and relaunched; corrupted update rejected; external state preserved."
    );
    Ok(())
}

fn resolve_sparkle_tools(repository_root: &Path) -> Result<SparkleTools, String> {
    let menu_package = repository_root.join("apps/astronomical-menu");
    run_checked(
        Command::new("swift")
            .args(["package", "resolve", "--package-path"])
            .arg(&menu_package),
        None,
    )?;
    run_checked(
        Command::new("swift")
            .args(["build", "--configuration", "release", "--package-path"])
            .arg(&menu_package)
            .args(["-Xswiftc", "-warnings-as-errors"]),
        None,
    )?;
    let sparkle_checkout = menu_package.join(".build/checkouts/Sparkle");
    let sparkle_cli = sparkle_checkout.join("build/Release/sparkle.app/Contents/MacOS/sparkle");
    if !sparkle_cli.is_file() {
        // Sparkle's Xcode project pins MACOSX_DEPLOYMENT_TARGET=10.13, which
        // current Xcode rejects; the override only affects this local
        // fixture updater, never the shipped SwiftPM artifacts.
        run_checked(
            Command::new("xcodebuild")
                .arg("-project")
                .arg(sparkle_checkout.join("Sparkle.xcodeproj"))
                .args([
                    "-target",
                    "sparkle-cli",
                    "-configuration",
                    "Release",
                    "ONLY_ACTIVE_ARCH=YES",
                    "ARCHS=arm64",
                ])
                .args(["MACOSX_DEPLOYMENT_TARGET=12.0", "build"]),
            None,
        )?;
    }
    let sparkle_tools = SparkleTools {
        sparkle_framework: menu_package.join(".build/release/Sparkle.framework"),
        appcast_tool: menu_package.join(".build/artifacts/sparkle/Sparkle/bin/generate_appcast"),
        sparkle_cli,
    };
    for (tool_path, tool_description) in [
        (&sparkle_tools.sparkle_framework, "Sparkle framework"),
        (&sparkle_tools.appcast_tool, "Sparkle appcast tool"),
        (&sparkle_tools.sparkle_cli, "Sparkle command-line updater"),
    ] {
        if !tool_path.exists() {
            return Err(format!(
                "{tool_description} is unavailable: {}",
                tool_path.display()
            ));
        }
    }
    Ok(sparkle_tools)
}

fn start_loopback_server(
    repository_root: &Path,
    acceptance_run: &mut AcceptanceRun,
) -> Result<String, String> {
    let server_step = start_step("start-loopback-update-server");
    let port_file = acceptance_run.acceptance_root.join("server.port");
    let server_process = Command::new("python3")
        .arg(repository_root.join("scripts/release/fixtures/sparkle_update_loopback_server.py"))
        .arg("--directory")
        .arg(acceptance_run.acceptance_root.join("served"))
        .arg("--port-file")
        .arg(&port_file)
        .spawn();
    acceptance_run.server_process = match server_process {
        Ok(server_process) => Some(server_process),
        Err(spawn_error) => {
            return Err(format!(
                "the loopback server could not start: {spawn_error}"
            ));
        }
    };
    wait_for_file(&port_file, "loopback server port", acceptance_run.deadline)?;
    let server_port = std::fs::read_to_string(&port_file)
        .unwrap_or_default()
        .trim()
        .to_owned();
    server_step.finish("success");
    Ok(format!("http://127.0.0.1:{server_port}/appcast.xml"))
}

fn assemble_acceptance_apps(
    repository_root: &Path,
    acceptance_run: &AcceptanceRun,
    sparkle_tools: &SparkleTools,
    feed_url: &str,
) -> Result<(PathBuf, PathBuf), String> {
    let assemble_step = start_step("assemble-old-and-new-apps");
    let acceptance_executable = acceptance_run.acceptance_root.join("acceptance-app");
    run_checked(
        Command::new("clang")
            .args(["-fobjc-arc", "-Wall", "-Werror", "-framework", "AppKit"])
            .arg(repository_root.join("scripts/release/fixtures/sparkle-update-acceptance-app.m"))
            .arg("-o")
            .arg(&acceptance_executable),
        Some(acceptance_run.deadline),
    )?;
    let old_app = acceptance_run
        .acceptance_root
        .join("old")
        .join(ACCEPTANCE_APP_NAME);
    let new_app = acceptance_run
        .acceptance_root
        .join("new")
        .join(ACCEPTANCE_APP_NAME);
    for (app_bundle, bundle_version, display_version) in
        [(&old_app, "1", "1.0.0"), (&new_app, "2", "2.0.0")]
    {
        let plist_text = acceptance_info_plist(
            bundle_version,
            display_version,
            feed_url,
            &acceptance_run.acceptance_root.join("launch.log"),
            &acceptance_run.acceptance_root.join("app.pid"),
        );
        assemble_acceptance_app(
            app_bundle,
            &plist_text,
            &acceptance_executable,
            sparkle_tools,
            acceptance_run.deadline,
        )?;
    }
    run_checked(
        Command::new("hdiutil")
            .args([
                "create",
                "-volname",
                "Astronomical Update Acceptance",
                "-srcfolder",
            ])
            .arg(&new_app)
            .args(["-format", "UDZO"])
            .arg(
                acceptance_run
                    .acceptance_root
                    .join("served")
                    .join(UPDATE_IMAGE_NAME),
            ),
        Some(acceptance_run.deadline),
    )?;
    assemble_step.finish("success");
    Ok((old_app, new_app))
}

fn assemble_acceptance_app(
    app_bundle: &Path,
    plist_text: &str,
    acceptance_executable: &Path,
    sparkle_tools: &SparkleTools,
    deadline: Instant,
) -> Result<(), String> {
    create_directory(&app_bundle.join("Contents/MacOS"))?;
    create_directory(&app_bundle.join("Contents/Frameworks"))?;
    write_text(&app_bundle.join("Contents/Info.plist"), plist_text)?;
    if let Err(copy_error) = std::fs::copy(
        acceptance_executable,
        app_bundle.join("Contents/MacOS/acceptance-app"),
    ) {
        return Err(format!(
            "could not copy the acceptance executable: {copy_error}"
        ));
    }
    let framework_destination = app_bundle.join("Contents/Frameworks/Sparkle.framework");
    run_checked(
        Command::new("ditto")
            .arg(&sparkle_tools.sparkle_framework)
            .arg(&framework_destination),
        Some(deadline),
    )?;
    run_checked(
        Command::new("codesign")
            .args(["--force", "--sign", "-"])
            .arg(app_bundle.join("Contents/MacOS/acceptance-app")),
        Some(deadline),
    )?;
    run_checked(
        Command::new("codesign")
            .args(["--force", "--sign", "-", "--deep"])
            .arg(&framework_destination),
        Some(deadline),
    )?;
    run_checked(
        Command::new("codesign")
            .args(["--force", "--sign", "-"])
            .arg(app_bundle),
        Some(deadline),
    )?;
    run_checked(
        Command::new("codesign")
            .args(["--verify", "--deep", "--strict"])
            .arg(app_bundle),
        Some(deadline),
    )
}

/// The fixture app's plist: the product's Sparkle policy (signed feed,
/// verify before extraction) pointed at the loopback feed, plus the two
/// paths the fixture app writes its launches and process id to.
pub(crate) fn acceptance_info_plist(
    bundle_version: &str,
    display_version: &str,
    feed_url: &str,
    launch_log: &Path,
    process_identifier_file: &Path,
) -> String {
    let mut plist_text = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n",
    );
    for (plist_key, plist_value) in [
        ("CFBundleName", "Astronomical Update Acceptance"),
        ("CFBundleDisplayName", "Astronomical Update Acceptance"),
        ("CFBundleExecutable", "acceptance-app"),
        ("CFBundleIdentifier", ACCEPTANCE_BUNDLE_IDENTIFIER),
        ("CFBundlePackageType", "APPL"),
        ("CFBundleVersion", bundle_version),
        ("CFBundleShortVersionString", display_version),
        ("LSMinimumSystemVersion", "14.0"),
        ("SUFeedURL", feed_url),
        ("SUPublicEDKey", SPARKLE_PUBLIC_ED25519_KEY),
        (
            "AstronomicalAcceptanceLaunchLog",
            &launch_log.display().to_string(),
        ),
        (
            "AstronomicalAcceptancePIDFile",
            &process_identifier_file.display().to_string(),
        ),
    ] {
        plist_text.push_str(&format!(
            "  <key>{plist_key}</key><string>{}</string>\n",
            escape_xml_text(plist_value)
        ));
    }
    for (plist_key, is_enabled) in [
        ("LSUIElement", true),
        ("SUVerifyUpdateBeforeExtraction", true),
        ("SURequireSignedFeed", true),
        ("SUEnableAutomaticChecks", true),
        ("SUEnableSystemProfiling", false),
    ] {
        plist_text.push_str(&format!(
            "  <key>{plist_key}</key>{}\n",
            if is_enabled { "<true/>" } else { "<false/>" }
        ));
    }
    plist_text.push_str("  <key>SUSignedFeedFailureExpirationInterval</key><integer>0</integer>\n");
    plist_text.push_str("  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>\n");
    plist_text.push_str("</dict>\n</plist>\n");
    plist_text
}

fn generate_signed_appcast(
    acceptance_run: &AcceptanceRun,
    sparkle_tools: &SparkleTools,
    feed_url: &str,
) -> Result<(), String> {
    let appcast_step = start_step("generate-signed-local-appcast");
    let served_directory = acceptance_run.acceptance_root.join("served");
    let download_prefix = feed_url.trim_end_matches("appcast.xml");
    run_checked(
        Command::new(&sparkle_tools.appcast_tool)
            .args([
                "--download-url-prefix",
                download_prefix,
                "--maximum-deltas",
                "0",
                "-o",
            ])
            .arg(served_directory.join("appcast.xml"))
            .arg(&served_directory),
        Some(acceptance_run.deadline),
    )?;
    let appcast_text =
        std::fs::read_to_string(served_directory.join("appcast.xml")).unwrap_or_default();
    if !appcast_text.contains("sparkle:edSignature=")
        || !appcast_text.contains("sparkle-signatures:")
    {
        return Err("the local appcast is not signed".to_owned());
    }
    run_checked(
        Command::new("curl").args([
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            "5",
            feed_url,
        ]),
        Some(acceptance_run.deadline),
    )?;
    appcast_step.finish("success");
    Ok(())
}

/// Stand-ins for the config, models, cache, and logs that live outside the
/// app bundle; an update must never touch them.
fn seed_external_state(state_directory: &Path) -> Result<String, String> {
    for (marker_path, marker_text) in STATE_MARKER_PATHS.into_iter().zip([
        "{\"acceptance\":true}\n",
        "model artifact remains external\n",
        "cache remains external\n",
        "log remains external\n",
    ]) {
        let marker_file = state_directory.join(marker_path);
        if let Some(marker_parent) = marker_file.parent() {
            create_directory(marker_parent)?;
        }
        write_text(&marker_file, marker_text)?;
    }
    state_fingerprint(state_directory)
}

pub(crate) fn state_fingerprint(state_directory: &Path) -> Result<String, String> {
    let mut fingerprint_text = String::new();
    for marker_path in STATE_MARKER_PATHS {
        match std::fs::read(state_directory.join(marker_path)) {
            Ok(marker_bytes) => {
                fingerprint_text.push_str(&format!(
                    "{}  {marker_path}\n",
                    astronomical_native_identity::digest_bytes(&marker_bytes)
                ));
            }
            Err(_) => return Err(format!("external state marker is missing: {marker_path}")),
        }
    }
    Ok(fingerprint_text)
}

fn install_and_relaunch(
    acceptance_run: &mut AcceptanceRun,
    sparkle_tools: &SparkleTools,
    old_app: &Path,
    feed_url: &str,
    state_before: &str,
) -> Result<(), String> {
    let install_step = start_step("install-signed-update-and-relaunch");
    let installed_app = acceptance_run
        .acceptance_root
        .join("installed")
        .join(ACCEPTANCE_APP_NAME);
    run_checked(
        Command::new("ditto").arg(old_app).arg(&installed_app),
        Some(acceptance_run.deadline),
    )?;
    acceptance_run.app_process =
        match Command::new(installed_app.join("Contents/MacOS/acceptance-app")).spawn() {
            Ok(app_process) => Some(app_process),
            Err(spawn_error) => return Err(format!("the old app could not launch: {spawn_error}")),
        };
    let launch_log = acceptance_run.acceptance_root.join("launch.log");
    wait_for_file(
        &launch_log,
        "old application launch",
        acceptance_run.deadline,
    )?;
    run_checked(
        &mut sparkle_update_command(sparkle_tools, &installed_app, feed_url, true),
        Some(acceptance_run.deadline),
    )?;
    if installed_bundle_version(&installed_app)? != "2" {
        return Err("signed update did not install".to_owned());
    }
    wait_for_text(
        &launch_log,
        "2",
        "updated app relaunch",
        acceptance_run.deadline,
    )?;
    if state_fingerprint(&acceptance_run.acceptance_root.join("state"))? != state_before {
        return Err("the signed update changed external state".to_owned());
    }
    stop_relaunched_app(&acceptance_run.acceptance_root);
    install_step.finish("success");
    Ok(())
}

fn reject_corrupted_update(
    acceptance_run: &AcceptanceRun,
    sparkle_tools: &SparkleTools,
    old_app: &Path,
    feed_url: &str,
    state_before: &str,
) -> Result<(), String> {
    let rejection_step = start_step("reject-corrupted-update");
    let installed_app = acceptance_run
        .acceptance_root
        .join("installed")
        .join(ACCEPTANCE_APP_NAME);
    let _ = std::fs::remove_dir_all(&installed_app);
    run_checked(
        Command::new("ditto").arg(old_app).arg(&installed_app),
        Some(acceptance_run.deadline),
    )?;
    append_corruption(
        &acceptance_run
            .acceptance_root
            .join("served")
            .join(UPDATE_IMAGE_NAME),
    )?;
    if run_checked(
        &mut sparkle_update_command(sparkle_tools, &installed_app, feed_url, false),
        Some(acceptance_run.deadline),
    )
    .is_ok()
    {
        return Err("Sparkle accepted a corrupted update archive".to_owned());
    }
    if installed_bundle_version(&installed_app)? != "1" {
        return Err("corrupted update replaced the installed app".to_owned());
    }
    if state_fingerprint(&acceptance_run.acceptance_root.join("state"))? != state_before {
        return Err("the corrupted update changed external state".to_owned());
    }
    rejection_step.finish("success");
    Ok(())
}

/// Sparkle relaunches the updated app as a new process this run does not
/// own; the fixture app writes its pid, and only a process whose command is
/// that installed fixture executable is asked to stop.
fn stop_relaunched_app(acceptance_root: &Path) {
    let process_identifier =
        std::fs::read_to_string(acceptance_root.join("app.pid")).unwrap_or_default();
    let process_identifier = process_identifier.trim();
    if process_identifier.is_empty() || !process_identifier.bytes().all(is_ascii_digit_byte) {
        return;
    }
    let installed_executable = acceptance_root
        .join("installed")
        .join(ACCEPTANCE_APP_NAME)
        .join("Contents/MacOS/acceptance-app");
    let process_command = match Command::new("ps")
        .args(["-p", process_identifier, "-o", "command="])
        .output()
    {
        Ok(ps_output) => String::from_utf8_lossy(&ps_output.stdout).into_owned(),
        Err(_) => return,
    };
    if process_command.contains(&installed_executable.display().to_string()) {
        let _ = Command::new("kill")
            .args(["-TERM", process_identifier])
            .status();
    }
}

fn is_ascii_digit_byte(candidate_byte: u8) -> bool {
    candidate_byte.is_ascii_digit()
}

fn sparkle_update_command(
    sparkle_tools: &SparkleTools,
    installed_app: &Path,
    feed_url: &str,
    relaunches_app: bool,
) -> Command {
    let mut update_command = Command::new(&sparkle_tools.sparkle_cli);
    update_command.arg(installed_app);
    if relaunches_app {
        update_command.arg("--application").arg(installed_app);
    }
    update_command.args([
        "--check-immediately",
        "--feed-url",
        feed_url,
        "--user-agent-name",
        "Astronomical update acceptance",
        "--verbose",
    ]);
    update_command
}

fn append_corruption(update_image: &Path) -> Result<(), String> {
    use std::io::Write;
    let mut update_file = match std::fs::OpenOptions::new().append(true).open(update_image) {
        Ok(update_file) => update_file,
        Err(open_error) => return Err(format!("could not open the update image: {open_error}")),
    };
    match update_file.write_all(b"corruption") {
        Ok(()) => Ok(()),
        Err(write_error) => Err(format!("could not corrupt the update image: {write_error}")),
    }
}

fn installed_bundle_version(installed_app: &Path) -> Result<String, String> {
    let plist_output = Command::new("plutil")
        .args(["-extract", "CFBundleVersion", "raw", "-o", "-"])
        .arg(installed_app.join("Contents/Info.plist"))
        .output();
    match plist_output {
        Ok(plist_output) if plist_output.status.success() => {
            Ok(String::from_utf8_lossy(&plist_output.stdout)
                .trim()
                .to_owned())
        }
        _ => Err("the installed app version could not be read".to_owned()),
    }
}

/// Runs a tool to completion; with a deadline, the tool is stopped once the
/// acceptance budget runs out instead of hanging the run.
fn run_checked(tool_command: &mut Command, deadline: Option<Instant>) -> Result<(), String> {
    let tool_name = tool_command.get_program().to_string_lossy().into_owned();
    let mut tool_process = match tool_command.spawn() {
        Ok(tool_process) => tool_process,
        Err(spawn_error) => return Err(format!("{tool_name} could not start: {spawn_error}")),
    };
    loop {
        match tool_process.try_wait() {
            Ok(Some(exit_status)) if exit_status.success() => return Ok(()),
            Ok(Some(exit_status)) => return Err(format!("{tool_name} failed with {exit_status}")),
            Ok(None) => {}
            Err(wait_error) => return Err(format!("could not wait for {tool_name}: {wait_error}")),
        }
        if let Some(deadline) = deadline
            && Instant::now() >= deadline
        {
            let _ = tool_process.kill();
            return Err(format!(
                "{tool_name} exceeded the {}-second acceptance budget",
                ACCEPTANCE_BUDGET.as_secs()
            ));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

fn wait_for_file(
    expected_file: &Path,
    file_description: &str,
    deadline: Instant,
) -> Result<(), String> {
    wait_for_text(expected_file, "", file_description, deadline)
}

/// Waits until the file exists, is non-empty, and contains `expected_text`,
/// for at most thirty seconds and never past the acceptance deadline.
fn wait_for_text(
    expected_file: &Path,
    expected_text: &str,
    wait_description: &str,
    deadline: Instant,
) -> Result<(), String> {
    let wait_started_at = Instant::now();
    let mut last_progress_at = Instant::now();
    loop {
        let file_text = std::fs::read_to_string(expected_file).unwrap_or_default();
        if !file_text.is_empty() && file_text.contains(expected_text) {
            return Ok(());
        }
        if wait_started_at.elapsed() >= FILE_WAIT || Instant::now() >= deadline {
            return Err(format!("timed out waiting for {wait_description}"));
        }
        if last_progress_at.elapsed() >= Duration::from_secs(5) {
            log_line(&format!(
                "status=waiting description=\"{wait_description}\" elapsed_seconds={}",
                wait_started_at.elapsed().as_secs()
            ));
            last_progress_at = Instant::now();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

fn create_directory(directory_path: &Path) -> Result<(), String> {
    match std::fs::create_dir_all(directory_path) {
        Ok(()) => Ok(()),
        Err(create_error) => Err(format!(
            "could not create {}: {create_error}",
            directory_path.display()
        )),
    }
}

fn write_text(file_path: &Path, file_text: &str) -> Result<(), String> {
    match std::fs::write(file_path, file_text) {
        Ok(()) => Ok(()),
        Err(write_error) => Err(format!(
            "could not write {}: {write_error}",
            file_path.display()
        )),
    }
}
