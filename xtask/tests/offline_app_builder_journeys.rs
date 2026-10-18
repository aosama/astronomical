//! The macOS app-builder packaging journey as an offline integration test
//! (issue #1346): a sandboxed repository copy runs the real
//! `scripts/internal/build-macos-app.sh` with fixture-bin tool shims on PATH,
//! and the produced bundle is asserted through `plutil` extraction the same
//! way `validate-macos-app.sh` reads real bundles. Contracts the validator
//! already owns (URL scheme, icon key, build identity, post-signature
//! entitlements) live only there; this journey pins what the *builder* emits
//! and how it drives its tools.
//!
//! Shim `open` and `pgrep` keep the journey side-effect free: without them a
//! successful Development build would hand the fake bundle to the host's
//! Launch Services and probe the operator's real processes.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const SPARKLE_PUBLIC_ED25519_KEY: &str = "g0URwy+j86uDYcmOu0k/IUVWwCOSrGOPSoFnVoYQ9AQ=";
const EXPECTED_UPDATE_FEED_URL: &str = "https://example.github.io/astronomical/appcast.xml";
const APP_STORE_STATE_ROOT_FEATURE: &str = "--features astronomical-daemon/app-store-state-root";
const BUILDER_DEADLINE_SECONDS: u64 = 90;

struct AppBuilderSandbox {
    root_directory: PathBuf,
    repository_directory: PathBuf,
    shim_directory: PathBuf,
    lane_root_directory: PathBuf,
    cargo_arguments_log: PathBuf,
    swift_arguments_log: PathBuf,
    fixture_metallib_path: PathBuf,
}

struct BuilderJourney {
    channel: &'static str,
    signing_identity: Option<&'static str>,
    metallib_path: PathBuf,
    journey_log_stem: &'static str,
}

struct BuilderRunOutcome {
    succeeded: bool,
    standard_output_text: String,
    standard_error_text: String,
    codesign_log_path: PathBuf,
    cargo_target_record_path: PathBuf,
}

impl AppBuilderSandbox {
    fn install() -> Self {
        // Tests run in parallel threads inside one process, so the process id
        // alone cannot name the sandbox root: a sibling test's cleanup would
        // wipe this test's repository mid-journey.
        static SANDBOX_SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let sandbox_sequence = SANDBOX_SEQUENCE.fetch_add(1, Ordering::SeqCst);
        let root_directory = std::env::temp_dir().join(format!(
            "astronomical-app-builder-{}-{sandbox_sequence}",
            std::process::id()
        ));
        let repository_directory = root_directory.join("repository");
        let menu_build_directory = repository_directory.join("apps/astronomical-menu/.build");
        let sparkle_framework_directory = menu_build_directory.join("release/Sparkle.framework/Versions/B");
        let shim_directory = root_directory.join("shim-bin");
        let lane_root_directory = root_directory.join("cargo-lanes");
        for directory in [
            &shim_directory,
            &lane_root_directory,
            &repository_directory.join("scripts/internal/entitlements"),
            &repository_directory.join("scripts/ci"),
            &sparkle_framework_directory.join("Updater.app"),
            &sparkle_framework_directory.join("XPCServices/Downloader.xpc"),
            &sparkle_framework_directory.join("XPCServices/Installer.xpc"),
            &menu_build_directory.join("checkouts/Sparkle"),
            &repository_directory.join("apps/thin-talk/web"),
            &repository_directory.join("third-party"),
        ] {
            std::fs::create_dir_all(directory).expect("the sandbox directory layout should be creatable");
        }

        let repository_source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for (source_relative_path, sandbox_relative_path, make_executable) in [
            ("scripts/internal/build-macos-app.sh", "scripts/internal/build-macos-app.sh", true),
            ("scripts/internal/package-mlx-metallib.sh", "scripts/internal/package-mlx-metallib.sh", false),
            ("scripts/run-in-disposable-cargo-target.sh", "scripts/run-in-disposable-cargo-target.sh", true),
        ] {
            let source_path = repository_source_root.join(source_relative_path);
            let destination_path = repository_directory.join(sandbox_relative_path);
            std::fs::copy(&source_path, &destination_path)
                .unwrap_or_else(|error| panic!("{} should copy into the sandbox: {error}", source_path.display()));
            make_executable_if_requested(&destination_path, make_executable);
        }
        for entitlements_entry in std::fs::read_dir(repository_source_root.join("scripts/internal/entitlements"))
            .expect("the entitlements directory should list its files")
        {
            let entitlements_entry = entitlements_entry.expect("the entitlements entry should be readable");
            let entitlements_file_name = entitlements_entry.file_name();
            if entitlements_file_name.to_string_lossy().ends_with(".entitlements") {
                std::fs::copy(
                    entitlements_entry.path(),
                    repository_directory.join("scripts/internal/entitlements").join(&entitlements_file_name),
                )
                .expect("the entitlements fixture should copy into the sandbox");
            }
        }

        for fixture_relative_path in [
            "LICENSE",
            "third-party/THIRD_PARTY_NOTICES",
            "third-party/RUST_DEPENDENCY_NOTICES",
            "apps/astronomical-menu/.build/checkouts/Sparkle/LICENSE",
            "apps/astronomical-menu/.build/release/Sparkle.framework/Versions/B/Autoupdate",
        ] {
            std::fs::write(repository_directory.join(fixture_relative_path), "fixture\n")
                .expect("the sandbox fixture file should be writable");
        }

        let fixture_metallib_path = root_directory.join("mlx.metallib");
        std::fs::write(&fixture_metallib_path, "fixture-metallib\n")
            .expect("the fixture metallib should be writable");
        let cargo_arguments_log = root_directory.join("fake-cargo-args.log");
        std::fs::write(&cargo_arguments_log, "").expect("the cargo arguments log should start empty");
        let swift_arguments_log = root_directory.join("fake-swift-args.log");
        std::fs::write(&swift_arguments_log, "").expect("the swift arguments log should start empty");

        let sandbox = Self {
            root_directory,
            repository_directory,
            shim_directory,
            lane_root_directory,
            cargo_arguments_log,
            swift_arguments_log,
            fixture_metallib_path,
        };
        sandbox.write_tool_shims();
        sandbox
    }

    fn write_tool_shims(&self) {
        // The fake cargo answers metadata, the xtask port, and the release
        // build: it records the disposable CARGO_TARGET_DIR the runner owns
        // and materializes stub release binaries inside it, so the builder's
        // copy steps find exactly what a real build would have produced.
        self.write_shim(
            "cargo",
            &format!(
                "#!/bin/sh\n\
                 printf '%s\\n' \"$*\" >> \"{cargo_log}\"\n\
                 if [ \"${{1:-}}\" = \"metadata\" ]; then\n\
                 \x20   printf '%s\\n' '{{\"packages\":[{{\"name\":\"astronomical-daemon\",\"version\":\"0.2.0\",\"repository\":\"https://github.com/example/astronomical\"}}]}}'\n\
                 \x20   exit 0\n\
                 fi\n\
                 if [ \"${{1:-}}\" = \"xtask\" ]; then\n\
                 \x20   exit 0\n\
                 fi\n\
                 cargo_target_directory=\"${{CARGO_TARGET_DIR:?disposable Cargo target is required}}\"\n\
                 printf '%s\\n' \"$cargo_target_directory\" > \"${{FAKE_CARGO_TARGET_RECORD:?fake cargo target record is required}}\"\n\
                 target_triple=\"\"\n\
                 while [ \"$#\" -gt 0 ]; do\n\
                 \x20   if [ \"$1\" = \"--target\" ]; then target_triple=\"$2\"; break; fi\n\
                 \x20   shift\n\
                 done\n\
                 [ -n \"$target_triple\" ] || exit 1\n\
                 cargo_release_directory=\"${{cargo_target_directory}}/${{target_triple}}/release\"\n\
                 mkdir -p \"$cargo_release_directory\"\n\
                 printf '%s\\n' '#!/bin/sh' 'exit 0' > \"${{cargo_release_directory}}/astronomicald\"\n\
                 cp \"${{cargo_release_directory}}/astronomicald\" \"${{cargo_release_directory}}/astronomical-inference-worker\"\n\
                 printf '%s\\n' '#!/bin/sh' '[ \"$1\" = \"--version\" ] && {{ printf \"%s\\n\" \"0.0.0\"; exit 0; }}' 'exit 0' > \"${{cargo_release_directory}}/astronomical\"\n\
                 chmod +x \"${{cargo_release_directory}}/astronomicald\" \"${{cargo_release_directory}}/astronomical-inference-worker\" \"${{cargo_release_directory}}/astronomical\"\n",
                cargo_log = self.cargo_arguments_log.display(),
            ),
        );
        self.write_shim(
            "rustc",
            "#!/bin/sh\n[ \"$*\" = \"--print host-tuple\" ] || exit 1\nprintf '%s\\n' 'fixture-apple-darwin'\n",
        );
        // The fake swift serves both invocation shapes: the icon renderer
        // (anything that is not `swift build`) emits the channel-named
        // iconset the fake iconutil converts, and `swift build` materializes
        // the menu product plus the Thin Talk canvas bundle the builder
        // copies. Arguments are logged so journeys can pin which SwiftPM
        // product each channel builds — the actual mechanism that excludes
        // Sparkle from the App Store menu binary.
        self.write_shim(
            "swift",
            &format!(
                "#!/bin/sh\n\
                 printf '%s\\n' \"$*\" >> \"{swift_log}\"\n\
                 if [ \"${{1:-}}\" != \"build\" ]; then\n\
                 \x20   iconset_directory=\"\"\n\
                 \x20   icon_channel=\"\"\n\
                 \x20   while [ \"$#\" -gt 0 ]; do\n\
                 \x20       if [ \"$1\" = \"--output-directory\" ]; then iconset_directory=\"$2\"; shift 2; continue; fi\n\
                 \x20       if [ \"$1\" = \"--channel\" ]; then icon_channel=\"$2\"; shift 2; continue; fi\n\
                 \x20       shift\n\
                 \x20   done\n\
                 \x20   mkdir -p \"${{iconset_directory:?iconset directory is required}}\"\n\
                 \x20   case \"$icon_channel\" in stable|development) ;; *) exit 1 ;; esac\n\
                 \x20   printf '%s\\n' \"$icon_channel\" > \"${{iconset_directory}}/.channel\"\n\
                 \x20   for icon_name in icon_16x16.png icon_16x16@2x.png icon_32x32.png icon_32x32@2x.png \\
                 \x20       icon_128x128.png icon_128x128@2x.png icon_256x256.png icon_256x256@2x.png \\
                 \x20       icon_512x512.png icon_512x512@2x.png; do\n\
                 \x20       printf '%s\\n' fixture > \"${{iconset_directory}}/${{icon_name}}\"\n\
                 \x20   done\n\
                 \x20   exit 0\n\
                 fi\n\
                 package_path=\"\"\n\
                 menu_product=\"astronomical-menu\"\n\
                 while [ \"$#\" -gt 0 ]; do\n\
                 \x20   if [ \"$1\" = \"--package-path\" ]; then package_path=\"$2\"; shift 2; continue; fi\n\
                 \x20   if [ \"$1\" = \"--product\" ]; then menu_product=\"$2\"; shift 2; continue; fi\n\
                 \x20   shift\n\
                 done\n\
                 mkdir -p \"${{package_path:?package path is required}}/.build/release\"\n\
                 printf '%s\\n' '#!/bin/sh' 'exit 0' > \"${{package_path}}/.build/release/${{menu_product}}\"\n\
                 chmod +x \"${{package_path}}/.build/release/${{menu_product}}\"\n\
                 canvas_resource_bundle=\"${{package_path}}/.build/out/Products/Release/ThinTalk_ThinTalkCanvas.bundle\"\n\
                 mkdir -p \"$canvas_resource_bundle\"\n\
                 printf '%s\\n' fixture > \"${{canvas_resource_bundle}}/fixture\"\n",
                swift_log = self.swift_arguments_log.display(),
            ),
        );
        self.write_shim(
            "iconutil",
            "#!/bin/sh\n\
             output_path=\"\"\n\
             while [ \"$#\" -gt 0 ]; do\n\
             \x20   if [ \"$1\" = \"--output\" ]; then output_path=\"$2\"; break; fi\n\
             \x20   shift\n\
             done\n\
             for command_argument in \"$@\"; do iconset_directory=\"$command_argument\"; done\n\
             cp \"${iconset_directory}/.channel\" \"${output_path:?icon output path is required}\"\n",
        );
        self.write_shim(
            "git",
            "#!/bin/sh\ncase \"${1:-}\" in\n\
             \x20   rev-parse) printf '%s\\n' abcdef123456 ;;\n\
             \x20   rev-list) printf '%s\\n' 107 ;;\n\
             \x20   status) exit 0 ;;\n\
             \x20   *) exit 1 ;;\n\
             esac\n",
        );
        self.write_shim("sysctl", "#!/bin/sh\nprintf '%s\\n' 4\n");
        self.write_shim(
            "jq",
            "#!/bin/sh\ncase \"$*\" in\n\
             \x20   *'.version'*) printf '%s\\n' '0.2.0' ;;\n\
             \x20   *'.repository'*) printf '%s\\n' 'https://github.com/example/astronomical' ;;\n\
             \x20   *) exit 1 ;;\n\
             esac\n",
        );
        self.write_shim(
            "codesign",
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"${FAKE_CODESIGN_LOG:?}\"\nexit 0\n",
        );
        for benign_command in ["npm", "cmake", "xcrun", "install_name_tool", "plutil", "open", "pgrep"] {
            self.write_shim(benign_command, "#!/bin/sh\nexit 0\n");
        }
        // The builder invokes two repository scripts by path; the sandbox
        // must provide stubs so the journey never touches the machine-wide
        // native store or launches the produced bundle for validation.
        self.write_shim_at(
            &self.repository_directory.join("scripts/ci/prewarm-native-build.sh"),
            "#!/bin/sh\nexit 0\n",
        );
        self.write_shim_at(
            &self.repository_directory.join("scripts/internal/validate-macos-app.sh"),
            "#!/bin/sh\nexit 0\n",
        );
    }

    fn write_shim(&self, command_name: &str, shim_body: &str) {
        self.write_shim_at(&self.shim_directory.join(command_name), shim_body);
    }

    fn write_shim_at(&self, shim_path: &Path, shim_body: &str) {
        std::fs::write(shim_path, shim_body).expect("the tool shim should be writable");
        make_executable_if_requested(shim_path, true);
    }

    fn run_builder(&self, journey: &BuilderJourney) -> BuilderRunOutcome {
        let codesign_log_path = self.root_directory.join(format!("{}-codesign.log", journey.journey_log_stem));
        let cargo_target_record_path =
            self.root_directory.join(format!("{}-cargo-target", journey.journey_log_stem));
        let standard_output_path = self.root_directory.join(format!("{}-stdout.log", journey.journey_log_stem));
        let standard_error_path = self.root_directory.join(format!("{}-stderr.log", journey.journey_log_stem));
        let standard_output_file = std::fs::File::create(&standard_output_path)
            .expect("the builder stdout log should be creatable");
        let standard_error_file = std::fs::File::create(&standard_error_path)
            .expect("the builder stderr log should be creatable");
        let mut builder_command = Command::new(
            self.repository_directory.join("scripts/internal/build-macos-app.sh"),
        );
        builder_command
            .arg("--channel")
            .arg(journey.channel)
            .current_dir(&self.repository_directory)
            .env("PATH", format!("{}:{}", self.shim_directory.display(), std::env::var("PATH").unwrap_or_default()))
            .env("FAKE_CARGO_ARGS_LOG", &self.cargo_arguments_log)
            .env("FAKE_SWIFT_ARGS_LOG", &self.swift_arguments_log)
            .env("FAKE_CODESIGN_LOG", &codesign_log_path)
            .env("FAKE_CARGO_TARGET_RECORD", &cargo_target_record_path)
            .env("ASTRONOMICAL_CARGO_LANE_ROOT", &self.lane_root_directory)
            .env("ASTRONOMICAL_MLX_METALLIB_PATH", &journey.metallib_path)
            .stdout(Stdio::from(standard_output_file))
            .stderr(Stdio::from(standard_error_file));
        if let Some(signing_identity) = journey.signing_identity {
            builder_command.arg("--signing-identity").arg(signing_identity);
        }

        // The builder is a real subprocess with no built-in deadline; a shim
        // bug can hang it forever, so the wait is bounded in-test and the
        // process group is killed on overrun instead of stalling the lane.
        let mut builder_process = builder_command
            .spawn()
            .expect("the sandboxed builder should spawn");
        let wait_deadline = Instant::now() + Duration::from_secs(BUILDER_DEADLINE_SECONDS);
        let exit_status = loop {
            match builder_process.try_wait().expect("the builder process should be observable") {
                Some(exit_status) => break exit_status,
                None => {
                    if Instant::now() > wait_deadline {
                        let _ = builder_process.kill();
                        panic!(
                            "the sandboxed {} builder run exceeded {BUILDER_DEADLINE_SECONDS}s",
                            journey.channel
                        );
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        };

        BuilderRunOutcome {
            succeeded: exit_status.success(),
            standard_output_text: file_text(&standard_output_path),
            standard_error_text: file_text(&standard_error_path),
            codesign_log_path,
            cargo_target_record_path,
        }
    }

    fn development_bundle_path(&self) -> PathBuf {
        self.repository_directory
            .join("target/astronomical-macos-development.noindex/Astronomical Development.app")
    }

    fn stable_bundle_path(&self) -> PathBuf {
        self.repository_directory
            .join("target/astronomical-macos-stable.noindex/Astronomical.app")
    }

    fn app_store_bundle_path(&self) -> PathBuf {
        self.repository_directory
            .join("target/astronomical-macos-app-store.noindex/Astronomical.app")
    }

    fn cargo_arguments_text(&self) -> String {
        file_text(&self.cargo_arguments_log)
    }

    fn swift_arguments_text(&self) -> String {
        file_text(&self.swift_arguments_log)
    }
}

impl Drop for AppBuilderSandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root_directory);
    }
}

fn make_executable_if_requested(path: &Path, make_executable: bool) {
    if !make_executable {
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .unwrap_or_else(|error| panic!("{} should become executable: {error}", path.display()));
    }
    #[cfg(not(unix))]
    let _ = path;
}

fn file_text(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()))
}

/// Reads one Info.plist value the way the real validator does: plutil
/// extraction by key, with absence expressed as the extraction failing.
/// Absolute-path invocation bypasses the shimmed plutil on PATH.
fn plist_string(bundle_path: &Path, plist_key: &str) -> Option<String> {
    let extraction_output = Command::new("/usr/bin/plutil")
        .args(["-extract", plist_key, "raw", "-o", "-"])
        .arg(bundle_path.join("Contents/Info.plist"))
        .output()
        .expect("plutil should run in the test process");
    if !extraction_output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&extraction_output.stdout).trim().to_owned())
}

fn resource_text(bundle_path: &Path, resource_relative_path: &str) -> String {
    file_text(&bundle_path.join("Contents/Resources").join(resource_relative_path))
        .trim()
        .to_owned()
}

fn log_line_count_containing(log_text: &str, needle: &str) -> usize {
    log_text.lines().filter(|log_line| log_line.contains(needle)).count()
}

fn stable_development_journey(sandbox: &AppBuilderSandbox) -> BuilderJourney {
    BuilderJourney {
        channel: "development",
        signing_identity: None,
        metallib_path: sandbox.fixture_metallib_path.clone(),
        journey_log_stem: "preparatory-development",
    }
}

#[test]
fn missing_metallib_source_is_rejected() {
    let sandbox = AppBuilderSandbox::install();
    let outcome = sandbox.run_builder(&BuilderJourney {
        channel: "development",
        signing_identity: None,
        metallib_path: sandbox.root_directory.join("absent.metallib"),
        journey_log_stem: "missing-metallib",
    });

    assert!(!outcome.succeeded, "the builder accepted a missing mlx.metallib source");
    assert!(
        outcome.standard_error_text.contains("required MLX AOT metallib is unavailable"),
        "the builder did not report the missing mlx.metallib source; stderr:\n{}",
        outcome.standard_error_text
    );
}

#[test]
fn development_bundle_carries_channel_policy_without_distribution_signing() {
    let sandbox = AppBuilderSandbox::install();
    let outcome = sandbox.run_builder(&BuilderJourney {
        channel: "development",
        signing_identity: None,
        metallib_path: sandbox.fixture_metallib_path.clone(),
        journey_log_stem: "development",
    });

    assert!(
        outcome.succeeded,
        "the development journey failed; stderr:\n{}",
        outcome.standard_error_text
    );
    assert!(
        outcome.standard_output_text.contains("Astronomical development app ready"),
        "the development journey did not reach its success summary; stdout:\n{}",
        outcome.standard_output_text
    );
    let development_bundle = sandbox.development_bundle_path();
    assert!(
        plist_string(&development_bundle, "AstronomicalChannel").as_deref() == Some("development"),
        "the development bundle did not declare its channel"
    );
    assert!(
        plist_string(&development_bundle, "CFBundleIdentifier").as_deref()
            == Some("dev.astronomical.app.development"),
        "the development bundle did not carry its development identifier"
    );
    assert!(
        plist_string(&development_bundle, "AstronomicalStateDirectoryName").as_deref()
            == Some(".astronomical-dev"),
        "the development bundle did not carry its isolated state directory name"
    );
    assert!(
        plist_string(&development_bundle, "SUFeedURL").as_deref() == Some(EXPECTED_UPDATE_FEED_URL),
        "the development bundle did not carry the GitHub Pages update feed"
    );
    assert!(
        plist_string(&development_bundle, "SUPublicEDKey").as_deref() == Some(SPARKLE_PUBLIC_ED25519_KEY),
        "the development bundle did not pin the Sparkle public key"
    );
    assert_eq!(
        resource_text(&development_bundle, "share/mlx/mlx.metallib"),
        "fixture-metallib",
        "the packaged metallib is not the build-produced copy"
    );
    assert_eq!(
        resource_text(&development_bundle, "Astronomical.icns"),
        "development",
        "the development bundle did not request the Development icon identity"
    );
    let development_codesign_text = file_text(&outcome.codesign_log_path);
    assert!(
        !development_codesign_text.contains("--options runtime"),
        "the development bundle unexpectedly requested distribution signing"
    );
    assert!(
        !sandbox.cargo_arguments_text().contains(APP_STORE_STATE_ROOT_FEATURE),
        "the development build must not enable the App Store state-root feature"
    );
    assert!(
        sandbox.swift_arguments_text().contains("--product astronomical-menu"),
        "the development build did not build the standard menu product"
    );
    let recorded_cargo_target = file_text(&outcome.cargo_target_record_path);
    assert!(
        !Path::new(&recorded_cargo_target).exists(),
        "the development build retained its disposable Cargo target"
    );
}

#[test]
fn stable_bundle_signs_for_distribution_and_keeps_the_development_artifact() {
    let sandbox = AppBuilderSandbox::install();
    let preparatory_development_outcome = sandbox.run_builder(&stable_development_journey(&sandbox));
    assert!(
        preparatory_development_outcome.succeeded,
        "the preparatory development journey failed; stderr:\n{}",
        preparatory_development_outcome.standard_error_text
    );
    let outcome = sandbox.run_builder(&BuilderJourney {
        channel: "stable",
        signing_identity: Some("Developer ID Application: Example (ABCDE12345)"),
        metallib_path: sandbox.fixture_metallib_path.clone(),
        journey_log_stem: "stable",
    });

    assert!(
        outcome.succeeded,
        "the stable journey failed; stderr:\n{}",
        outcome.standard_error_text
    );
    assert!(
        outcome.standard_output_text.contains("Astronomical stable app ready"),
        "the stable journey did not reach its success summary; stdout:\n{}",
        outcome.standard_output_text
    );
    let stable_bundle = sandbox.stable_bundle_path();
    assert_eq!(
        resource_text(&stable_bundle, "Astronomical.icns"),
        "stable",
        "the stable bundle did not request the Stable icon identity"
    );
    assert!(
        plist_string(&stable_bundle, "SUFeedURL").as_deref() == Some(EXPECTED_UPDATE_FEED_URL),
        "the stable bundle did not carry the GitHub Pages update feed"
    );
    assert!(
        sandbox.development_bundle_path().exists(),
        "building stable unexpectedly removed the development build artifact"
    );
    let stable_codesign_text = file_text(&outcome.codesign_log_path);
    assert!(
        stable_codesign_text
            .contains("--sign Developer ID Application: Example (ABCDE12345) --options runtime --timestamp"),
        "the stable bundle was not signed with Developer ID, hardened runtime, and timestamping"
    );
    assert!(
        !sandbox.cargo_arguments_text().contains(APP_STORE_STATE_ROOT_FEATURE),
        "the stable build must not enable the App Store state-root feature"
    );
    let recorded_cargo_target = file_text(&outcome.cargo_target_record_path);
    assert!(
        !Path::new(&recorded_cargo_target).exists(),
        "the stable build retained its disposable Cargo target"
    );
    let lane_root_residue: Vec<_> = std::fs::read_dir(&sandbox.lane_root_directory)
        .expect("the cargo lane root should remain listable")
        .collect();
    assert!(
        lane_root_residue.is_empty(),
        "app builds retained files beneath the disposable Cargo lane root"
    );
}

#[test]
fn app_store_bundle_is_sparkle_free_and_assigns_sandbox_entitlements() {
    let sandbox = AppBuilderSandbox::install();
    let outcome = sandbox.run_builder(&BuilderJourney {
        channel: "app-store",
        signing_identity: Some("Apple Distribution: Example (ABCDE12345)"),
        metallib_path: sandbox.fixture_metallib_path.clone(),
        journey_log_stem: "app-store",
    });

    assert!(
        outcome.succeeded,
        "the app-store journey failed; stderr:\n{}",
        outcome.standard_error_text
    );
    assert!(
        outcome.standard_output_text.contains("Astronomical app-store app ready"),
        "the app-store journey did not reach its success summary; stdout:\n{}",
        outcome.standard_output_text
    );
    let app_store_bundle = sandbox.app_store_bundle_path();
    assert!(
        plist_string(&app_store_bundle, "AstronomicalChannel").as_deref() == Some("app-store"),
        "the app-store bundle did not declare its channel"
    );
    assert!(
        plist_string(&app_store_bundle, "CFBundleIdentifier").as_deref()
            == Some("app.astronomical.appstore"),
        "the app-store bundle did not carry the app-store bundle identifier"
    );
    for absent_plist_key in [
        "SUFeedURL",
        "SUPublicEDKey",
        "SUEnableAutomaticChecks",
        "AstronomicalStateDirectoryName",
    ] {
        assert!(
            plist_string(&app_store_bundle, absent_plist_key).is_none(),
            "the app-store bundle unexpectedly embeds {absent_plist_key}"
        );
    }
    assert!(
        !app_store_bundle.join("Contents/Frameworks/Sparkle.framework").exists(),
        "the app-store bundle unexpectedly embeds the Sparkle framework"
    );
    assert!(
        !app_store_bundle.join("Contents/Resources/SPARKLE_LICENSE").exists(),
        "the app-store bundle unexpectedly carries the Sparkle license"
    );
    // The App Store menu binary excludes Sparkle by building a distinct
    // SwiftPM product; the fake toolchain cannot link-check a stub, so the
    // product request is the observable contract here (the real validator
    // owns the post-build entitlement and linkage checks on real bundles).
    assert!(
        sandbox
            .swift_arguments_text()
            .contains("--product astronomical-menu-app-store"),
        "the app-store build did not build the Sparkle-free menu product"
    );
    assert_eq!(
        resource_text(&app_store_bundle, "Astronomical.icns"),
        "stable",
        "the app-store bundle did not request the Stable icon identity"
    );

    // The fake codesign only records its arguments, so the entitlement
    // contract is asserted from the signing log: the main executable and the
    // bundle seal carry the sandbox profile; helpers join it through
    // inheritance.
    let app_store_codesign_text = file_text(&outcome.codesign_log_path);
    for (entitlements_file_name, code_object_name) in [
        ("app-store-application.entitlements", "astronomical-menu"),
        ("app-store-inherited.entitlements", "astronomicald"),
        ("app-store-inherited.entitlements", "astronomical-inference-worker"),
        ("app-store-application.entitlements", "Astronomical.app"),
    ] {
        assert!(
            app_store_codesign_text
                .lines()
                .any(|codesign_line| codesign_line.contains(entitlements_file_name)
                    && codesign_line.contains(code_object_name)),
            "the app-store {code_object_name} was not signed with {entitlements_file_name}"
        );
    }
    // The CLI runs from the user's terminal, so it must be signed bare: no
    // entitlements profile of any kind may ride on it. The trailing-object
    // match keeps the CLI distinct from astronomicald and the bundle seal.
    let cli_entitlements_line_count = app_store_codesign_text
        .lines()
        .filter(|codesign_line| {
            codesign_line.contains("--entitlements") && codesign_line.trim_end().ends_with("MacOS/astronomical")
        })
        .count();
    assert_eq!(
        cli_entitlements_line_count, 0,
        "the app-store CLI was signed with an entitlements profile"
    );

    let cargo_arguments_text = sandbox.cargo_arguments_text();
    assert_eq!(
        log_line_count_containing(&cargo_arguments_text, APP_STORE_STATE_ROOT_FEATURE),
        1,
        "the app-store state-root feature must be enabled exactly once, for the app-store build only"
    );
    assert!(
        cargo_arguments_text.contains("bootstrap-native-dependencies --verify"),
        "the builder did not verify the native dependency archives through xtask"
    );
    let recorded_cargo_target = file_text(&outcome.cargo_target_record_path);
    assert!(
        !Path::new(&recorded_cargo_target).exists(),
        "the app-store build retained its disposable Cargo target"
    );
}
