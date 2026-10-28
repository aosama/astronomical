//! The sandboxed repository copy that runs the real
//! `scripts/internal/build-macos-app.sh` against fixture-bin tool shims, so
//! the journey is deterministic and side-effect free.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::journey::{BuilderJourney, BuilderRunOutcome, file_text};

const BUILDER_DEADLINE_SECONDS: u64 = 90;

pub(crate) struct AppBuilderSandbox {
    pub(crate) root_directory: PathBuf,
    repository_directory: PathBuf,
    shim_directory: PathBuf,
    pub(crate) lane_root_directory: PathBuf,
    cargo_arguments_log: PathBuf,
    swift_arguments_log: PathBuf,
    pub(crate) fixture_metallib_path: PathBuf,
}

impl AppBuilderSandbox {
    #[expect(clippy::too_many_lines)]
    pub(crate) fn install() -> Self {
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
        let sparkle_framework_directory =
            menu_build_directory.join("release/Sparkle.framework/Versions/B");
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
            std::fs::create_dir_all(directory)
                .expect("the sandbox directory layout should be creatable");
        }

        let repository_source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for (source_relative_path, sandbox_relative_path, make_executable) in [
            (
                "scripts/internal/build-macos-app.sh",
                "scripts/internal/build-macos-app.sh",
                true,
            ),
            (
                "scripts/internal/package-mlx-metallib.sh",
                "scripts/internal/package-mlx-metallib.sh",
                false,
            ),
        ] {
            let source_path = repository_source_root.join(source_relative_path);
            let destination_path = repository_directory.join(sandbox_relative_path);
            std::fs::copy(&source_path, &destination_path).unwrap_or_else(|error| {
                panic!(
                    "{} should copy into the sandbox: {error}",
                    source_path.display()
                )
            });
            make_executable_if_requested(&destination_path, make_executable);
        }
        for entitlements_entry in
            std::fs::read_dir(repository_source_root.join("scripts/internal/entitlements"))
                .expect("the entitlements directory should list its files")
        {
            let entitlements_entry =
                entitlements_entry.expect("the entitlements entry should be readable");
            let entitlements_file_name = entitlements_entry.file_name();
            if entitlements_file_name
                .to_string_lossy()
                .ends_with(".entitlements")
            {
                std::fs::copy(
                    entitlements_entry.path(),
                    repository_directory
                        .join("scripts/internal/entitlements")
                        .join(&entitlements_file_name),
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
            std::fs::write(
                repository_directory.join(fixture_relative_path),
                "fixture\n",
            )
            .expect("the sandbox fixture file should be writable");
        }

        let fixture_metallib_path = root_directory.join("mlx.metallib");
        std::fs::write(&fixture_metallib_path, "fixture-metallib\n")
            .expect("the fixture metallib should be writable");
        let cargo_arguments_log = root_directory.join("fake-cargo-args.log");
        std::fs::write(&cargo_arguments_log, "")
            .expect("the cargo arguments log should start empty");
        let swift_arguments_log = root_directory.join("fake-swift-args.log");
        std::fs::write(&swift_arguments_log, "")
            .expect("the swift arguments log should start empty");

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

    #[expect(clippy::too_many_lines)]
    fn write_tool_shims(&self) {
        // The fake cargo answers metadata, the xtask port, and the release
        // build: it records the disposable CARGO_TARGET_DIR the runner owns
        // and materializes stub release binaries inside it, so the builder's
        // copy steps find exactly what a real build would have produced.
        // `cargo xtask disposable-target` re-execs the real runner binary
        // pinned to the sandbox repository, so the journey exercises the
        // genuine lifecycle (marker, handoff, measurement, removal) instead
        // of a shell emulation of it.
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
                 \x20   if [ \"${{2:-}}\" = \"disposable-target\" ]; then\n\
                 \x20       shift\n\
                 \x20       exec \"{xtask_binary}\" --repository-root \"{sandbox_repository}\" \"$@\"\n\
                 \x20   fi\n\
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
                xtask_binary = env!("CARGO_BIN_EXE_xtask"),
                sandbox_repository = self.repository_directory.display(),
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
                 \x20   for icon_name in icon_16x16.png icon_16x16@2x.png icon_32x32.png icon_32x32@2x.png \\\n\
                 \x20       icon_128x128.png icon_128x128@2x.png icon_256x256.png icon_256x256@2x.png \\\n\
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
        for benign_command in [
            "npm",
            "cmake",
            "xcrun",
            "install_name_tool",
            "plutil",
            // Shimmed open and pgrep keep the journey side-effect free: a
            // successful Development build would otherwise hand the fake
            // bundle to the host's Launch Services, and the builder's
            // post-build note would probe the operator's real processes.
            "open",
            "pgrep",
        ] {
            self.write_shim(benign_command, "#!/bin/sh\nexit 0\n");
        }
        // The builder invokes two repository scripts by path; the sandbox
        // must provide stubs so the journey never touches the machine-wide
        // native store or launches the produced bundle for validation.
        self.write_shim_at(
            &self
                .repository_directory
                .join("scripts/ci/prewarm-native-build.sh"),
            "#!/bin/sh\nexit 0\n",
        );
        self.write_shim_at(
            &self
                .repository_directory
                .join("scripts/internal/validate-macos-app.sh"),
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

    pub(crate) fn run_builder(&self, journey: &BuilderJourney) -> BuilderRunOutcome {
        let codesign_log_path = self
            .root_directory
            .join(format!("{}-codesign.log", journey.journey_log_stem));
        let cargo_target_record_path = self
            .root_directory
            .join(format!("{}-cargo-target", journey.journey_log_stem));
        let standard_output_path = self
            .root_directory
            .join(format!("{}-stdout.log", journey.journey_log_stem));
        let standard_error_path = self
            .root_directory
            .join(format!("{}-stderr.log", journey.journey_log_stem));
        let standard_output_file = std::fs::File::create(&standard_output_path)
            .expect("the builder stdout log should be creatable");
        let standard_error_file = std::fs::File::create(&standard_error_path)
            .expect("the builder stderr log should be creatable");
        let mut builder_command = Command::new(
            self.repository_directory
                .join("scripts/internal/build-macos-app.sh"),
        );
        builder_command
            .arg("--channel")
            .arg(journey.channel)
            .current_dir(&self.repository_directory)
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    self.shim_directory.display(),
                    std::env::var("PATH").unwrap_or_default()
                ),
            )
            .env("FAKE_CARGO_ARGS_LOG", &self.cargo_arguments_log)
            .env("FAKE_SWIFT_ARGS_LOG", &self.swift_arguments_log)
            .env("FAKE_CODESIGN_LOG", &codesign_log_path)
            .env("FAKE_CARGO_TARGET_RECORD", &cargo_target_record_path)
            .env("ASTRONOMICAL_CARGO_LANE_ROOT", &self.lane_root_directory)
            .env("ASTRONOMICAL_MLX_METALLIB_PATH", &journey.metallib_path)
            .stdout(Stdio::from(standard_output_file))
            .stderr(Stdio::from(standard_error_file));
        if let Some(signing_identity) = journey.signing_identity {
            builder_command
                .arg("--signing-identity")
                .arg(signing_identity);
        }

        // The builder is a real subprocess with no built-in deadline; a shim
        // bug can hang it forever, so the wait is bounded in-test and the
        // process is killed on overrun instead of stalling the lane.
        let mut builder_process = builder_command
            .spawn()
            .expect("the sandboxed builder should spawn");
        let wait_deadline = Instant::now() + Duration::from_secs(BUILDER_DEADLINE_SECONDS);
        let exit_status = loop {
            match builder_process
                .try_wait()
                .expect("the builder process should be observable")
            {
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

    pub(crate) fn development_bundle_path(&self) -> PathBuf {
        self.repository_directory
            .join("target/astronomical-macos-development.noindex/Astronomical Development.app")
    }

    pub(crate) fn stable_bundle_path(&self) -> PathBuf {
        self.repository_directory
            .join("target/astronomical-macos-stable.noindex/Astronomical.app")
    }

    pub(crate) fn app_store_bundle_path(&self) -> PathBuf {
        self.repository_directory
            .join("target/astronomical-macos-app-store.noindex/Astronomical.app")
    }

    pub(crate) fn cargo_arguments_text(&self) -> String {
        file_text(&self.cargo_arguments_log)
    }

    pub(crate) fn swift_arguments_text(&self) -> String {
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
