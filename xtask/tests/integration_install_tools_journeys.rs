//! Installer journeys with a shimmed `curl`: the download-and-install seam is
//! the subject, so these are integration journeys. PATH is replaced with the
//! shim directory (plus a symlinked `uname`), so the real network is
//! unreachable and no unpinned installer path can hide (issue #1250).

use astronomical_native_identity::digest_bytes;
use std::path::{Path, PathBuf};
use std::process::Command;

struct InstallerSandbox {
    root_directory: PathBuf,
    prefix_directory: PathBuf,
    fixture_directory: PathBuf,
    curl_log_path: PathBuf,
    shim_directory: PathBuf,
}

impl InstallerSandbox {
    fn install() -> Self {
        // Tests run in parallel threads inside one process, so the process
        // id alone cannot name the sandbox root: a sibling test's cleanup
        // would wipe this test's prefix mid-install.
        static SANDBOX_SEQUENCE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let sandbox_sequence = SANDBOX_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let root_directory = std::env::temp_dir().join(format!(
            "astronomical-install-tools-{}-{sandbox_sequence}",
            std::process::id()
        ));
        let prefix_directory = root_directory.join("prefix");
        let fixture_directory = root_directory.join("fixtures");
        let shim_directory = root_directory.join("shim-bin");
        for directory in [
            &prefix_directory,
            &fixture_directory,
            &shim_directory,
            &root_directory.join("home"),
        ] {
            std::fs::create_dir_all(directory).unwrap();
        }
        let curl_log_path = root_directory.join("curl.log");
        std::fs::write(&curl_log_path, "").unwrap();
        let sandbox = Self {
            root_directory,
            prefix_directory,
            fixture_directory,
            curl_log_path,
            shim_directory,
        };
        sandbox.write_curl_shim();
        // The port probes the host with uname and the curl shim copies
        // fixture files with cp; PATH is replaced, so the shim directory
        // carries wrappers that exec those platform tools by absolute path
        // (no PATH lookup, no symlink or copy needed).
        for (wrapper_name, wrapped_absolute_path) in
            [("uname", "/usr/bin/uname"), ("cp", "/bin/cp")]
        {
            std::fs::write(
                sandbox.shim_directory.join(wrapper_name),
                format!("#!/bin/sh\nexec {wrapped_absolute_path} \"$@\"\n"),
            )
            .expect("the shim wrapper should be writable");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(
                    sandbox.shim_directory.join(wrapper_name),
                    std::fs::Permissions::from_mode(0o755),
                )
                .unwrap();
            }
        }
        sandbox
    }

    /// The shim serves fixture files by URL base name and appends every URL
    /// to the log; only shell builtins are used because the shim directory
    /// is the whole PATH.
    fn write_curl_shim(&self) {
        let shim_script = format!(
            "#!/bin/sh\nurl=\"\"\ndest=\"\"\nprevious=\"\"\nfor argument in \"$@\"; do\n    case \"$previous\" in\n        --output) dest=\"$argument\" ;;\n    esac\n    case \"$argument\" in\n        --*) ;;\n        *) url=\"$argument\" ;;\n    esac\n    previous=\"$argument\"\ndone\nprintf '%s\\n' \"$url\" >> \"{}\"\nbase_name=\"${{url##*/}}\"\ncp \"{}/$base_name\" \"$dest\"\n",
            self.curl_log_path.display(),
            self.fixture_directory.display()
        );
        let curl_path = self.shim_directory.join("curl");
        std::fs::write(&curl_path, shim_script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&curl_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    /// Builds a one-directory tarball whose single binary prints the given
    /// version line, plus its sidecar carrying the real archive digest.
    fn create_release_fixture(&self, archive_name: &str, tool_name: &str, version_line: &str) {
        let staging_directory = self.root_directory.join(format!("stage-{tool_name}"));
        let payload_directory = staging_directory.join(format!("{tool_name}-dir"));
        std::fs::create_dir_all(&payload_directory).unwrap();
        let binary_path = payload_directory.join(tool_name);
        std::fs::write(
            &binary_path,
            format!("#!/bin/sh\nprintf '%s\\n' '{version_line}'\n"),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&binary_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let archive_path = self.fixture_directory.join(archive_name);
        let tar_status = Command::new("/usr/bin/tar")
            .args(["-czf"])
            .arg(&archive_path)
            .arg("-C")
            .arg(&staging_directory)
            .arg(format!("{tool_name}-dir"))
            .status()
            .expect("tar should run in the test process");
        assert!(tar_status.success());
        let archive_bytes = std::fs::read(&archive_path).unwrap();
        let sidecar_text = format!("{}  {archive_name}\n", digest_bytes(&archive_bytes));
        std::fs::write(
            self.fixture_directory
                .join(format!("{archive_name}.sha256")),
            sidecar_text,
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&staging_directory);
    }

    fn write_version_stub(&self, tool_name: &str, version_line: &str) {
        let stub_path = self.prefix_directory.join(tool_name);
        std::fs::write(
            &stub_path,
            format!("#!/bin/sh\nprintf '%s\\n' '{version_line}'\n"),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&stub_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    fn installed_version(&self, tool_name: &str) -> String {
        let output = Command::new(self.prefix_directory.join(tool_name))
            .arg("--version")
            .output()
            .expect("the installed stub should run");
        String::from_utf8_lossy(&output.stdout).trim().to_owned()
    }

    fn run_installer(&self) -> std::process::Output {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["install-verification-tools", "--prefix"])
            .arg(&self.prefix_directory)
            .env("PATH", &self.shim_directory)
            .env("HOME", self.root_directory.join("home"))
            .env_remove("CARGO_HOME")
            .env_remove("GITHUB_PATH")
            .env("TMPDIR", &self.root_directory)
            .output()
            .expect("the xtask binary should run")
    }
}

impl Drop for InstallerSandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root_directory);
    }
}

fn pinned_version(repository_root: &Path, pin_file_name: &str) -> String {
    std::fs::read_to_string(repository_root.join("third-party").join(pin_file_name))
        .expect("the pin file should exist")
        .trim()
        .to_owned()
}

fn host_target_triple() -> &'static str {
    let kernel = Command::new("uname").arg("-s").output().unwrap();
    let machine = Command::new("uname").arg("-m").output().unwrap();
    match (
        String::from_utf8_lossy(&kernel.stdout).trim(),
        String::from_utf8_lossy(&machine.stdout).trim(),
    ) {
        ("Darwin", "arm64") => "aarch64-apple-darwin",
        ("Darwin", "x86_64") => "x86_64-apple-darwin",
        ("Linux", "x86_64") => "x86_64-unknown-linux-musl",
        _ => panic!("unsupported journey host"),
    }
}

#[test]
fn matching_prefix_skips_all_downloads() {
    let sandbox = InstallerSandbox::install();
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .unwrap()
        .to_path_buf();
    let cargo_about_version = pinned_version(&repository_root, "cargo-about-version");
    let sccache_version = pinned_version(&repository_root, "sccache-version");
    sandbox.write_version_stub("cargo-about", &format!("cargo-about {cargo_about_version}"));
    sandbox.write_version_stub("sccache", &format!("sccache {sccache_version}"));

    let installer_run = sandbox.run_installer();
    assert!(installer_run.status.success());
    let curl_log = std::fs::read_to_string(&sandbox.curl_log_path).unwrap();
    assert!(
        curl_log.trim().is_empty(),
        "matching prefix must not download: {curl_log}"
    );
}

#[test]
fn fresh_prefix_installs_both_pinned_prebuilts_from_their_release_urls() {
    let sandbox = InstallerSandbox::install();
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .unwrap()
        .to_path_buf();
    let cargo_about_version = pinned_version(&repository_root, "cargo-about-version");
    let sccache_version = pinned_version(&repository_root, "sccache-version");
    let target = host_target_triple();
    sandbox.create_release_fixture(
        &format!("cargo-about-{cargo_about_version}-{target}.tar.gz"),
        "cargo-about",
        &format!("cargo-about {cargo_about_version}"),
    );
    sandbox.create_release_fixture(
        &format!("sccache-v{sccache_version}-{target}.tar.gz"),
        "sccache",
        &format!("sccache {sccache_version}"),
    );

    let installer_run = sandbox.run_installer();
    assert!(
        installer_run.status.success(),
        "installer failed: {}",
        String::from_utf8_lossy(&installer_run.stderr)
    );

    let installed_cargo_about = sandbox.installed_version("cargo-about");
    assert_eq!(
        installed_cargo_about,
        format!("cargo-about {cargo_about_version}"),
        "prefix listing: {:?}",
        std::fs::read_dir(&sandbox.prefix_directory)
            .map(|entries| entries.flatten().map(|entry| entry.path()))
            .map(|paths| paths
                .map(|path| (
                    path.clone(),
                    std::fs::read_to_string(&path).unwrap_or_default()
                ))
                .collect::<Vec<_>>())
    );
    assert_eq!(
        sandbox.installed_version("sccache"),
        format!("sccache {sccache_version}")
    );
    let curl_log = std::fs::read_to_string(&sandbox.curl_log_path).unwrap();
    assert!(curl_log.contains(&format!(
        "github.com/EmbarkStudios/cargo-about/releases/download/{cargo_about_version}/"
    )));
    assert!(curl_log.contains(&format!(
        "github.com/mozilla/sccache/releases/download/v{sccache_version}/"
    )));
}
