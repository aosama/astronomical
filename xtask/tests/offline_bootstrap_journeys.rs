//! Bootstrap journeys over the real xtask binary with shimmed `cmake`, `curl`,
//! and `df`: the pinned-manifest seam and the verify-or-download seam are the
//! subjects. PATH is replaced with the shim directory, so the real network and
//! the real CMake pins are unreachable (issue #1250).

use astronomical_native_identity::digest_bytes;
use std::path::PathBuf;
use std::process::Command;

const FIXTURE_ARCHIVE_NAMES: [&str; 5] = [
    "mlx.tar.gz",
    "mlx-c.tar.gz",
    "metal-cpp.tar.gz",
    "nlohmann-json.tar.gz",
    "fmt.tar.gz",
];

struct BootstrapSandbox {
    root_directory: PathBuf,
    cache_directory: PathBuf,
    fixture_repository_root: PathBuf,
    shim_directory: PathBuf,
}

impl BootstrapSandbox {
    fn install() -> Self {
        // Tests run in parallel threads inside one process, so the process
        // id alone cannot name the sandbox root: a sibling test's cleanup
        // would wipe this test's cache mid-run.
        static SANDBOX_SEQUENCE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let sandbox_sequence = SANDBOX_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let root_directory = std::env::temp_dir().join(format!(
            "astronomical-bootstrap-{}-{sandbox_sequence}",
            std::process::id()
        ));
        let cache_directory = root_directory.join("cache");
        let shim_directory = root_directory.join("shim-bin");
        let fixture_repository_root = root_directory.join("repository");
        for directory in [
            &root_directory,
            &cache_directory,
            &shim_directory,
            &fixture_repository_root.join("third-party"),
        ] {
            std::fs::create_dir_all(directory).unwrap();
        }
        // The real CMake manifest is never executed (cmake is shimmed), but
        // the path must exist because the command names it.
        std::fs::write(
            fixture_repository_root.join("third-party/native-dependency-manifest.cmake"),
            "# fixture manifest\n",
        )
        .unwrap();
        Self {
            root_directory,
            cache_directory,
            fixture_repository_root,
            shim_directory,
        }
    }

    fn archive_bytes(archive_index: usize) -> String {
        format!("fixture-archive-payload-{archive_index}")
    }

    /// Writes one fixture archive into the cache, optionally with tampered
    /// content; returns its pinned SHA-256.
    fn seed_archive(&self, archive_index: usize, tamper: bool) -> String {
        let mut archive_bytes = Self::archive_bytes(archive_index);
        if tamper {
            archive_bytes.push_str("-tampered");
        }
        let archive_path = self
            .cache_directory
            .join(FIXTURE_ARCHIVE_NAMES[archive_index]);
        std::fs::write(archive_path, archive_bytes).unwrap();
        digest_bytes(Self::archive_bytes(archive_index).as_bytes())
    }

    fn seed_full_warm_cache(&self) -> [String; 5] {
        let mut pinned_digests = Vec::new();
        for archive_index in 0..FIXTURE_ARCHIVE_NAMES.len() {
            pinned_digests.push(self.seed_archive(archive_index, false));
        }
        pinned_digests.try_into().unwrap()
    }

    /// The pinned digests for the five fixture archives, without touching
    /// the cache: journeys that exercise the download path need an empty
    /// cache while the manifest still names the pinned digests.
    fn pinned_digests() -> [String; 5] {
        core::array::from_fn(|archive_index| {
            digest_bytes(Self::archive_bytes(archive_index).as_bytes())
        })
    }

    /// The digest of whatever the curl shim serves, for journeys that
    /// exercise the download path end to end.
    fn downloaded_payload_digests() -> [String; 5] {
        core::array::from_fn(|_| digest_bytes(b"downloaded-archive-payload"))
    }

    /// The cmake shim writes the pinned manifest with shell builtins only;
    /// the pinned digests are baked in when the shim is authored.
    fn write_cmake_shim(&self, pinned_digests: &[String]) {
        let mut manifest_lines = String::new();
        for (archive_index, archive_name) in FIXTURE_ARCHIVE_NAMES.iter().enumerate() {
            manifest_lines.push_str(&format!(
                "'{}|https://fixtures.invalid/{}|{}|fixture archive {archive_index}' ",
                archive_name, archive_name, pinned_digests[archive_index]
            ));
        }
        let shim_script = format!(
            "#!/bin/sh\nmanifest_path=\"\"\nfor argument in \"$@\"; do\n    case \"$argument\" in\n        -DASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH=*) manifest_path=\"${{argument#-DASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH=}}\" ;;\n    esac\ndone\nprintf '%s\\n' {manifest_lines} > \"$manifest_path\"\n",
        );
        self.write_executable_shim("cmake", &shim_script);
    }

    /// The curl shim serves canned bytes for every URL; only shell builtins
    /// are used because the shim directory is the whole PATH.
    fn write_curl_shim(&self) {
        let shim_script = format!(
            "#!/bin/sh\ndestination=\"\"\nprevious=\"\"\nfor argument in \"$@\"; do\n    case \"$previous\" in\n        --output) destination=\"$argument\" ;;\n    esac\n    previous=\"$argument\"\ndone\nprintf '%s' 'downloaded-archive-payload' > \"$destination\"\n"
        );
        self.write_executable_shim("curl", &shim_script);
    }

    fn write_df_shim(&self, available_kib_blocks: &str) {
        let shim_script = format!(
            "#!/bin/sh\nprintf '%s\\n' 'Filesystem 1024-blocks Used Available Capacity Mounted on' '/dev/disk1s1 1000000000 500000000 {available_kib_blocks} 50% /'\n"
        );
        self.write_executable_shim("df", &shim_script);
    }

    fn write_executable_shim(&self, shim_name: &str, shim_script: &str) {
        let shim_path = self.shim_directory.join(shim_name);
        std::fs::write(&shim_path, shim_script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&shim_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    fn run_xtask(&self, extra_arguments: &[&str]) -> std::process::Output {
        let mut xtask_arguments = vec![
            "--repository-root".to_owned(),
            self.fixture_repository_root.display().to_string(),
            "bootstrap-native-dependencies".to_owned(),
            "--cache-dir".to_owned(),
            self.cache_directory.display().to_string(),
        ];
        xtask_arguments.extend(extra_arguments.iter().map(|argument| argument.to_string()));
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(&xtask_arguments)
            // PATH is replaced, never appended: no real cmake, curl, or df
            // can hide behind the shims.
            .env("PATH", &self.shim_directory)
            .output()
            .expect("the xtask binary should be runnable")
    }
}

impl Drop for BootstrapSandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root_directory);
    }
}

fn command_text(output: &std::process::Output) -> String {
    format!(
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn verify_only_provisions_from_a_warm_cache() {
    let sandbox = BootstrapSandbox::install();
    sandbox.seed_full_warm_cache();
    // The cmake shim is the only command on PATH: verify-only must reach
    // neither curl nor df.
    sandbox.write_cmake_shim(&BootstrapSandbox::pinned_digests());
    let output = sandbox.run_xtask(&["--verify"]);
    assert!(
        output.status.success(),
        "verify-only run failed: {}",
        command_text(&output)
    );
    let stdout_text = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout_text.matches("status=verified-cached").count(), 5);
    assert!(stdout_text.contains("status=ready"));
    for archive_index in 0..FIXTURE_ARCHIVE_NAMES.len() {
        let archive_path = sandbox
            .cache_directory
            .join(FIXTURE_ARCHIVE_NAMES[archive_index]);
        assert_eq!(
            std::fs::read(&archive_path).unwrap(),
            BootstrapSandbox::archive_bytes(archive_index).into_bytes(),
            "a verified cache entry must be left untouched: {}",
            archive_path.display()
        );
    }
}

#[test]
fn verify_only_refuses_a_tampered_archive() {
    let sandbox = BootstrapSandbox::install();
    let pinned_digests = sandbox.seed_full_warm_cache();
    sandbox.seed_archive(2, true);
    sandbox.write_cmake_shim(&pinned_digests);
    let output = sandbox.run_xtask(&["--verify"]);
    assert!(
        !output.status.success(),
        "a tampered archive must fail the verify run: {}",
        command_text(&output)
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr_text.contains("missing or invalid"),
        "the refusal should name the invalid archive: {stderr_text}"
    );
}

#[test]
fn verify_only_refuses_a_missing_archive() {
    let sandbox = BootstrapSandbox::install();
    sandbox.write_cmake_shim(&BootstrapSandbox::pinned_digests());
    let output = sandbox.run_xtask(&["--verify"]);
    assert!(
        !output.status.success(),
        "an empty cache must fail the verify run: {}",
        command_text(&output)
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    assert!(stderr_text.contains("missing or invalid"));
}

#[test]
fn download_verifies_and_installs_archives_atomically() {
    let sandbox = BootstrapSandbox::install();
    sandbox.write_cmake_shim(&BootstrapSandbox::downloaded_payload_digests());
    sandbox.write_curl_shim();
    sandbox.write_df_shim("4000000000");
    let output = sandbox.run_xtask(&[]);
    assert!(
        output.status.success(),
        "download run failed: {}",
        command_text(&output)
    );
    let stdout_text = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout_text
            .matches("status=downloaded-and-verified")
            .count(),
        5
    );
    for archive_name in FIXTURE_ARCHIVE_NAMES {
        let archive_path = sandbox.cache_directory.join(archive_name);
        let archive_bytes = std::fs::read(&archive_path).unwrap();
        assert_eq!(archive_bytes, b"downloaded-archive-payload");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let archive_mode = std::fs::metadata(&archive_path)
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(archive_mode & 0o777, 0o600, "installed {archive_name} mode");
        }
    }
    let leftover_staging_files: Vec<PathBuf> = std::fs::read_dir(&sandbox.cache_directory)
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .map(|file_name| file_name.to_string_lossy().contains(".download."))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        leftover_staging_files.is_empty(),
        "download staging files must not survive the run"
    );
}

#[test]
fn download_refuses_an_insufficient_free_space_cache() {
    let sandbox = BootstrapSandbox::install();
    // The free-space gate fires before the manifest read, so df alone is
    // enough to reach the refusal.
    sandbox.write_df_shim("1000");
    let output = sandbox.run_xtask(&[]);
    assert!(
        !output.status.success(),
        "a nearly-full filesystem must refuse the download: {}",
        command_text(&output)
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr_text.contains("needs at least 2147483648 free bytes"),
        "the refusal should name the free-space requirement: {stderr_text}"
    );
}

#[test]
fn download_refuses_an_archive_that_fails_its_pinned_digest() {
    // The manifest pins the fixture archives' digests while the curl shim
    // serves different bytes: the run must fail at the digest check, install
    // nothing, and leave no staging file behind.
    let sandbox = BootstrapSandbox::install();
    sandbox.write_cmake_shim(&BootstrapSandbox::pinned_digests());
    sandbox.write_curl_shim();
    sandbox.write_df_shim("4000000000");
    let output = sandbox.run_xtask(&[]);
    assert!(
        !output.status.success(),
        "a download that fails its pinned digest must fail the run: {}",
        command_text(&output)
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr_text.contains("SHA-256 verification failed"),
        "the refusal should name the digest failure: {stderr_text}"
    );
    let published_files: Vec<PathBuf> = std::fs::read_dir(&sandbox.cache_directory)
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .collect();
    assert!(
        published_files.is_empty(),
        "a refused download must not publish any archive"
    );
}

#[test]
fn manifest_url_scheme_is_validated_before_any_download() {
    // The journey proves the manifest policy holds end to end: a shimmed
    // manifest with an http:// URL fails before any download starts.
    let sandbox = BootstrapSandbox::install();
    let insecure_manifest_line = "'mlx.tar.gz|http://fixtures.invalid/mlx.tar.gz|".to_owned()
        + &digest_bytes(b"payload")
        + "|fixture'";
    let shim_script = format!(
        "#!/bin/sh\nmanifest_path=\"\"\nfor argument in \"$@\"; do\n    case \"$argument\" in\n        -DASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH=*) manifest_path=\"${{argument#-DASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH=}}\" ;;\n    esac\ndone\nprintf '%s\\n' {insecure_manifest_line} > \"$manifest_path\"\n"
    );
    sandbox.write_executable_shim("cmake", &shim_script);
    sandbox.write_df_shim("4000000000");
    let output = sandbox.run_xtask(&[]);
    assert!(
        !output.status.success(),
        "an insecure manifest URL must fail the run: {}",
        command_text(&output)
    );
    let stderr_text = String::from_utf8_lossy(&output.stderr);
    assert!(stderr_text.contains("HTTPS archive URL"));
}
