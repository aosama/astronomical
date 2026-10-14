//! `cargo xtask install-verification-tools` — installs the pinned cargo-about
//! and sccache prebuilt archives from GitHub Releases. Hosted CI once spent
//! minutes compiling cargo-about from crates.io and brewing sccache; both
//! publishers already ship SHA-256-verified binaries. The pure policy (pin
//! reading, host mapping, archive naming, sidecar digest parsing) is
//! unit-tested beside this source; `curl` and the platform `tar` stay thin
//! edges (issue #1250).

use std::path::{Path, PathBuf};
use std::process::Command;

use astronomical_native_identity::digest_bytes;

const CARGO_ABOUT_RELEASE_URL_ROOT: &str =
    "https://github.com/EmbarkStudios/cargo-about/releases/download";
const SCCACHE_RELEASE_URL_ROOT: &str = "https://github.com/mozilla/sccache/releases/download";

/// The drift-failure contract: the installed binary must report the pinned
/// version, and this message names the requirement (pinned by the dependency
/// notices contract).
const DRIFT_ERROR_TEMPLATE: &str = "installed {tool} is {installed}; Astronomical requires cargo-about and sccache at the pinned versions ({expected})";

/// Maps the running kernel and machine onto the prebuilt archive target
/// triple. No mapping means no prebuilt archive exists and the caller must
/// fail loudly instead of compiling or brewing.
pub fn host_rust_target(kernel: &str, machine: &str) -> Result<&'static str, String> {
    match (kernel, machine) {
        ("Darwin", "arm64") => Ok("aarch64-apple-darwin"),
        ("Darwin", "x86_64") => Ok("x86_64-apple-darwin"),
        ("Linux", "aarch64" | "arm64") => Ok("aarch64-unknown-linux-musl"),
        ("Linux", "x86_64") => Ok("x86_64-unknown-linux-musl"),
        (kernel, machine) => Err(format!(
            "no prebuilt verification-tool archive mapping for {kernel} {machine}"
        )),
    }
}

/// A pin file must carry exactly one dotted version.
pub fn read_dotted_version_pin(pin_text: &str) -> Result<String, String> {
    let pinned_version = pin_text.trim();
    let is_dotted_version = !pinned_version.is_empty()
        && pinned_version
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'.'));
    if is_dotted_version {
        Ok(pinned_version.to_owned())
    } else {
        Err("pin must be a dotted version".to_owned())
    }
}

pub fn cargo_about_archive_name(version: &str, target: &str) -> String {
    format!("cargo-about-{version}-{target}.tar.gz")
}

pub fn sccache_archive_name(version: &str, target: &str) -> String {
    format!("sccache-v{version}-{target}.tar.gz")
}

pub fn cargo_about_release_url(version: &str, archive_name: &str) -> String {
    format!("{CARGO_ABOUT_RELEASE_URL_ROOT}/{version}/{archive_name}")
}

pub fn sccache_release_url(version: &str, archive_name: &str) -> String {
    format!("{SCCACHE_RELEASE_URL_ROOT}/v{version}/{archive_name}")
}

pub fn expected_version_line(tool_name: &str, version: &str) -> String {
    format!("{tool_name} {version}")
}

/// The published sidecar carries `<64-hex digest>  <file name>`; any digest
/// hex token wins because publishers disagree about trailing fields.
pub fn published_digest_from_sidecar(sidecar_text: &str) -> Result<String, String> {
    for token in sidecar_text.split_whitespace() {
        let is_sixty_four_hex = token.len() == 64
            && token
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F'));
        if is_sixty_four_hex {
            return Ok(token.to_lowercase());
        }
    }
    Err("published SHA-256 sidecar is missing a digest".to_owned())
}

fn actual_archive_digest(archive_path: &Path) -> Result<String, String> {
    let archive_bytes =
        std::fs::read(archive_path).map_err(|error| format!("archive is unreadable: {error}"))?;
    Ok(digest_bytes(&archive_bytes))
}

struct ToolPlan {
    tool_name: &'static str,
    version: String,
    archive_name: String,
    archive_url: String,
    digest_url: String,
}

fn uname_value(uname_argument: &str) -> Result<String, String> {
    Command::new("uname")
        .arg(uname_argument)
        .output()
        .map(|uname_output| {
            String::from_utf8_lossy(&uname_output.stdout)
                .trim()
                .to_owned()
        })
        .map_err(|error| format!("uname failed: {error}"))
}

fn tool_plans(repository_root: &Path) -> Result<[ToolPlan; 2], String> {
    let cargo_about_pin_path = repository_root.join("third-party/cargo-about-version");
    let sccache_pin_path = repository_root.join("third-party/sccache-version");
    let cargo_about_version = read_dotted_version_pin(
        &std::fs::read_to_string(&cargo_about_pin_path).map_err(|error| {
            format!(
                "missing cargo-about pin file {}: {error}",
                cargo_about_pin_path.display()
            )
        })?,
    )
    .map_err(|error| format!("cargo-about {error}"))?;
    let sccache_version = read_dotted_version_pin(
        &std::fs::read_to_string(&sccache_pin_path).map_err(|error| {
            format!(
                "missing sccache pin file {}: {error}",
                sccache_pin_path.display()
            )
        })?,
    )
    .map_err(|error| format!("sccache {error}"))?;
    let kernel = uname_value("-s")?;
    let machine = uname_value("-m")?;
    let target = host_rust_target(&kernel, &machine)?;
    let cargo_about_archive = cargo_about_archive_name(&cargo_about_version, target);
    let sccache_archive = sccache_archive_name(&sccache_version, target);
    Ok([
        ToolPlan {
            tool_name: "cargo-about",
            version: cargo_about_version.clone(),
            archive_name: cargo_about_archive.clone(),
            archive_url: cargo_about_release_url(&cargo_about_version, &cargo_about_archive),
            digest_url: format!(
                "{}.sha256",
                cargo_about_release_url(&cargo_about_version, &cargo_about_archive)
            ),
        },
        ToolPlan {
            tool_name: "sccache",
            version: sccache_version.clone(),
            archive_name: sccache_archive.clone(),
            archive_url: sccache_release_url(&sccache_version, &sccache_archive),
            digest_url: format!(
                "{}.sha256",
                sccache_release_url(&sccache_version, &sccache_archive)
            ),
        },
    ])
}

fn tool_reports_expected_version(executable_path: &Path, expected_line: &str) -> bool {
    let version_output = Command::new(executable_path)
        .arg("--version")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    version_output
        .map(|version_line| version_line == expected_line)
        .unwrap_or(false)
}

fn download_with_curl(download_destination: &Path, download_url: &str) -> Result<(), String> {
    let curl_status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--retry",
            "3",
            "--retry-delay",
            "1",
        ])
        .arg("--output")
        .arg(download_destination)
        .arg(download_url)
        .status()
        .map_err(|error| format!("curl failed to start: {error}"))?;
    if curl_status.success() {
        Ok(())
    } else {
        Err(format!("failed to download {download_url}"))
    }
}

/// Extracts the single named binary from a one-directory tarball
/// (`tool-dir/tool` or a bare `tool` at the root).
fn extract_named_binary(
    archive_path: &Path,
    binary_name: &str,
    destination_path: &Path,
    scratch_directory: &Path,
) -> Result<(), String> {
    let extraction_directory = scratch_directory.join(format!("extract-{binary_name}"));
    let _ = std::fs::remove_dir_all(&extraction_directory);
    std::fs::create_dir_all(&extraction_directory)
        .map_err(|error| format!("extraction directory failed: {error}"))?;
    // The platform tar by absolute path: this runs on the macOS and ubuntu
    // hosted runners and developer machines, where /usr/bin/tar exists on
    // both; keeping tar external avoids two new crate dependencies.
    let tar_status = Command::new("/usr/bin/tar")
        .args(["-xzf"])
        .arg(archive_path)
        .arg("-C")
        .arg(&extraction_directory)
        .status()
        .map_err(|error| format!("tar failed to start: {error}"))?;
    if !tar_status.success() {
        return Err(format!("tar extraction failed for {binary_name}"));
    }
    let mut matching_binaries: Vec<PathBuf> = ["", "*/"]
        .iter()
        .filter_map(|layout_prefix| {
            let candidate_pattern = format!("{layout_prefix}{binary_name}");
            let candidate_path = extraction_directory.join(&candidate_pattern);
            candidate_path.is_file().then_some(candidate_path)
        })
        .collect();
    if let Ok(read_matches) = std::fs::read_dir(&extraction_directory) {
        for entry in read_matches.flatten() {
            let candidate_path = entry.path().join(binary_name);
            if candidate_path.is_file() {
                matching_binaries.push(candidate_path);
            }
        }
    }
    matching_binaries.dedup();
    if matching_binaries.len() != 1 {
        return Err(format!(
            "archive for {binary_name} did not contain exactly one {binary_name} binary"
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            &matching_binaries[0],
            std::fs::Permissions::from_mode(0o755),
        )
        .map_err(|error| format!("chmod failed: {error}"))?;
    }
    std::fs::rename(&matching_binaries[0], destination_path)
        .map_err(|error| format!("move into place failed: {error}"))?;
    Ok(())
}

fn install_pinned_binary(
    plan: &ToolPlan,
    install_prefix: &Path,
    scratch_directory: &Path,
) -> Result<(), String> {
    let expected_line = expected_version_line(plan.tool_name, &plan.version);
    let installed_binary_path = install_prefix.join(plan.tool_name);
    if tool_reports_expected_version(&installed_binary_path, &expected_line) {
        eprintln!(
            "[verification-tools] status=already-installed tool={}",
            plan.tool_name
        );
        return Ok(());
    }
    eprintln!(
        "[verification-tools] status=start tool={} phase=download",
        plan.tool_name
    );
    let archive_path = scratch_directory.join(&plan.archive_name);
    let sidecar_path = scratch_directory.join(format!("{}.sha256", plan.archive_name));
    download_with_curl(&archive_path, &plan.archive_url)?;
    download_with_curl(&sidecar_path, &plan.digest_url)?;
    eprintln!(
        "[verification-tools] status=success tool={} phase=download",
        plan.tool_name
    );
    eprintln!(
        "[verification-tools] status=start tool={} phase=verify-digest",
        plan.tool_name
    );
    let published_digest = published_digest_from_sidecar(
        &std::fs::read_to_string(&sidecar_path)
            .map_err(|error| format!("sidecar unreadable: {error}"))?,
    )?;
    let actual_digest = actual_archive_digest(&archive_path)?;
    if actual_digest != published_digest {
        return Err(format!("SHA-256 mismatch for {}", plan.tool_name));
    }
    eprintln!(
        "[verification-tools] status=success tool={} phase=verify-digest",
        plan.tool_name
    );
    eprintln!(
        "[verification-tools] status=start tool={} phase=install",
        plan.tool_name
    );
    extract_named_binary(
        &archive_path,
        plan.tool_name,
        &installed_binary_path,
        scratch_directory,
    )?;
    if !tool_reports_expected_version(&installed_binary_path, &expected_line) {
        let installed_line = Command::new(&installed_binary_path)
            .arg("--version")
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .unwrap_or_else(|_| "unreadable".to_owned());
        return Err(DRIFT_ERROR_TEMPLATE
            .replace("{tool}", plan.tool_name)
            .replace("{installed}", &installed_line)
            .replace("{expected}", &expected_line));
    }
    eprintln!(
        "[verification-tools] status=success tool={} phase=install",
        plan.tool_name
    );
    Ok(())
}

pub fn run(mut arguments: impl Iterator<Item = String>) -> Result<(), String> {
    let started_at = std::time::Instant::now();
    let mut install_prefix: Option<PathBuf> = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--prefix" => {
                install_prefix = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or("--prefix requires a directory".to_owned())?,
                ));
            }
            "--help" | "-h" => {
                eprintln!("usage: cargo xtask install-verification-tools [--prefix DIR]");
                return Ok(());
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }
    eprintln!("[verification-tools] status=start");
    let repository_root = crate::workspace_root_from_manifest_directory();
    let install_prefix = install_prefix.unwrap_or_else(|| {
        let cargo_home = std::env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    std::env::var_os("HOME").expect("HOME selects the cargo bin directory"),
                )
                .join(".cargo")
            });
        cargo_home.join("bin")
    });
    std::fs::create_dir_all(&install_prefix)
        .map_err(|error| format!("install prefix failed: {error}"))?;
    let scratch_directory = tempfile_directory()?;
    for tool_plan in tool_plans(&repository_root)? {
        install_pinned_binary(&tool_plan, &install_prefix, &scratch_directory)?;
    }
    if let Ok(github_path) = std::env::var("GITHUB_PATH") {
        use std::io::Write;
        if let Ok(mut github_path_file) =
            std::fs::OpenOptions::new().append(true).open(&github_path)
        {
            let _ = writeln!(github_path_file, "{}", install_prefix.display());
        }
    }
    let _ = std::fs::remove_dir_all(&scratch_directory);
    eprintln!(
        "[verification-tools] status=success elapsed_seconds={} prefix={}",
        started_at.elapsed().as_secs(),
        install_prefix.display()
    );
    Ok(())
}

fn tempfile_directory() -> Result<PathBuf, String> {
    let base_directory = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir());
    for attempt in 0..1000 {
        let candidate = base_directory.join(format!(
            "astronomical-verification-tools-{}-{attempt}",
            std::process::id()
        ));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("scratch directory failed: {error}")),
        }
    }
    Err("scratch directory failed: exhausted attempts".to_owned())
}
