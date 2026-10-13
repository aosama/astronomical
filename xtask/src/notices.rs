//! Typed port of `scripts/ci/generate-rust-dependency-notices.sh`: regenerate
//! or check `third-party/RUST_DEPENDENCY_NOTICES` with the pinned cargo-about,
//! keeping the committed digest file byte-compatible so hooks and CI gates
//! survive the move unchanged.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use sha2::{Digest, Sha256};

const NOTICES_RELATIVE_PATH: &str = "third-party/RUST_DEPENDENCY_NOTICES";
const DIGEST_RELATIVE_PATH: &str = "third-party/RUST_DEPENDENCY_NOTICES.digest";
const PIN_RELATIVE_PATH: &str = "third-party/cargo-about-version";

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut hexadecimal = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hexadecimal.push_str(&format!("{byte:02x}"));
    }
    hexadecimal
}

fn read_pinned_cargo_about_version(repository_root: &Path) -> Result<String, String> {
    let pin_path = repository_root.join(PIN_RELATIVE_PATH);
    let pinned_version = fs::read_to_string(&pin_path)
        .map_err(|_| {
            format!(
                "[rust-dependency-notices] missing cargo-about pin file: {}",
                pin_path.display()
            )
        })?
        .trim()
        .to_owned();
    if pinned_version.is_empty()
        || pinned_version
            .chars()
            .any(|character| !(character.is_ascii_digit() || character == '.'))
    {
        return Err(format!(
            "[rust-dependency-notices] cargo-about pin must be a dotted version: {}",
            pin_path.display()
        ));
    }
    Ok(pinned_version)
}

fn require_pinned_cargo_about(required_version: &str) -> Result<(), String> {
    let installed_version = Command::new("cargo-about")
        .arg("--version")
        .output()
        .map_err(|_| {
            format!(
                "[rust-dependency-notices] cargo-about {required_version} is required; install that exact version"
            )
        })?;
    let installed_version_text = String::from_utf8_lossy(&installed_version.stdout)
        .trim()
        .to_owned();
    if installed_version_text != format!("cargo-about {required_version}") {
        return Err(format!(
            "[rust-dependency-notices] notices generation requires cargo-about {required_version}; found {installed_version_text}"
        ));
    }
    Ok(())
}

fn is_notices_input_path(relative_path: &str) -> bool {
    const TOP_LEVEL_INPUTS: [&str; 5] = [
        "Cargo.lock",
        "Cargo.toml",
        "third-party/about.toml",
        "third-party/rust-dependency-notices.hbs",
        "third-party/cargo-about-version",
    ];
    if TOP_LEVEL_INPUTS.contains(&relative_path) {
        return true;
    }
    const PACKAGE_TREES: [&str; 3] = ["apps/", "crates/", "experimental/"];
    if !PACKAGE_TREES
        .iter()
        .any(|tree| relative_path.starts_with(tree))
    {
        return false;
    }
    relative_path.ends_with("/Cargo.toml")
        || ["LICENSE", "LICENSE.md", "COPYING"]
            .iter()
            .any(|license_name| relative_path.ends_with(&format!("/{license_name}")))
}

fn list_notices_input_paths(repository_root: &Path) -> Result<Vec<String>, String> {
    let listed = Command::new("git")
        .args([
            "ls-files",
            "--",
            "Cargo.lock",
            "Cargo.toml",
            "third-party/about.toml",
            "third-party/rust-dependency-notices.hbs",
            "third-party/cargo-about-version",
            "apps",
            "crates",
            "experimental",
        ])
        .current_dir(repository_root)
        .output()
        .map_err(|error| format!("[rust-dependency-notices] git is required to digest tracked notices inputs: {error}"))?;
    if !listed.status.success() {
        return Err("[rust-dependency-notices] git ls-files failed for notices inputs".to_owned());
    }
    Ok(String::from_utf8_lossy(&listed.stdout)
        .lines()
        .filter(|path_text| is_notices_input_path(path_text))
        .map(str::to_owned)
        .collect())
}

fn compute_notices_input_digest(repository_root: &Path) -> Result<String, String> {
    let mut digest_material = String::new();
    let mut input_paths = list_notices_input_paths(repository_root)?;
    input_paths.sort();
    input_paths.dedup();
    for relative_path in input_paths {
        let input_path = repository_root.join(&relative_path);
        let Ok(input_bytes) = fs::read(&input_path) else {
            continue;
        };
        digest_material.push_str(&format!("{}  {relative_path}\n", sha256_hex(&input_bytes)));
    }
    Ok(sha256_hex(digest_material.as_bytes()))
}

fn read_digest_field(digest_text: &str, field_name: &str) -> Option<String> {
    digest_text.lines().find_map(|line| {
        line.strip_prefix(&format!("{field_name}="))
            .map(str::to_owned)
    })
}

fn notices_digest_is_current(
    repository_root: &Path,
    expected_input_digest: &str,
) -> Result<bool, String> {
    let Some(digest_text) = fs::read_to_string(repository_root.join(DIGEST_RELATIVE_PATH)).ok()
    else {
        return Ok(false);
    };
    let Ok(notices_bytes) = fs::read(repository_root.join(NOTICES_RELATIVE_PATH)) else {
        return Ok(false);
    };
    let Some(recorded_input_digest) = read_digest_field(&digest_text, "input_sha256") else {
        return Ok(false);
    };
    let Some(recorded_notices_digest) = read_digest_field(&digest_text, "notices_sha256") else {
        return Ok(false);
    };
    Ok(recorded_input_digest == expected_input_digest
        && recorded_notices_digest == sha256_hex(&notices_bytes))
}

fn write_notices_digest(
    digest_path: &Path,
    input_digest: &str,
    notices_digest: &str,
) -> Result<(), String> {
    fs::write(
        digest_path,
        format!("input_sha256={input_digest}\nnotices_sha256={notices_digest}\n"),
    )
    .map_err(|error| format!("[rust-dependency-notices] digest write failed: {error}"))
}

/// The perl one-liner from the shell script: normalize line endings, strip
/// trailing horizontal whitespace before each newline, and collapse any run
/// of trailing newlines to exactly one (absent trailing newline: unchanged).
/// Applied without changing license wording.
fn normalize_generated_notices(notices_text: &str) -> String {
    let normalized_newlines = notices_text.replace("\r\n", "\n");
    let mut without_trailing_whitespace = String::with_capacity(normalized_newlines.len());
    for line in normalized_newlines.split_inclusive('\n') {
        match line.strip_suffix('\n') {
            // The perl lookahead strips horizontal whitespace only before a
            // newline; a final line without a trailing newline is untouched.
            Some(line_body) => {
                without_trailing_whitespace.push_str(line_body.trim_end_matches([' ', '\t']));
                without_trailing_whitespace.push('\n');
            }
            None => without_trailing_whitespace.push_str(line),
        }
    }
    let body_without_trailing_newlines = without_trailing_whitespace.trim_end_matches('\n');
    let mut normalized = String::from(body_without_trailing_newlines);
    if without_trailing_whitespace.ends_with('\n') {
        normalized.push('\n');
    }
    normalized
}

fn generate_notices_file(
    repository_root: &Path,
    destination_path: &Path,
    pinned_version: &str,
) -> Result<(), String> {
    println!(
        "[rust-dependency-notices] status=generating destination={} cargo_about={pinned_version}",
        destination_path.display()
    );
    let generated = Command::new("cargo-about")
        .args([
            "generate",
            "--workspace",
            "--all-features",
            "--locked",
            "--fail",
            "--config",
        ])
        .arg(repository_root.join("third-party/about.toml"))
        .arg("--output-file")
        .arg(destination_path)
        .arg(repository_root.join("third-party/rust-dependency-notices.hbs"))
        .output()
        .map_err(|error| {
            format!("[rust-dependency-notices] cargo-about generate failed to start: {error}")
        })?;
    if !generated.status.success() {
        return Err(format!(
            "[rust-dependency-notices] cargo-about generate failed: {}",
            String::from_utf8_lossy(&generated.stderr)
        ));
    }
    let notices_text = fs::read_to_string(destination_path).map_err(|error| {
        format!("[rust-dependency-notices] generated notices unreadable: {error}")
    })?;
    fs::write(destination_path, normalize_generated_notices(&notices_text)).map_err(|error| {
        format!("[rust-dependency-notices] generated notices unwritable: {error}")
    })?;
    Ok(())
}

pub fn run(repository_root: &Path, check_only: bool) -> Result<(), String> {
    let started_at = Instant::now();
    let pinned_version = read_pinned_cargo_about_version(repository_root)?;
    require_pinned_cargo_about(&pinned_version)?;
    let generated_notices_path = repository_root.join(NOTICES_RELATIVE_PATH);
    let digest_path = repository_root.join(DIGEST_RELATIVE_PATH);
    let input_digest = compute_notices_input_digest(repository_root)?;

    if check_only && notices_digest_is_current(repository_root, &input_digest)? {
        println!(
            "[rust-dependency-notices] status=current reason=input-digest elapsed_seconds={} cargo_about={pinned_version}",
            started_at.elapsed().as_secs()
        );
        return Ok(());
    }

    let generation_destination_path: PathBuf = if check_only {
        std::env::temp_dir().join(format!("astronomical-rust-notices-{}", std::process::id()))
    } else {
        generated_notices_path.clone()
    };
    let generation_result = generate_notices_file(
        repository_root,
        &generation_destination_path,
        &pinned_version,
    );
    let regenerated_bytes = generation_result.and_then(|()| {
        fs::read(&generation_destination_path).map_err(|error| {
            format!("[rust-dependency-notices] regenerated notices unreadable: {error}")
        })
    });
    if check_only {
        let _ = fs::remove_file(&generation_destination_path);
    }
    let regenerated_bytes = regenerated_bytes?;

    if check_only {
        let committed_bytes = fs::read(&generated_notices_path).map_err(|error| {
            format!("[rust-dependency-notices] committed notices unreadable: {error}")
        })?;
        if regenerated_bytes != committed_bytes {
            return Err(
                "[rust-dependency-notices] third-party/RUST_DEPENDENCY_NOTICES is stale; regenerate it"
                    .to_owned(),
            );
        }
        if !notices_digest_is_current(repository_root, &input_digest)? {
            return Err(
                "[rust-dependency-notices] third-party/RUST_DEPENDENCY_NOTICES.digest is stale; run cargo xtask notices"
                    .to_owned(),
            );
        }
        println!(
            "[rust-dependency-notices] status=current reason=harvest elapsed_seconds={} cargo_about={pinned_version}",
            started_at.elapsed().as_secs()
        );
        return Ok(());
    }

    let notices_digest = sha256_hex(&regenerated_bytes);
    write_notices_digest(&digest_path, &input_digest, &notices_digest)?;
    println!(
        "[rust-dependency-notices] status=updated elapsed_seconds={} cargo_about={pinned_version}",
        started_at.elapsed().as_secs()
    );
    Ok(())
}
