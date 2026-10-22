//! Native build identity: the fingerprint shared by the Cargo
//! build scripts, the standalone native-build tool, and CI cache keys.
//!
//! Ownership rationale: this logic previously lived in a POSIX shell script
//! that every consumer had to spawn as a subprocess; the Cargo build scripts
//! cannot spawn `cargo run` from inside their own build, so the policy lives
//! here as a library consumed in-process (build-dependency) and through the
//! `cargo xtask native-fingerprint` CLI everywhere else (issue #1250).

pub mod probes;

#[cfg(test)]
mod lib_tests;
#[cfg(test)]
mod probes_tests;

use std::path::{Path, PathBuf};
use std::process::Command;

use probes::ToolchainOverrides;

/// The identity manifest format revision. The format itself (tab-separated
/// schema/profile/source/toolchain lines) is unchanged from the shell era;
/// bump only when line grammar changes. The toolchain section's historical
/// `compatibility` line label is hashed bytes, so the label itself stays.
const MANIFEST_SCHEMA_VERSION: u32 = 2;

/// Source-identity pathspecs: every input that determines the native
/// artifact's bytes. The Rust build-orchestration sources (`build.rs` and
/// its `build_*` modules) and the `xtask` CLI executor are deliberately
/// excluded because they are executors, not policy: edits to progress
/// logging, parallelism resolution, or store plumbing must not force a full
/// CMake rebuild. Every artifact-affecting cmake argument is already
/// fingerprinted through the toolchain probes, the profile, and the
/// third-party files; a change to cmake invocation semantics must bump
/// `crates/runtime-integration/native-build-store-schema-version` instead.
pub const SOURCE_IDENTITY_PATHSPECS: &[&str] = &[
    "crates/native-identity/src",
    "crates/runtime-integration/native-build-store-schema-version",
    "crates/runtime-integration/native",
    "third-party/native-dependency-manifest.cmake",
    "third-party/pins",
    "third-party/patches",
];

/// The native build profiles the store understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeBuildProfile {
    Core,
    CoreMemoryContract,
}

impl NativeBuildProfile {
    pub fn from_profile_text(profile_text: &str) -> Result<Self, NativeIdentityError> {
        match profile_text {
            "core" => Ok(Self::Core),
            "core+memory-contract" => Ok(Self::CoreMemoryContract),
            other => Err(NativeIdentityError::UnsupportedProfile(other.to_owned())),
        }
    }

    pub fn as_profile_text(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::CoreMemoryContract => "core+memory-contract",
        }
    }
}

/// Whether the identity covers only tracked sources or the full hosted
/// toolchain environment. Source-only stays portable so lightweight
/// contexts (bindgen header provisioning) can coordinate without a
/// toolchain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityScope {
    SourceOnly,
    Full,
}

/// Everything the identity computation needs, stated explicitly so callers
/// cannot accidentally inherit ambient environment: the Cargo build scripts
/// must fingerprint the real toolchain, never fixture overrides.
#[derive(Debug)]
pub struct FingerprintRequest<'a> {
    pub repository_root: &'a Path,
    pub profile: NativeBuildProfile,
    pub scope: IdentityScope,
    /// Toolchain identity overrides (fixture determinism) and the full-mode
    /// target triple. `None` entries mean "probe the real toolchain".
    pub overrides: ToolchainOverrides,
}

#[derive(Debug, PartialEq, Eq)]
pub enum NativeIdentityError {
    UnsupportedProfile(String),
    MissingTargetTriple,
    UnsupportedBuildType(String),
    TrackedInputsUnavailable,
    TrackedInputMissing(String),
    GitEnumeration(String),
    ManifestDigest(String),
    Probe(String),
}

impl std::fmt::Display for NativeIdentityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedProfile(profile) => {
                write!(formatter, "unsupported native build profile: {profile}")
            }
            Self::MissingTargetTriple => {
                write!(
                    formatter,
                    "TARGET or ASTRONOMICAL_NATIVE_IDENTITY_TARGET is required"
                )
            }
            Self::UnsupportedBuildType(build_type) => {
                write!(
                    formatter,
                    "only the Release native build type is supported, got {build_type}"
                )
            }
            Self::TrackedInputsUnavailable => {
                write!(formatter, "no tracked native build inputs were found")
            }
            Self::TrackedInputMissing(tracked_path) => {
                write!(
                    formatter,
                    "tracked native build input is unavailable: {tracked_path}"
                )
            }
            Self::GitEnumeration(message) => {
                write!(
                    formatter,
                    "tracked native input enumeration failed: {message}"
                )
            }
            Self::ManifestDigest(message) => {
                write!(
                    formatter,
                    "native build identity digest is invalid: {message}"
                )
            }
            Self::Probe(message) => write!(formatter, "{message}"),
        }
    }
}

/// Computes the 64-hex-character native build identity. The manifest is
/// assembled in a fixed order (schema, profile, sorted source digests, then
/// the toolchain lines under their historical `compatibility` label) so the
/// digest is reproducible byte-for-byte.
pub fn native_build_identity(
    request: &FingerprintRequest<'_>,
) -> Result<String, NativeIdentityError> {
    native_build_identity_with_probe_values(request).map(|(identity, _)| identity)
}

/// Single-pass identity: captures the toolchain probes once, composes and
/// hashes the manifest from that capture, and returns the values alongside
/// the digest so `--explain` can log exactly the hashed inputs. The two-pass
/// shape this replaces hashed one capture and logged another, which left a
/// blind spot where an identity flap could never be explained from the log.
pub fn native_build_identity_with_probe_values(
    request: &FingerprintRequest<'_>,
) -> Result<(String, Vec<probes::ProbeValue>), NativeIdentityError> {
    let started_at = std::time::Instant::now();
    eprintln!(
        "[native-build-cache-fingerprint] status=start source_only={} profile={}",
        request.scope == IdentityScope::SourceOnly,
        request.profile.as_profile_text()
    );

    let mut manifest = format!(
        "schema\t{MANIFEST_SCHEMA_VERSION}\nprofile\t{}\n",
        request.profile.as_profile_text()
    );

    let tracked_paths = enumerate_tracked_source_paths(request.repository_root)?;
    for tracked_path in tracked_paths {
        let source_path = request.repository_root.join(&tracked_path);
        let source_bytes = std::fs::read(&source_path).map_err(|_| {
            NativeIdentityError::TrackedInputMissing(tracked_path.display().to_string())
        })?;
        let source_digest = digest_bytes(&source_bytes);
        manifest.push_str(&format!(
            "source\t{source_digest}\t{}\n",
            tracked_path.display()
        ));
    }

    let probe_values = if request.scope == IdentityScope::Full {
        let captured_values = probes::toolchain_probe_values(request)?;
        manifest.push_str(&probes::manifest_lines_from_probe_values(&captured_values));
        captured_values
    } else {
        Vec::new()
    };

    let identity_digest = digest_bytes(manifest.as_bytes());
    validate_identity_digest(&identity_digest)?;
    eprintln!(
        "[native-build-cache-fingerprint] status=success elapsed_seconds={}",
        started_at.elapsed().as_secs()
    );
    Ok((identity_digest, probe_values))
}

/// Enumerates the committed files under the source-identity pathspecs,
/// byte-sorted like `LC_ALL=C sort` so the manifest order is stable across
/// machines. Tracked-only is deliberate: hosted CI checkouts are always
/// clean, so this changes nothing there, and the native CMakeLists
/// references its sources explicitly (no GLOB), so an unreferenced scratch
/// file on a developer machine is not a build input and must not churn the
/// machine-wide store key.
pub fn enumerate_tracked_source_paths(
    repository_root: &Path,
) -> Result<Vec<PathBuf>, NativeIdentityError> {
    let mut git_enumeration = Command::new("git");
    git_enumeration
        .arg("-C")
        .arg(repository_root)
        .args(["ls-files", "--cached", "--"])
        .args(SOURCE_IDENTITY_PATHSPECS);
    let git_output = git_enumeration
        .output()
        .map_err(|error| NativeIdentityError::GitEnumeration(error.to_string()))?;
    if !git_output.status.success() {
        return Err(NativeIdentityError::GitEnumeration(format!(
            "git ls-files exited with {}: {}",
            git_output.status,
            String::from_utf8_lossy(&git_output.stderr).trim()
        )));
    }
    let tracked_paths: Vec<PathBuf> = String::from_utf8_lossy(&git_output.stdout)
        .lines()
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .collect();
    if tracked_paths.is_empty() {
        return Err(NativeIdentityError::TrackedInputsUnavailable);
    }
    let mut sorted_paths = tracked_paths;
    sorted_paths.sort();
    Ok(sorted_paths)
}

/// SHA-256 over raw bytes, lowercase hex — the shared digest primitive for
/// both source files and manifest folding.
pub fn digest_bytes(input_bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input_bytes);
    let digest_bytes = hasher.finalize();
    let mut digest_text = String::with_capacity(digest_bytes.len() * 2);
    for digest_byte in digest_bytes {
        digest_text.push_str(&format!("{digest_byte:02x}"));
    }
    digest_text
}

fn validate_identity_digest(identity_digest: &str) -> Result<(), NativeIdentityError> {
    let is_sixty_four_lowercase_hex = identity_digest.len() == 64
        && identity_digest
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    if is_sixty_four_lowercase_hex {
        Ok(())
    } else {
        Err(NativeIdentityError::ManifestDigest(
            "expected 64 lowercase hex characters".to_owned(),
        ))
    }
}
