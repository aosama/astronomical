//! The live coverage contract: the provisioned pinned headers and the
//! compiled bridge inventory must agree exactly, the committed inventory
//! artifact must be current, and the platform-exclusion registry must stay
//! justified.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

use crate::mlx_c_coverage::coverage_engine::{
    BridgeInventory, Exclusions, HeaderSurface, coverage_gaps, parse_header_surface,
};

/// The documented coverage exceptions.
///
/// The platform audit that justified the current emptiness of the
/// platform-absent category: every symbol declared in the pinned MLX-C 0.7.0
/// headers — including the CUDA and distributed families — is defined in the
/// Apple Silicon native image, so nothing qualifies as platform-absent.
/// Backends that are unavailable at runtime still bridge and return MLX
/// errors.
const COVERAGE_EXCLUSIONS: Exclusions = &[(
    "mlx_error",
    "error.h declares mlx_error as a C preprocessor variadic macro that forwards to the internal _mlx_error helper, so no function declaration exists for bindgen or the header parser to see; raw.rs declares the variadic signature by hand and the build script appends it to the inventory.",
)];

/// Resolves the provisioned bindgen header extraction directory exactly the
/// way the mlx-c-rust build script does.
fn resolve_header_directory() -> Result<PathBuf, String> {
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("repository root resolution failed: {error}"))?;
    let mut identity_command =
        Command::new(repository_root.join("scripts/ci/native-build-cache-fingerprint.sh"));
    identity_command
        .arg("--source-only")
        .arg("--profile")
        .arg("core")
        .arg(&repository_root);
    for override_variable in [
        "ASTRONOMICAL_NATIVE_IDENTITY_XCODE",
        "ASTRONOMICAL_NATIVE_IDENTITY_SDK",
        "ASTRONOMICAL_NATIVE_IDENTITY_CLANG",
        "ASTRONOMICAL_NATIVE_IDENTITY_CMAKE",
        "ASTRONOMICAL_NATIVE_IDENTITY_RUSTC",
        "ASTRONOMICAL_NATIVE_IDENTITY_TARGET",
    ] {
        identity_command.env_remove(override_variable);
    }
    let identity_output = identity_command
        .output()
        .map_err(|error| format!("native identity fingerprint failed to run: {error}"))?;
    if !identity_output.status.success() {
        return Err(format!(
            "native identity fingerprint failed: {}",
            String::from_utf8_lossy(&identity_output.stderr)
        ));
    }
    let identity = String::from_utf8_lossy(&identity_output.stdout)
        .trim()
        .to_owned();
    let cache_directory = std::env::var_os("ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_directory().join("Library/Caches/Astronomical/native-dependencies")
        });
    let header_directory = cache_directory
        .join("bindgen-headers")
        .join(&identity)
        .join("mlx_c-src/mlx/c");
    if !header_directory.is_dir() {
        return Err(format!(
            "no provisioned bindgen headers for the current source identity at {header_directory:?}; run scripts/provision-bindgen-headers.sh"
        ));
    }
    Ok(header_directory)
}

fn home_directory() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME selects the native dependency cache")
}

/// Parses every top-level public header in the extraction directory.
fn parse_pinned_surface(header_directory: &std::path::Path) -> Result<HeaderSurface, String> {
    let mut surface = HeaderSurface::default();
    let mut header_paths: Vec<PathBuf> = std::fs::read_dir(header_directory)
        .map_err(|error| format!("header directory unreadable: {error}"))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "h"))
        .collect();
    header_paths.sort();
    println!("coverage contract: parsing {} headers", header_paths.len());
    for header_path in &header_paths {
        let header_text = std::fs::read_to_string(header_path)
            .map_err(|error| format!("header {:?} unreadable: {error}", header_path))?;
        let header_surface = parse_header_surface(&header_text);
        surface.functions.extend(header_surface.functions);
        surface.types.extend(header_surface.types);
    }
    println!(
        "coverage contract: parsed {} public functions and {} public types",
        surface.functions.len(),
        surface.types.len()
    );
    Ok(surface)
}

/// The bridge inventory the mlx-c-rust build script emitted.
fn compiled_bridge_inventory() -> BridgeInventory {
    BridgeInventory {
        functions: astronomical_mlx_c_rust::raw::BRIDGED_FUNCTIONS
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        types: astronomical_mlx_c_rust::raw::BRIDGED_TYPES
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
    }
}

/// Renders the committed inventory artifact from live data.
fn render_inventory(surface: &HeaderSurface, bridged: &BridgeInventory) -> String {
    let mut document = String::new();
    document.push_str("# MLX-C API coverage inventory\n\n");
    document.push_str("Generated by the coverage contract from the provisioned pinned headers.\n");
    document.push_str("Refresh with `scripts/generate-mlx-c-coverage-inventory.sh`.\n\n");
    document.push_str(&format!(
        "- Public upstream functions: {}\n- Public upstream types: {}\n- Bridged functions: {}\n- Bridged types: {}\n- Exclusions: {}\n\n",
        surface.functions.len(),
        surface.types.len(),
        bridged.functions.len(),
        bridged.types.len(),
        COVERAGE_EXCLUSIONS.len(),
    ));
    let exclusion_names: BTreeSet<String> = COVERAGE_EXCLUSIONS
        .iter()
        .map(|(name, _reason)| (*name).to_owned())
        .collect();
    document.push_str("## Functions\n\n");
    document.push_str("| upstream symbol | bridge state |\n|---|---|\n");
    for name in &surface.functions {
        let state = if bridged.functions.contains(name) {
            "bridged"
        } else if exclusion_names.contains(name) {
            "excluded (see registry)"
        } else {
            "UNBRIDGED"
        };
        document.push_str(&format!("| `{name}` | {state} |\n"));
    }
    document.push_str("\n## Types\n\n| upstream symbol | bridge state |\n|---|---|\n");
    for name in &surface.types {
        let state = if bridged.types.contains(name) {
            "bridged"
        } else if exclusion_names.contains(name) {
            "excluded (see registry)"
        } else {
            "UNBRIDGED"
        };
        document.push_str(&format!("| `{name}` | {state} |\n"));
    }
    document.push_str("\n## Exclusion registry\n\n");
    document.push_str("| symbol | justification |\n|---|---|\n");
    for (name, reason) in COVERAGE_EXCLUSIONS {
        document.push_str(&format!("| `{name}` | {reason} |\n"));
    }
    document
}

fn inventory_artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../mlx-c-rust/COVERAGE_INVENTORY.md")
}

#[test]
fn should_bridge_every_public_header_function() {
    let header_directory = resolve_header_directory().expect("provisioned headers resolve");
    let surface = parse_pinned_surface(&header_directory).expect("headers parse");
    let bridged = compiled_bridge_inventory();

    println!(
        "coverage contract: bridge carries {} functions and {} types",
        bridged.functions.len(),
        bridged.types.len()
    );
    let gaps = coverage_gaps(&surface, &bridged, COVERAGE_EXCLUSIONS);
    assert!(
        gaps.unbridged_functions.is_empty() && gaps.stale_functions.is_empty(),
        "function coverage contract failed:\n{}",
        gaps.report()
    );
}

#[test]
fn should_bridge_every_public_header_type() {
    let header_directory = resolve_header_directory().expect("provisioned headers resolve");
    let surface = parse_pinned_surface(&header_directory).expect("headers parse");
    let bridged = compiled_bridge_inventory();

    let gaps = coverage_gaps(&surface, &bridged, COVERAGE_EXCLUSIONS);
    assert!(
        gaps.unbridged_types.is_empty() && gaps.stale_types.is_empty(),
        "type coverage contract failed:\n{}",
        gaps.report()
    );
}

#[test]
fn should_keep_the_exclusions_registry_justified() {
    for (name, reason) in COVERAGE_EXCLUSIONS {
        assert!(
            !reason.trim().is_empty(),
            "exclusion {name} needs a written justification"
        );
    }
}

#[test]
fn should_keep_the_committed_coverage_inventory_current() {
    let header_directory = resolve_header_directory().expect("provisioned headers resolve");
    let surface = parse_pinned_surface(&header_directory).expect("headers parse");
    let bridged = compiled_bridge_inventory();
    let rendered_inventory = render_inventory(&surface, &bridged);

    let artifact_path = inventory_artifact_path();
    if let Some(output_path) = std::env::var_os("MLX_C_COVERAGE_INVENTORY_PATH") {
        std::fs::write(&output_path, &rendered_inventory)
            .expect("coverage inventory artifact writes");
        println!("coverage contract: refreshed inventory at {output_path:?}");
        return;
    }

    let committed_inventory = std::fs::read_to_string(&artifact_path).unwrap_or_else(|error| {
        panic!(
            "the committed coverage inventory is missing at {:?} ({}); run scripts/generate-mlx-c-coverage-inventory.sh",
            artifact_path, error
        )
    });
    assert_eq!(
        committed_inventory, rendered_inventory,
        "the committed coverage inventory is stale; run scripts/generate-mlx-c-coverage-inventory.sh"
    );
}
