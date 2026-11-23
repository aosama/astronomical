//! Which files the duplication measure reads, whether each is production or
//! test code, and which area of the workspace it belongs to. Every function
//! here works on repository-relative paths with `/` separators, so none of it
//! touches the filesystem.

/// The top-level folders whose Rust sources are measured.
pub(crate) const MEASURED_ROOT_DIRECTORIES: [&str; 4] = ["crates", "apps", "verification", "xtask"];

/// Folders never descended into: build output, and vendored or pinned code
/// this repository does not author.
const SKIPPED_DIRECTORY_NAMES: [&str; 2] = ["target", "third-party"];

/// Crates that hold several model families side by side, with the path depth
/// of the folder that names the family. Reporting them as one area would hide
/// the copies between their families, which are what issue #2473 is after.
/// The depth counts path components from the repository root:
/// `crates/qwen35/src/qwen3_5_streaming` names its area at depth 3, and
/// `crates/speech/src/speech_models/kokoro` at depth 4.
const FAMILY_HOLDING_CRATE_AREA_DEPTHS: [(&str, usize); 4] = [
    ("qwen35", 3),
    ("image-generation", 3),
    ("speech", 4),
    ("decision-models", 4),
];

/// Whether the walk should skip a directory with this name.
pub(crate) fn is_skipped_directory_name(directory_name: &str) -> bool {
    SKIPPED_DIRECTORY_NAMES.contains(&directory_name)
}

/// Production code is any Rust file whose name mentions neither "test" nor
/// "fixture" and that does not sit under a `tests` folder.
pub(crate) fn is_production_source(relative_path: &str) -> bool {
    let path_components: Vec<&str> = relative_path.split('/').collect();
    let Some((file_name, parent_components)) = path_components.split_last() else {
        return false;
    };
    !file_name.contains("test")
        && !file_name.contains("fixture")
        && !parent_components.contains(&"tests")
}

/// The area a file is reported under: its crate or app (`crates/config`,
/// `apps/daemon`), the family folder inside a family-holding crate
/// (`crates/qwen35/qwen3_5_resident`), or `xtask` for the tooling crate.
pub(crate) fn area_of(relative_path: &str) -> String {
    let path_components: Vec<&str> = relative_path.split('/').collect();
    if path_components.first() == Some(&"xtask") {
        return "xtask".to_owned();
    }
    let owning_folder = match path_components.get(..2) {
        Some(owning_components) => owning_components.join("/"),
        None => relative_path.to_owned(),
    };
    match family_folder_name(&path_components) {
        Some(family_folder) => format!("{owning_folder}/{family_folder}"),
        None => owning_folder,
    }
}

/// The family folder or module file name of a path inside a family-holding
/// crate, without `.rs` (`crates/qwen35/src/qwen3_5_core.rs` belongs with the
/// folder it declares); `None` for any other path, and for a file that sits
/// above the family folders.
fn family_folder_name<'path>(path_components: &[&'path str]) -> Option<&'path str> {
    if path_components.first() != Some(&"crates") {
        return None;
    }
    let crate_name = path_components.get(1)?;
    for (family_holding_crate, area_depth) in FAMILY_HOLDING_CRATE_AREA_DEPTHS {
        if *crate_name == family_holding_crate && path_components.len() > area_depth {
            let family_component = path_components[area_depth];
            return Some(
                family_component
                    .strip_suffix(".rs")
                    .unwrap_or(family_component),
            );
        }
    }
    None
}
