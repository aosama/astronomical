//! Which files the duplication measure reads, whether each is production or
//! test code, and which area of the workspace it belongs to. Every function
//! here works on repository-relative paths with `/` separators, so none of it
//! touches the filesystem.

/// Crates that hold several model families side by side, with how many
/// folders below the crate's own folder the family's folder sits. Reporting
/// them as one area would hide the copies between their families, which are
/// what issue #2473 is after: `qwen35/src/qwen3_5_text` names its area
/// two folders down, and `speech/src/speech_models/kokoro` three.
const FAMILY_HOLDING_CRATE_AREA_OFFSETS: [(&str, usize); 4] = [
    ("qwen35", 2),
    ("image-generation", 2),
    ("speech", 3),
    ("decision-models", 3),
];

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

/// The area a file is reported under: its crate or app (`crates/foundation/config`,
/// `apps/daemon`), the family folder inside a family-holding crate
/// (`crates/families/text/qwen35/qwen3_5_text`), or `xtask` for the tooling crate.
pub(crate) fn area_of(relative_path: &str) -> String {
    let path_components: Vec<&str> = relative_path.split('/').collect();
    if path_components.first() == Some(&"xtask") {
        return "xtask".to_owned();
    }
    let owning_depth =
        crate_folder_index(&path_components).map_or(2, |crate_index| crate_index + 1);
    let owning_folder = match path_components.get(..owning_depth) {
        Some(owning_components) => owning_components.join("/"),
        None => relative_path.to_owned(),
    };
    match family_folder_name(&path_components) {
        Some(family_folder) => format!("{owning_folder}/{family_folder}"),
        None => owning_folder,
    }
}

/// The index of a crate's own folder in a path under `crates/`: crates sit
/// one folder per layer (`crates/foundation/config`), and family crates one
/// folder deeper, per kind (`crates/families/text/qwen35`).
fn crate_folder_index(path_components: &[&str]) -> Option<usize> {
    match path_components {
        ["crates", "families", _, _, ..] => Some(3),
        ["crates", _, _, ..] => Some(2),
        _ => None,
    }
}

/// The family folder or module file name of a path inside a family-holding
/// crate, without `.rs` (`crates/families/text/qwen35/src/qwen3_5_core.rs` belongs with the
/// folder it declares); `None` for any other path, and for a file that sits
/// above the family folders.
fn family_folder_name<'path>(path_components: &[&'path str]) -> Option<&'path str> {
    let crate_index = crate_folder_index(path_components)?;
    let crate_name = path_components[crate_index];
    for (family_holding_crate, area_offset) in FAMILY_HOLDING_CRATE_AREA_OFFSETS {
        let area_depth = crate_index + area_offset;
        if crate_name == family_holding_crate && path_components.len() > area_depth {
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
