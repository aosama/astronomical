//! What the folder-derived architecture rules' tests share: a `cargo
//! metadata` document built from packages placed in folders.

/// Minimal `cargo metadata` document whose packages are the given
/// `(name, manifest folder below the workspace, dependency names)` tuples.
pub(crate) fn metadata_document_for(packages: &[(&str, &str, Vec<&str>)]) -> String {
    let mut package_json_entries = Vec::new();
    for (package_name, manifest_folder, dependency_names) in packages {
        let mut dependency_json_entries = Vec::new();
        for dependency_name in dependency_names {
            dependency_json_entries.push(format!(
                "{{\"name\":\"{dependency_name}\",\"kind\":\"normal\"}}"
            ));
        }
        package_json_entries.push(format!(
            "{{\"name\":\"{package_name}\",\"manifest_path\":\"/workspace/{manifest_folder}/Cargo.toml\",\"dependencies\":[{}]}}",
            dependency_json_entries.join(",")
        ));
    }
    format!("{{\"packages\":[{}]}}", package_json_entries.join(","))
}
