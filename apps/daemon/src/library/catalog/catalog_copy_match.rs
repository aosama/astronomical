//! Which discovered copy on this Mac stands for a catalog entry.
//!
//! A copy the Models page downloaded records its repository and revision in
//! its provenance file, so it matches its entry exactly. A copy that arrived
//! another way (copied by hand into a models folder, or a build from before
//! the catalog moved to a newer revision) is still that model, and the engine
//! serves it under the same name. Such a copy is shown as on this Mac, so the
//! page lists what every client of the engine sees, but it is not "ready" in
//! the download coordinator's sense until Get verifies and adopts it.

use std::path::Path;

use config::DiscoveredModel;

/// The discovered copy of a catalog entry known by `catalog_entry_names` (its
/// current Hugging Face name first, then any names it had before a rename),
/// preferring the catalog's exact revision, then any copy whose provenance
/// names the repository, then a copy without provenance whose
/// `<publisher>/<model>` folder spells one of its names.
pub(crate) fn find_discovered_copy<'a>(
    catalog_entry_names: &[&str],
    catalog_revision: &str,
    discovered_models: &'a [DiscoveredModel],
) -> Option<&'a DiscoveredModel> {
    find_exact_copy(catalog_entry_names, catalog_revision, discovered_models).or_else(|| {
        discovered_models
            .iter()
            .find(|discovered_model| is_copy_of_repository(discovered_model, catalog_entry_names))
    })
}

/// The copy whose provenance names one of `catalog_entry_names` at the
/// catalog's revision: the only copy the download coordinator treats as
/// installed, because a folder without provenance at the download destination
/// may be a publication that was interrupted and still needs its manifest
/// check. Hugging Face keeps a repository's commits when it is renamed, so a
/// copy downloaded under a former name at this revision holds the same files.
pub(crate) fn find_exact_copy<'a>(
    catalog_entry_names: &[&str],
    catalog_revision: &str,
    discovered_models: &'a [DiscoveredModel],
) -> Option<&'a DiscoveredModel> {
    discovered_models.iter().find(|discovered_model| {
        discovered_model
            .provider_model_id
            .as_deref()
            .is_some_and(|provider_model_id| catalog_entry_names.contains(&provider_model_id))
            && discovered_model.revision == catalog_revision
    })
}

/// Whether `discovered_model` is a copy of the entry at any revision.
/// Provenance, when present, is the only evidence: a copy that names another
/// repository is another model even if its folder looks the same.
fn is_copy_of_repository(discovered_model: &DiscoveredModel, catalog_entry_names: &[&str]) -> bool {
    match discovered_model.provider_model_id.as_deref() {
        Some(provider_model_id) => catalog_entry_names.contains(&provider_model_id),
        None => {
            let folder_identity = folder_repository_identity(&discovered_model.model_directory);
            catalog_entry_names.iter().any(|huggingface_id| {
                folder_identity.as_deref() == Some(*huggingface_id)
                    && config::leaf_model_id(huggingface_id) == discovered_model.model_id
            })
        }
    }
}

/// `<publisher>/<model>` from a model folder's last two path components, the
/// layout of the Library folders, or None when the path is shorter.
fn folder_repository_identity(model_directory: &Path) -> Option<String> {
    let model_folder_name = model_directory.file_name()?.to_str()?;
    let publisher_folder_name = model_directory.parent()?.file_name()?.to_str()?;
    Some(format!("{publisher_folder_name}/{model_folder_name}"))
}
