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

/// The discovered copy of `huggingface_id`, preferring the catalog's exact
/// revision, then any copy whose provenance names the repository, then a copy
/// without provenance whose `<publisher>/<model>` folder spells it.
pub(crate) fn find_discovered_copy<'a>(
    huggingface_id: &str,
    catalog_revision: &str,
    discovered_models: &'a [DiscoveredModel],
) -> Option<&'a DiscoveredModel> {
    find_exact_copy(huggingface_id, catalog_revision, discovered_models).or_else(|| {
        discovered_models
            .iter()
            .find(|discovered_model| is_copy_of_repository(discovered_model, huggingface_id))
    })
}

/// The copy whose provenance names `huggingface_id` at the catalog's
/// revision: the only copy the download coordinator treats as installed,
/// because a folder without provenance at the download destination may be a
/// publication that was interrupted and still needs its manifest check.
pub(crate) fn find_exact_copy<'a>(
    huggingface_id: &str,
    catalog_revision: &str,
    discovered_models: &'a [DiscoveredModel],
) -> Option<&'a DiscoveredModel> {
    discovered_models.iter().find(|discovered_model| {
        discovered_model.provider_model_id.as_deref() == Some(huggingface_id)
            && discovered_model.revision == catalog_revision
    })
}

/// Whether `discovered_model` is a copy of `huggingface_id` at any revision.
/// Provenance, when present, is the only evidence: a copy that names another
/// repository is another model even if its folder looks the same.
fn is_copy_of_repository(discovered_model: &DiscoveredModel, huggingface_id: &str) -> bool {
    match discovered_model.provider_model_id.as_deref() {
        Some(provider_model_id) => provider_model_id == huggingface_id,
        None => {
            folder_repository_identity(&discovered_model.model_directory).as_deref()
                == Some(huggingface_id)
                && config::leaf_model_id(huggingface_id) == discovered_model.model_id
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
