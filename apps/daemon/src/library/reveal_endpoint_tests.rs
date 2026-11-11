//! Folder resolution for the Library's Show in Finder control. Only the pure
//! resolution is exercised here, so no test ever opens a Finder window.

use std::path::PathBuf;

use super::{LibraryRevealRefusal, ready_model_folder};
use crate::DownloadCatalog;
use crate::library::CatalogEntryProjection;

const SINGLE_ENTRY_CATALOG_JSON: &str = r#"{
    "schema_version": 5,
    "entries": [{
        "huggingface_id": "astronomical-test/example-qwen",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "family": "qwen3_5",
        "approximate_size_bytes": 4000000000,
        "approximate_memory_bytes": 5000000000,
        "role": "chat",
        "public": true
    }]
}"#;

fn example_catalog() -> DownloadCatalog {
    DownloadCatalog::parse_json(SINGLE_ENTRY_CATALOG_JSON)
        .expect("the example catalog should parse")
}

fn projection_of<'catalog>(
    download_catalog: &'catalog DownloadCatalog,
    ready_on_this_mac: bool,
    discovered_model_directory: Option<&str>,
) -> CatalogEntryProjection<'catalog> {
    let catalog_entry = &download_catalog.entries()[0];
    CatalogEntryProjection {
        catalog_entry,
        ready_on_this_mac,
        discovered_model_directory: discovered_model_directory.map(str::to_owned),
        requestable_model_id: None,
        download_state: None,
        capabilities: catalog_entry.capabilities().clone(),
    }
}

#[test]
fn should_refuse_an_identity_that_is_not_in_the_catalog() {
    assert_eq!(
        ready_model_folder(None, Some(PathBuf::from("/library/unknown"))),
        Err(LibraryRevealRefusal::UnknownModel)
    );
}

#[test]
fn should_refuse_a_catalog_model_that_is_not_on_this_mac() {
    let download_catalog = example_catalog();
    let not_downloaded = projection_of(&download_catalog, false, None);
    assert_eq!(
        ready_model_folder(
            Some(&not_downloaded),
            Some(PathBuf::from("/library/example"))
        ),
        Err(LibraryRevealRefusal::NotOnThisMac),
        "a model that is not ready must not reveal its would-be download folder"
    );
}

#[test]
fn should_reveal_the_folder_discovery_found_before_the_download_folder() {
    let download_catalog = example_catalog();
    let discovered = projection_of(&download_catalog, true, Some("/models/example-qwen"));
    assert_eq!(
        ready_model_folder(Some(&discovered), Some(PathBuf::from("/library/example"))),
        Ok(PathBuf::from("/models/example-qwen"))
    );
}

#[test]
fn should_reveal_the_published_download_folder_for_a_model_ready_only_by_download() {
    let download_catalog = example_catalog();
    let downloaded = projection_of(&download_catalog, true, None);
    assert_eq!(
        ready_model_folder(Some(&downloaded), Some(PathBuf::from("/library/example"))),
        Ok(PathBuf::from("/library/example"))
    );
    assert_eq!(
        ready_model_folder(Some(&downloaded), None),
        Err(LibraryRevealRefusal::NotOnThisMac),
        "without a download coordinator there is no published folder to reveal"
    );
}
