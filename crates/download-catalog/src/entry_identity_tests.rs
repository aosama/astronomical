use super::{SeenEntryIdentities, claim_former_identities};
use crate::download_catalog::{DownloadCatalog, DownloadCatalogError};

const VALID_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn seen_with(huggingface_id: &str) -> SeenEntryIdentities {
    let mut seen_identities = SeenEntryIdentities::default();
    seen_identities
        .claim(huggingface_id, 0)
        .expect("the first name is free");
    seen_identities
}

fn catalog_json_with_entries(entries_json: &str) -> String {
    format!(r#"{{"schema_version": 5, "entries": [{entries_json}]}}"#)
}

fn entry_json(huggingface_id: &str, former_huggingface_ids_json: &str) -> String {
    format!(
        r#"{{"huggingface_id": "{huggingface_id}", "former_huggingface_ids": {former_huggingface_ids_json},
            "revision": "{VALID_REVISION}", "family": "qwen3_5", "approximate_size_bytes": 4000,
            "approximate_memory_bytes": 5000, "role": "chat", "public": true}}"#
    )
}

#[test]
fn should_claim_former_names_that_no_entry_uses() {
    let mut seen_identities = seen_with("publisher/Model-4bit-vision");

    let claimed = claim_former_identities(
        &["publisher/Model-4bit".to_owned()],
        0,
        &mut seen_identities,
    );

    assert!(claimed.is_ok());
}

#[test]
fn should_accept_an_entry_without_former_names() {
    let mut seen_identities = seen_with("publisher/Model-4bit-vision");

    assert!(claim_former_identities(&[], 0, &mut seen_identities).is_ok());
}

#[test]
fn should_reject_a_former_name_that_is_not_a_hugging_face_identity() {
    let mut seen_identities = seen_with("publisher/Model-4bit-vision");

    let claimed = claim_former_identities(&["no-slash".to_owned()], 3, &mut seen_identities);

    assert!(matches!(
        claimed,
        Err(DownloadCatalogError::InvalidFormerHuggingFaceId { entry_index: 3 })
    ));
}

#[test]
fn should_reject_a_former_name_equal_to_the_current_name_ignoring_case() {
    let mut seen_identities = seen_with("publisher/Model-4bit-vision");

    let claimed = claim_former_identities(
        &["Publisher/model-4bit-VISION".to_owned()],
        0,
        &mut seen_identities,
    );

    assert!(matches!(
        claimed,
        Err(DownloadCatalogError::DuplicateHuggingFaceId)
    ));
}

/// A former name that another entry uses as its own name would send an old
/// copy to two rows, so the catalog refuses it whichever entry comes first.
#[test]
fn should_reject_a_catalog_where_a_former_name_is_another_entrys_name() {
    let catalog_json = catalog_json_with_entries(&format!(
        "{}, {}",
        entry_json("publisher/Model-4bit-vision", r#"["publisher/Model-4bit"]"#),
        entry_json("publisher/Model-4bit", "[]"),
    ));

    let parsed = DownloadCatalog::parse_json(&catalog_json);

    assert!(matches!(
        parsed,
        Err(DownloadCatalogError::DuplicateHuggingFaceId)
    ));
}

#[test]
fn should_list_the_current_name_first_then_the_former_names() {
    let catalog_json = catalog_json_with_entries(&entry_json(
        "publisher/Model-4bit-vision",
        r#"["publisher/Model-4bit"]"#,
    ));

    let catalog = DownloadCatalog::parse_json(&catalog_json).expect("the catalog is valid");
    let entry_names = catalog.entries()[0].huggingface_ids().collect::<Vec<_>>();

    assert_eq!(
        entry_names,
        ["publisher/Model-4bit-vision", "publisher/Model-4bit"]
    );
}
