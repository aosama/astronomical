use super::{CatalogIdentity, catalog_identity};

#[test]
fn should_read_a_catalog_entrys_repository_and_revision() {
    let catalog_text = r#"{
        "entries": [
            {"huggingface_id": "prism-ml/Ternary-Bonsai-2-4B-mlx-2bit", "revision": "abc"}
        ]
    }"#;

    assert_eq!(
        catalog_identity(catalog_text, "Ternary-Bonsai-2-4B-mlx-2bit"),
        Some(CatalogIdentity {
            huggingface_id: "prism-ml/Ternary-Bonsai-2-4B-mlx-2bit".to_owned(),
            revision: Some("abc".to_owned()),
        })
    );
    assert_eq!(catalog_identity(catalog_text, "missing"), None);
    assert_eq!(catalog_identity("not json", "missing"), None);
}

#[test]
fn should_leave_the_revision_absent_when_the_entry_names_none() {
    let catalog_text = r#"{"entries": [{"huggingface_id": "publisher/Model"}]}"#;

    assert_eq!(
        catalog_identity(catalog_text, "Model"),
        Some(CatalogIdentity {
            huggingface_id: "publisher/Model".to_owned(),
            revision: None,
        })
    );
}
