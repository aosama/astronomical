//! The catalog identity a speed measurement records beside a ledger row.
//!
//! The release speed ledger is the record of how fast a release replies. The
//! catalog no longer copies those figures, so a measurement looks a model up
//! here only to store the repository and revision the ledger compares later.

use serde_json::Value;

/// A catalog entry's identity: its Hugging Face repository and the revision
/// the catalog pins.
#[derive(Debug, Eq, PartialEq)]
pub struct CatalogIdentity {
    pub huggingface_id: String,
    pub revision: Option<String>,
}

/// The identity of the catalog entry whose model ID (the leaf of its
/// repository) is this one.
#[must_use]
pub fn catalog_identity(catalog_text: &str, model_id: &str) -> Option<CatalogIdentity> {
    let catalog_document: Value = serde_json::from_str(catalog_text).ok()?;
    let catalog_entry = catalog_document["entries"]
        .as_array()?
        .iter()
        .find(|catalog_entry| entry_model_id(catalog_entry) == Some(model_id))?;
    Some(CatalogIdentity {
        huggingface_id: catalog_entry["huggingface_id"].as_str()?.to_owned(),
        revision: catalog_entry["revision"].as_str().map(str::to_owned),
    })
}

fn entry_model_id(catalog_entry: &Value) -> Option<&str> {
    catalog_entry["huggingface_id"]
        .as_str()
        .and_then(|huggingface_id| huggingface_id.rsplit('/').next())
}

#[cfg(test)]
#[path = "speed_catalog_tests.rs"]
mod speed_catalog_tests;
