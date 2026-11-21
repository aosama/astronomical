//! The only way a model is named in a report: its public catalog id, or
//! nothing more than the fact that it is not a catalog model.

use serde::{Serialize, Serializer};

/// The wording for a model that is not in the download catalog. A model
/// someone converted or named themselves may carry a personal name, so its
/// name never enters a report.
const NOT_IN_CATALOG_TEXT: &str = "a model not in the catalog";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CatalogModelName {
    /// A Hugging Face id that the download catalog lists, so it is public.
    Listed(String),
    /// Any other model.
    NotInCatalog,
}

impl CatalogModelName {
    /// Names `model_id` by its catalog id when the catalog lists it. Ids are
    /// compared whole and also by their last part, because the engine often
    /// knows a model only by the folder name it was downloaded into.
    pub(crate) fn from_model_id(model_id: &str, catalog_huggingface_ids: &[String]) -> Self {
        for catalog_huggingface_id in catalog_huggingface_ids {
            let catalog_leaf_name = catalog_huggingface_id
                .rsplit('/')
                .next()
                .unwrap_or(catalog_huggingface_id);
            if model_id == catalog_huggingface_id || model_id == catalog_leaf_name {
                return Self::Listed(catalog_huggingface_id.clone());
            }
        }
        Self::NotInCatalog
    }

    pub(crate) fn display_text(&self) -> &str {
        match self {
            Self::Listed(catalog_huggingface_id) => catalog_huggingface_id,
            Self::NotInCatalog => NOT_IN_CATALOG_TEXT,
        }
    }
}

impl Serialize for CatalogModelName {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.display_text())
    }
}
