//! The model inventory a stub daemon reports: what is installed and what the
//! catalog offers, with one constructor per kind of model so a test names
//! the kind it needs instead of spelling out every capability flag.

/// One installed model the stub reports for `ModelsList`.
pub struct StubInstalledModel {
    pub model_id: String,
    pub context_window: Option<u32>,
    pub supports_embeddings: bool,
    pub supports_decisions: bool,
    pub supports_image_generation: bool,
    pub is_resident: bool,
}

impl StubInstalledModel {
    /// An installed text-to-image model.
    pub fn image(model_id: &str, is_resident: bool) -> Self {
        StubInstalledModel {
            model_id: model_id.to_owned(),
            context_window: None,
            supports_embeddings: false,
            supports_decisions: false,
            supports_image_generation: true,
            is_resident,
        }
    }

    /// A chat-capable installed model.
    pub fn chat(model_id: &str, is_resident: bool) -> Self {
        StubInstalledModel {
            model_id: model_id.to_owned(),
            context_window: Some(32_768),
            supports_embeddings: false,
            supports_decisions: false,
            supports_image_generation: false,
            is_resident,
        }
    }

    /// An installed decision model.
    pub fn decisions(model_id: &str, is_resident: bool) -> Self {
        StubInstalledModel {
            model_id: model_id.to_owned(),
            context_window: None,
            supports_embeddings: false,
            supports_decisions: true,
            supports_image_generation: false,
            is_resident,
        }
    }

    /// An embeddings-only installed model.
    pub fn embeddings(model_id: &str, is_resident: bool) -> Self {
        StubInstalledModel {
            model_id: model_id.to_owned(),
            context_window: None,
            supports_embeddings: true,
            supports_decisions: false,
            supports_image_generation: false,
            is_resident,
        }
    }
}

/// One catalog entry the stub reports for `Catalog`.
#[derive(Clone)]
pub struct StubCatalogEntry {
    pub huggingface_id: String,
    pub requestable_model_id: Option<String>,
    pub ready_on_this_mac: bool,
    pub download_state: Option<String>,
    pub context_window: Option<u32>,
    pub supports_embeddings: bool,
    pub supports_decisions: bool,
    pub supports_image_generation: bool,
}

impl StubCatalogEntry {
    /// A downloadable text-to-image model.
    pub fn image(
        huggingface_id: &str,
        requestable_model_id: &str,
        ready_on_this_mac: bool,
    ) -> Self {
        StubCatalogEntry {
            huggingface_id: huggingface_id.to_owned(),
            requestable_model_id: Some(requestable_model_id.to_owned()),
            ready_on_this_mac,
            download_state: None,
            context_window: None,
            supports_embeddings: false,
            supports_decisions: false,
            supports_image_generation: true,
        }
    }

    /// A downloadable chat model.
    pub fn chat(huggingface_id: &str, requestable_model_id: &str, ready_on_this_mac: bool) -> Self {
        StubCatalogEntry {
            huggingface_id: huggingface_id.to_owned(),
            requestable_model_id: Some(requestable_model_id.to_owned()),
            ready_on_this_mac,
            download_state: None,
            context_window: Some(32_768),
            supports_embeddings: false,
            supports_decisions: false,
            supports_image_generation: false,
        }
    }

    /// A downloadable decision model.
    pub fn decisions(
        huggingface_id: &str,
        requestable_model_id: &str,
        ready_on_this_mac: bool,
    ) -> Self {
        StubCatalogEntry {
            huggingface_id: huggingface_id.to_owned(),
            requestable_model_id: Some(requestable_model_id.to_owned()),
            ready_on_this_mac,
            download_state: None,
            context_window: None,
            supports_embeddings: false,
            supports_decisions: true,
            supports_image_generation: false,
        }
    }

    /// A downloadable embeddings-only model.
    pub fn embeddings(
        huggingface_id: &str,
        requestable_model_id: &str,
        ready_on_this_mac: bool,
    ) -> Self {
        StubCatalogEntry {
            huggingface_id: huggingface_id.to_owned(),
            requestable_model_id: Some(requestable_model_id.to_owned()),
            ready_on_this_mac,
            download_state: None,
            context_window: None,
            supports_embeddings: true,
            supports_decisions: false,
            supports_image_generation: false,
        }
    }
}
