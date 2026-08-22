//! Read-only REST projection of the validated release catalog.

use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;

use crate::application::ApplicationState;

pub(crate) fn library_catalog_routes() -> Router<ApplicationState> {
    Router::new().route("/v1/library/catalog", get(get_library_catalog))
}

async fn get_library_catalog(
    State(application_state): State<ApplicationState>,
) -> Json<LibraryCatalogResponse> {
    let current_job = match application_state.library_download_coordinator.as_ref() {
        Some(download_coordinator) => download_coordinator.current_job().await.ok().flatten(),
        None => None,
    };
    let discovered_models = application_state.discovered_models_snapshot();
    let validated_publications = match application_state.library_download_coordinator.as_ref() {
        Some(download_coordinator) => download_coordinator.validated_publications_snapshot().await,
        None => Default::default(),
    };
    let mut entries = Vec::with_capacity(application_state.download_catalog.entry_count());
    for catalog_entry in application_state.download_catalog.entries() {
        let huggingface_id = catalog_entry.huggingface_id();
        let discovered_model = discovered_models.iter().find(|model| {
            model.provider_model_id.as_deref() == Some(huggingface_id)
                && model.revision == catalog_entry.revision()
        });
        let has_validated_publication = validated_publications.contains(huggingface_id);
        let is_ready = discovered_model.is_some() || has_validated_publication;
        let destination_directory = discovered_model
            .map(|model| model.model_directory.display().to_string())
            .or_else(|| {
                application_state.library_download_coordinator.as_ref().map(
                    |download_coordinator| {
                        download_coordinator
                            .destination_directory(huggingface_id)
                            .display()
                            .to_string()
                    },
                )
            });
        let requestable_model_id = is_ready.then(|| {
            discovered_model.map_or_else(
                || requestable_model_id_from_huggingface_id(huggingface_id),
                |model| model.model_id.clone(),
            )
        });
        entries.push(LibraryCatalogEntryResponse::from_entry(
            catalog_entry,
            is_ready,
            destination_directory,
            requestable_model_id,
            current_job.as_ref(),
        ));
    }
    Json(LibraryCatalogResponse {
        schema_version: application_state.download_catalog.schema_version(),
        entries,
    })
}

#[derive(Debug, Serialize)]
struct LibraryCatalogResponse {
    schema_version: u32,
    entries: Vec<LibraryCatalogEntryResponse>,
}

#[derive(Debug, Serialize)]
struct LibraryCatalogEntryResponse {
    huggingface_id: String,
    revision: String,
    display_name: String,
    family: &'static str,
    approximate_size_bytes: u64,
    public: bool,
    ready_on_this_mac: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    destination_directory: Option<String>,
    download_state: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    architecture_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upstream_license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    requestable_model_id: Option<String>,
    capabilities: LibraryCatalogCapabilitiesResponse,
}

#[derive(Debug, Default, Serialize)]
struct LibraryCatalogCapabilitiesResponse {
    supports_reasoning: bool,
    supports_vision: bool,
    supports_tool_calls: bool,
    supports_image_generation: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

impl LibraryCatalogEntryResponse {
    fn from_entry(
        catalog_entry: &super::DownloadCatalogEntry,
        ready_on_this_mac: bool,
        destination_directory: Option<String>,
        requestable_model_id: Option<String>,
        current_job: Option<&super::DownloadJob>,
    ) -> Self {
        let capabilities = catalog_entry.capabilities();
        Self {
            huggingface_id: catalog_entry.huggingface_id().to_owned(),
            revision: catalog_entry.revision().to_owned(),
            display_name: catalog_entry.display_name().to_owned(),
            family: catalog_entry.family().as_str(),
            approximate_size_bytes: catalog_entry.approximate_size_bytes(),
            public: true,
            ready_on_this_mac,
            destination_directory,
            download_state: current_job
                .filter(|job| {
                    job.huggingface_id() == catalog_entry.huggingface_id() && !ready_on_this_mac
                })
                .map(|job| job.state().as_str()),
            description: catalog_entry.description().map(str::to_owned),
            quantization_label: catalog_entry.quantization_label().map(str::to_owned),
            architecture_summary: catalog_entry.architecture_summary().map(str::to_owned),
            upstream_license: catalog_entry.upstream_license().map(str::to_owned),
            requestable_model_id,
            capabilities: LibraryCatalogCapabilitiesResponse {
                supports_reasoning: capabilities.supports_reasoning,
                supports_vision: capabilities.supports_vision,
                supports_tool_calls: capabilities.supports_tool_calls,
                supports_image_generation: capabilities.supports_image_generation,
                context_window: capabilities.context_window,
                max_output_tokens: capabilities.max_output_tokens,
            },
        }
    }
}

/// Derives the local requestable model ID from the Hugging Face identity's leaf segment.
/// Discovery publishes Library models under their leaf directory name, so "org/Model-Name"
/// becomes requestable as "Model-Name".
fn requestable_model_id_from_huggingface_id(huggingface_id: &str) -> String {
    huggingface_id
        .rsplit('/')
        .next()
        .unwrap_or(huggingface_id)
        .to_owned()
}
