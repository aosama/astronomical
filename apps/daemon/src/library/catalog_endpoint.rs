//! Read-only REST projection of the validated release catalog.

use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;

use crate::{application::ApplicationState, library::project_catalog_entries};

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
    let projections = project_catalog_entries(
        &application_state.download_catalog,
        &discovered_models,
        &validated_publications,
        current_job.as_ref(),
    );
    let mut entries = Vec::with_capacity(projections.len());
    for projection in &projections {
        let destination_directory = projection.discovered_model_directory.clone().or_else(|| {
            application_state
                .library_download_coordinator
                .as_ref()
                .map(|download_coordinator| {
                    download_coordinator
                        .destination_directory(projection.catalog_entry.huggingface_id())
                        .display()
                        .to_string()
                })
        });
        entries.push(LibraryCatalogEntryResponse::from_entry(
            projection,
            destination_directory,
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
    supports_embeddings: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

impl LibraryCatalogEntryResponse {
    fn from_entry(
        projection: &super::CatalogEntryProjection,
        destination_directory: Option<String>,
    ) -> Self {
        let catalog_entry = projection.catalog_entry;
        let capabilities = &projection.capabilities;
        Self {
            huggingface_id: catalog_entry.huggingface_id().to_owned(),
            revision: catalog_entry.revision().to_owned(),
            display_name: catalog_entry.display_name().to_owned(),
            family: catalog_entry.family().as_str(),
            approximate_size_bytes: catalog_entry.approximate_size_bytes(),
            public: true,
            ready_on_this_mac: projection.ready_on_this_mac,
            destination_directory,
            download_state: projection.download_state,
            description: catalog_entry.description().map(str::to_owned),
            quantization_label: catalog_entry.quantization_label().map(str::to_owned),
            architecture_summary: catalog_entry.architecture_summary().map(str::to_owned),
            upstream_license: catalog_entry.upstream_license().map(str::to_owned),
            requestable_model_id: projection.requestable_model_id.clone(),
            capabilities: LibraryCatalogCapabilitiesResponse {
                supports_reasoning: capabilities.supports_reasoning,
                supports_vision: capabilities.supports_vision,
                supports_tool_calls: capabilities.supports_tool_calls,
                supports_image_generation: capabilities.supports_image_generation,
                supports_embeddings: capabilities.supports_embeddings,
                context_window: capabilities.context_window,
                max_output_tokens: capabilities.max_output_tokens,
            },
        }
    }
}
