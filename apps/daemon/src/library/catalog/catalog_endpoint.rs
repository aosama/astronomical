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
            is_removable_from_disk(&application_state, projection),
        ));
    }
    let models_download_directory =
        application_state
            .library_download_coordinator
            .as_ref()
            .map(|download_coordinator| {
                download_coordinator
                    .models_directory()
                    .display()
                    .to_string()
            });
    let models_directory_free_bytes = application_state
        .library_download_coordinator
        .as_ref()
        .and_then(|download_coordinator| download_coordinator.models_directory_free_bytes());
    Json(LibraryCatalogResponse {
        schema_version: application_state.download_catalog.schema_version(),
        models_download_directory,
        models_directory_free_bytes,
        entries,
    })
}

/// Offered only for a model on this Mac whose folder Download Models itself
/// published, so the page never offers to remove a copy the person placed,
/// and only where this Mac can move folders to the Trash; elsewhere the page
/// keeps showing the folder in Finder instead of a button that can only fail.
///
/// The check reads metadata, resolves two paths and reads a small provenance
/// file, synchronously inside this async handler. That is a few filesystem calls
/// served from the macOS file cache per downloaded model, well below what a blocking task
/// hand-off would cost, so it runs inline.
fn is_removable_from_disk(
    application_state: &ApplicationState,
    projection: &super::CatalogEntryProjection,
) -> bool {
    projection.ready_on_this_mac
        && application_state.model_folder_trash.is_available()
        && application_state
            .library_download_coordinator
            .as_ref()
            .is_some_and(|download_coordinator| {
                download_coordinator
                    .is_removable_installation(projection.catalog_entry.huggingface_id())
            })
}

#[derive(Debug, Serialize)]
struct LibraryCatalogResponse {
    schema_version: u32,
    /// Where new Library downloads are published, so clients can show the
    /// destination before the user commits to a multi-gigabyte download.
    #[serde(skip_serializing_if = "Option::is_none")]
    models_download_directory: Option<String>,
    /// Free bytes on the disk that holds the models folder, read the way the
    /// download check reads them, so the page's figure agrees with what a
    /// download is admitted against. Absent when the disk cannot be read.
    #[serde(skip_serializing_if = "Option::is_none")]
    models_directory_free_bytes: Option<u64>,
    entries: Vec<LibraryCatalogEntryResponse>,
}

#[derive(Debug, Serialize)]
struct LibraryCatalogEntryResponse {
    huggingface_id: String,
    revision: String,
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    family: &'static str,
    approximate_size_bytes: u64,
    approximate_memory_bytes: u64,
    role: &'static str,
    get_started: bool,
    experimental: bool,
    retiring: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    minimum_memory_bytes: Option<u64>,
    public: bool,
    ready_on_this_mac: bool,
    removable_from_disk: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    destination_directory: Option<String>,
    download_state: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upstream_license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    requestable_model_id: Option<String>,
    /// The model ID of the lead entry this is another build of; the page
    /// folds the build under that lead.
    #[serde(skip_serializing_if = "Option::is_none")]
    build_of: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    build_difference: Option<&'static str>,
    /// Reply speeds measured on a named Mac; absent for an unmeasured entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    measured_speed: Option<LibraryCatalogMeasuredSpeedResponse>,
    capabilities: LibraryCatalogCapabilitiesResponse,
}

#[derive(Debug, Serialize)]
struct LibraryCatalogMeasuredSpeedResponse {
    measured_on: String,
    astronomical_version: String,
    tokens_per_second: LibraryCatalogTaskSpeedsResponse,
}

#[derive(Debug, Serialize)]
struct LibraryCatalogTaskSpeedsResponse {
    essay_short_prompt: u16,
    essay_long_prompt: u16,
    code_short_prompt: u16,
    code_long_prompt: u16,
}

impl LibraryCatalogMeasuredSpeedResponse {
    fn from_measured_speed(
        measured_speed: &download_catalog::DownloadCatalogMeasuredSpeed,
    ) -> Self {
        let task_speeds = measured_speed.tokens_per_second();
        Self {
            measured_on: measured_speed.measured_on().to_owned(),
            astronomical_version: measured_speed.astronomical_version().to_owned(),
            tokens_per_second: LibraryCatalogTaskSpeedsResponse {
                essay_short_prompt: task_speeds.essay_short_prompt,
                essay_long_prompt: task_speeds.essay_long_prompt,
                code_short_prompt: task_speeds.code_short_prompt,
                code_long_prompt: task_speeds.code_long_prompt,
            },
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct LibraryCatalogCapabilitiesResponse {
    supports_reasoning: bool,
    supports_vision: bool,
    supports_tool_calls: bool,
    supports_image_generation: bool,
    supports_embeddings: bool,
    supports_decisions: bool,
    supports_speech: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

impl LibraryCatalogEntryResponse {
    fn from_entry(
        projection: &super::CatalogEntryProjection,
        destination_directory: Option<String>,
        removable_from_disk: bool,
    ) -> Self {
        let catalog_entry = projection.catalog_entry;
        let capabilities = &projection.capabilities;
        let build = catalog_entry.build();
        Self {
            huggingface_id: catalog_entry.huggingface_id().to_owned(),
            revision: catalog_entry.revision().to_owned(),
            model_id: catalog_entry.model_id().to_owned(),
            summary: catalog_entry.summary().map(str::to_owned),
            family: catalog_entry.family().as_str(),
            approximate_size_bytes: catalog_entry.approximate_size_bytes(),
            approximate_memory_bytes: catalog_entry.approximate_memory_bytes(),
            role: catalog_entry.role().as_str(),
            get_started: catalog_entry.get_started(),
            experimental: catalog_entry.experimental(),
            retiring: catalog_entry.retiring(),
            minimum_memory_bytes: catalog_entry.minimum_memory_bytes(),
            public: true,
            ready_on_this_mac: projection.ready_on_this_mac,
            removable_from_disk,
            destination_directory,
            download_state: projection.download_state,
            description: catalog_entry.description().map(str::to_owned),
            quantization_label: catalog_entry.quantization_label().map(str::to_owned),
            upstream_license: catalog_entry.upstream_license().map(str::to_owned),
            requestable_model_id: projection.requestable_model_id.clone(),
            build_of: build.map(|catalog_build| catalog_build.lead_model_id().to_owned()),
            build_difference: build.map(|catalog_build| catalog_build.difference().as_str()),
            measured_speed: catalog_entry
                .measured_speed()
                .map(LibraryCatalogMeasuredSpeedResponse::from_measured_speed),
            capabilities: LibraryCatalogCapabilitiesResponse {
                supports_reasoning: capabilities.supports_reasoning,
                supports_vision: capabilities.supports_vision,
                supports_tool_calls: capabilities.supports_tool_calls,
                supports_image_generation: capabilities.supports_image_generation,
                supports_embeddings: capabilities.supports_embeddings,
                supports_decisions: capabilities.supports_decisions,
                supports_speech: capabilities.supports_speech,
                context_window: capabilities.context_window,
                max_output_tokens: capabilities.max_output_tokens,
            },
        }
    }
}
