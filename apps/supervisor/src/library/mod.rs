//! Release-bundled model catalog and its local Library REST boundary.

mod catalog_endpoint;
mod download_catalog;
mod download_coordinator;
mod download_disk_preflight;
mod download_endpoint;
mod download_file_digest;
mod download_job;
mod download_job_store;
mod download_job_store_error;
mod download_job_store_filesystem;
mod download_job_store_lock;
mod download_job_store_publication;
mod download_manifest_preflight;
mod download_path_selection;
mod download_payload_response;
mod download_payload_transfer;
mod download_payload_verification;
mod download_progress_snapshot;
mod download_publication;
mod download_staged_file;
mod hub_payload_transport;
mod hub_transport;
mod hugging_face_hub;
mod hugging_face_hub_bounds;
mod reqwest_hub_transport;

pub(crate) use catalog_endpoint::library_catalog_routes;
pub use download_catalog::{
    DownloadCatalog, DownloadCatalogEntry, DownloadCatalogError, DownloadCatalogFamily,
};
pub(crate) use download_catalog::{is_valid_huggingface_id, is_valid_immutable_revision};
pub use download_coordinator::{LibraryDownloadCoordinator, LibraryDownloadCoordinatorError};
pub use download_disk_preflight::{
    DiskCapacityQuery, DownloadDiskCapacityCheck, DownloadDiskPreflight,
    DownloadDiskPreflightError, Fs4DiskCapacityQuery,
};
pub(crate) use download_endpoint::library_download_routes;
pub use download_file_digest::DownloadFileDigest;
pub use download_job::{
    DownloadJob, DownloadJobError, DownloadJobFile, DownloadJobPublicErrorCode, DownloadJobState,
};
pub use download_job_store::DownloadJobStore;
pub use download_job_store_error::DownloadJobStoreError;
pub use download_manifest_preflight::{DownloadManifestPreflight, DownloadManifestPreflightError};
pub use download_path_selection::DownloadPathSelection;
pub use download_payload_transfer::{
    DownloadPayloadTransfer, DownloadPayloadTransferError, DownloadPayloadTransferOutcome,
    DownloadTransferControl,
};
pub use download_progress_snapshot::DownloadProgressSnapshot;
pub use download_publication::{
    DownloadPublication, DownloadPublicationError, DownloadPublicationRefresh,
};
pub use hub_payload_transport::{
    HubPayloadByteStream, HubPayloadFuture, HubPayloadRequest, HubPayloadResponse,
    HubPayloadTransport,
};
pub use hub_transport::{
    HubHttpMethod, HubHttpRequest, HubHttpResponse, HubHttpResponseError, HubTransport,
    HubTransportError, HubTransportFuture,
};
pub use hugging_face_hub::{
    HubManifestFile, HuggingFaceHub, HuggingFaceHubError, HuggingFaceHubLimits, HuggingFaceManifest,
};
pub use reqwest_hub_transport::{ReqwestHubTransport, ReqwestHubTransportBuildError};
