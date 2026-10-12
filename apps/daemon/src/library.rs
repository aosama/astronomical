//! Release-bundled model catalog and its local Library REST boundary.

pub(crate) mod catalog;
pub(crate) mod download;
pub(crate) mod hub;
pub(crate) mod library_application;
pub(crate) mod library_model_discovery_refresh;

pub(crate) use catalog::{
    CatalogEntryProjection, library_catalog_routes, project_catalog_entries,
    requestable_model_id_from_huggingface_id,
};
pub use catalog::{
    DownloadCatalog, DownloadCatalogCapabilities, DownloadCatalogEntry, DownloadCatalogError,
    DownloadCatalogFamily,
};
pub(crate) use catalog::{is_valid_huggingface_id, is_valid_immutable_revision};
pub use download::job::{DownloadJob, DownloadJobError, DownloadJobFile, DownloadJobState};
pub use download::job::{DownloadJobPublicErrorCode, DownloadJobStore, DownloadJobStoreError};
pub(crate) use download::library_download_routes;
pub use download::transfer::download_payload_transfer::DownloadTransferControl;
pub use download::transfer::download_payload_transfer_segment::PAYLOAD_TRANSFER_SEGMENT_BYTES;
pub use download::transfer::download_payload_transfer_window::MAXIMUM_CONCURRENT_PAYLOAD_TRANSFERS;
pub use download::transfer::{DownloadFileDigest, DownloadPayloadTransfer};
pub use download::transfer::{DownloadPayloadTransferError, DownloadPayloadTransferOutcome};
pub use download::{
    DiskCapacityQuery, DownloadDiskCapacityCheck, DownloadDiskPreflight,
    DownloadDiskPreflightError, DownloadExecutablePreflight, DownloadExecutablePreflightError,
    DownloadManifestPreflight, DownloadManifestPreflightError, DownloadPathSelection,
    DownloadProgressSnapshot, DownloadPublication, DownloadPublicationError,
    DownloadPublicationRefresh, Fs4DiskCapacityQuery, LibraryDownloadCoordinator,
    LibraryDownloadCoordinatorError,
};
pub use hub::{
    HubHttpMethod, HubHttpRequest, HubHttpResponse, HubHttpResponseError, HubManifestFile,
    HubPayloadByteStream, HubPayloadFuture, HubPayloadRequest, HubPayloadResponse,
    HubPayloadTransport, HubTransport, HubTransportError, HubTransportFuture, HuggingFaceHub,
    HuggingFaceHubError, HuggingFaceHubLimits, HuggingFaceManifest,
};
pub use hub::{ReqwestHubTransport, ReqwestHubTransportBuildError};
