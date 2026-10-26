//! Hugging Face Hub client, protocol bounds, and pluggable transports.

mod hugging_face;
pub(crate) mod hugging_face_bounds;
mod payload_transport;
mod reqwest_transport;
mod transport;

#[cfg(test)]
mod hugging_face_tests;

pub use hugging_face::{
    HubManifestFile, HuggingFaceHub, HuggingFaceHubError, HuggingFaceHubLimits, HuggingFaceManifest,
};
pub use payload_transport::{
    HubPayloadByteStream, HubPayloadFuture, HubPayloadRequest, HubPayloadResponse,
    HubPayloadTransport,
};
pub use reqwest_transport::{ReqwestHubTransport, ReqwestHubTransportBuildError};
pub use transport::{
    HubHttpMethod, HubHttpRequest, HubHttpResponse, HubHttpResponseError, HubTransport,
    HubTransportError, HubTransportFuture,
};
