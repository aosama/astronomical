mod metadata;
mod metadata_document;
pub(crate) mod metadata_location;

pub use metadata::{OptiQMetadata, OptiQMetadataError};
pub use metadata_location::resolve_optiq_metadata_relative_path;
