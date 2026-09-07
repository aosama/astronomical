#![forbid(unsafe_code)]

mod aligned_expert_pack;
mod aligned_expert_pack_layout;
mod aligned_expert_pack_loader;
mod aligned_expert_pack_positional_io;
mod aligned_expert_pack_preparer;
mod per_expert_pack;
mod per_expert_pack_layout;
mod per_expert_pack_loader;
mod revision_manifest;
mod streaming_model_preparer;
mod streaming_model_resident_bundle;
mod streaming_model_revision;

pub use aligned_expert_pack::{
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES, AlignedExpertPackBuildRequest,
    AlignedExpertPackError, AlignedExpertPackHeader, AlignedExpertPackTensorDescriptor,
    build_aligned_expert_pack, read_aligned_expert_pack_header,
    validate_aligned_expert_pack_header, validate_aligned_expert_pack_payload,
};
pub use aligned_expert_pack_loader::build_aligned_expert_pack_metal_io_descriptors;
pub use aligned_expert_pack_preparer::{
    AlignedExpertPackPreparationError, AlignedExpertPackPreparationInspection,
    AlignedExpertPackPreparationProgress, AlignedExpertPackPreparationReport,
    AlignedExpertPackPreparer,
};
pub use per_expert_pack::{
    PER_EXPERT_PACK_FORMAT_VERSION, PerExpertPackBuildRequest, PerExpertPackHeader,
    PerExpertPackTensorDescriptor, build_per_expert_pack, per_expert_pack_relative_path,
    read_per_expert_pack_header, validate_per_expert_pack_header, validate_per_expert_pack_payload,
};
pub use per_expert_pack_loader::build_per_expert_pack_metal_io_descriptors;
pub use revision_manifest::{
    StreamingModelExpertFile, StreamingModelManifest, StreamingModelResidentFile,
};
pub use streaming_model_preparer::{
    STREAMING_MODEL_IDENTITY_SUFFIX, StreamingModelPreparationProgress,
    StreamingModelPreparationReport, StreamingModelPreparer, streaming_model_id_for,
};
