#![forbid(unsafe_code)]

mod aligned_expert_pack;
mod aligned_expert_pack_layout;
mod aligned_expert_pack_loader;
mod aligned_expert_pack_positional_io;
mod aligned_expert_pack_preparer;

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
