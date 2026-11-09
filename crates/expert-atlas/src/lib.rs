//! The Expert Atlas: an expert-major re-layout of a mixture-of-experts
//! checkpoint, kept as a model of its own.
//!
//! A stock safetensors checkpoint stores each expert-batched tensor whole, so
//! one routed expert is nine slices spread across gigabytes of a shard (three
//! projections, each a weight, scales and biases). The Atlas writes those
//! nine slices back to back as one record per expert, one record file per
//! layer, every record on a 16 KiB boundary. A routed expert is then one
//! contiguous span of its layer file instead of nine slices gigabytes apart,
//! so serving reads a page of experts as a few large sequential reads.
//!
//! Measured cold (file cache evicted) on one 8-expert page (26.7 MB) of a
//! 35B A3B 8-bit model: about 2.8 ms through the serving reader for the
//! Atlas, against about 5.0 ms for the stock layout read the way MLX loads
//! it.
//!
//! This crate owns the format (`atlas_manifest`), the new model's name
//! (`atlas_model_naming`) and the conversion (`atlas_conversion`). It needs
//! no MLX and no daemon: conversion copies bytes and decodes nothing.

pub mod atlas_conversion;
pub mod atlas_manifest;
pub mod atlas_model_naming;
mod layer_record_writer;
pub mod record_layout;
mod resident_bundle_writer;
pub mod source_checkpoint;
// The only module that needs unsafe code: `fcntl(F_NOCACHE)`, `statvfs` and
// page-aligned allocation for direct reads. Every block carries its SAFETY
// argument; the rest of the crate stays under the workspace's deny.
#[allow(unsafe_code)]
mod uncached_file_io;

pub use atlas_conversion::{
    AtlasConversionError, ConversionProgress, ConversionProgressObserver, ConversionRequest,
    ConversionStage, ConversionSummary, convert_to_atlas,
};
pub use atlas_manifest::{
    ATLAS_FORMAT_NAME, ATLAS_FORMAT_VERSION, ATLAS_MANIFEST_FILE_NAME,
    ATLAS_RECORD_ALIGNMENT_BYTES, AtlasDeclaredFile, AtlasExpertLayer, AtlasManifest,
    AtlasManifestError, AtlasRecordSlot, AtlasSource, RESIDENT_BUNDLE_FILE_NAME,
    declares_atlas_manifest,
};
pub use atlas_model_naming::{ATLAS_MODEL_NAME_SUFFIX, atlas_model_id_for, organization_and_name};
pub use source_checkpoint::VISION_BUNDLE_FILE_NAME;
