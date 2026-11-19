use std::path::Path;

use super::optiq::metadata_location::resolve_optiq_metadata_relative_path;
use super::oq::report::OQ_IMATRIX_REPORT_FILE_NAME;

/// The tool that chose an artifact's per-module bit widths. This is
/// provenance only: the loader's decisions depend on the MLX `quantization`
/// block, never on which producer wrote it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationProducer {
    OptiQ,
    OmlxOq,
    Unspecified,
}

/// Classifies an artifact from which producer sidecars it carries.
///
/// OptiQ wins when both sidecars are present: its metadata is written by the
/// quantizer itself, while an imatrix report can be copied alongside by any
/// later tooling that also calibrated the weights.
pub fn detect_quantization_producer(
    has_optiq_metadata: bool,
    has_oq_imatrix_report: bool,
) -> QuantizationProducer {
    if has_optiq_metadata {
        QuantizationProducer::OptiQ
    } else if has_oq_imatrix_report {
        QuantizationProducer::OmlxOq
    } else {
        QuantizationProducer::Unspecified
    }
}

/// Gathers the file-existence facts for `model_directory` and classifies it.
pub fn detect_quantization_producer_in_directory(model_directory: &Path) -> QuantizationProducer {
    detect_quantization_producer(
        resolve_optiq_metadata_relative_path(model_directory).is_some(),
        model_directory.join(OQ_IMATRIX_REPORT_FILE_NAME).is_file(),
    )
}

#[cfg(test)]
mod detection_tests;
