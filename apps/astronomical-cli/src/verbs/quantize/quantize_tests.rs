//! Unit tests for the `quantize` command handlers that need no filesystem,
//! no daemon, and no GPU. The default (MLX-free) build must refuse the
//! `convert` sub-verb by naming the feature that enables it, before any
//! path is touched — a recovery message a user can act on. The
//! filesystem-driven `plan` and `inspect` journeys live in
//! `tests/integration/quantize.rs`.

use std::path::PathBuf;
use std::time::Duration;

use crate::verbs::quantize::{
    QuantizeConvertArguments, QuantizeMethodArgument, QuantizeReferenceArgument,
};
use crate::verbs::quantize::{QuantizeDependencies, run_quantize_convert};

#[cfg(not(feature = "quantize-convert"))]
#[test]
fn should_refuse_convert_in_a_default_build_and_name_the_feature() {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut dependencies = QuantizeDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: Duration::from_secs(5),
    };
    let convert_arguments = QuantizeConvertArguments {
        // The refusal is a capability guard that fires before any path is
        // touched, so plain unused paths keep this a pure in-process unit.
        source_directory: PathBuf::from("unused-source"),
        output_directory: PathBuf::from("unused-output"),
        base_model_name: None,
        method: QuantizeMethodArgument::Static,
        reference: QuantizeReferenceArgument::Auto,
        target_bits_per_weight: 5.0,
        candidate_bits: vec![4, 8],
        group_size: 64,
        calibration_sequence_count: 24,
        calibration_mix_path: None,
        floor_components_per_block: 2,
        range_search_enabled: true,
        range_search_maximum_bits: 3,
        overwrite_output: false,
    };

    let convert_outcome = run_quantize_convert(&convert_arguments, &mut dependencies);

    let error_text = convert_outcome
        .expect_err("a default build refuses convert")
        .to_string();
    assert!(error_text.contains("quantize-convert"));
}
