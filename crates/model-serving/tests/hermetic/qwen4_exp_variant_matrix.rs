//! The `qwen4_exp` variant matrix: every generated variant validates, and
//! the matrix covers every published axis of variation.
//!
//! The user-visible outcome under test: a family implementation that
//! hard-wires one artifact's packaging fails at least one named variant, and
//! the matrix runs inside routine bounds — kilobytes per variant, no
//! downloaded checkpoint required.

use std::fs;

use crate::common::qwen4_exp::{
    NgramStorageForm, VariantQuantization, generate_variant, variant_matrix,
};
use astronomical_model_serving::{Qwen4ExpConfig, Qwen4ExpQuantizationMode};

#[test]
fn every_named_variant_validates_and_matches_its_spec() {
    let parent = tempfile::tempdir().expect("temporary directory should be created");
    let variants = variant_matrix();
    assert!(
        variants.len() >= 12,
        "the matrix must cover the published spread, got {}",
        variants.len()
    );
    for spec in &variants {
        let generated = generate_variant(parent.path(), spec)
            .unwrap_or_else(|error| panic!("variant {} should generate: {error}", spec.model_id));
        let config_bytes =
            fs::read(generated.model_directory.join("config.json")).unwrap_or_else(|error| {
                panic!("variant {} config should read: {error}", spec.model_id)
            });
        let config = Qwen4ExpConfig::from_json_bytes(&config_bytes)
            .unwrap_or_else(|error| panic!("variant {} should validate: {error}", spec.model_id));
        assert_eq!(
            config.decoder_layers, spec.decoder_layers,
            "variant {} layer count must round-trip",
            spec.model_id
        );
        assert_eq!(
            config.vocabulary_size, spec.vocabulary_size,
            "variant {} vocabulary must round-trip",
            spec.model_id
        );
        assert_eq!(
            config
                .ngram_embedding
                .as_ref()
                .expect("every variant carries the lookup group")
                .layer_ids_one_based,
            vec![2],
            "variant {} lookup layer must round-trip",
            spec.model_id
        );
        match (
            spec.quantization,
            config.default_quantization.map(|p| p.mode),
        ) {
            (VariantQuantization::None, None) => {}
            (VariantQuantization::Affine { .. }, Some(Qwen4ExpQuantizationMode::Affine)) => {}
            (VariantQuantization::Mxfp4, Some(Qwen4ExpQuantizationMode::Mxfp4)) => {}
            (spec_quantization, parsed_mode) => {
                panic!(
                    "variant {} quantization must round-trip: spec {spec_quantization:?}, parsed {parsed_mode:?}",
                    spec.model_id
                );
            }
        }
    }
}

#[test]
fn the_matrix_covers_every_published_axis_of_variation() {
    let variants = variant_matrix();
    // Expert count: both published geometries.
    let expert_counts: std::collections::BTreeSet<u32> =
        variants.iter().map(|spec| spec.routed_experts).collect();
    assert!(
        expert_counts.len() >= 2,
        "both expert geometries must appear"
    );
    // Quantization: native, 2-bit, 3-bit, 4-bit affine, and MXFP4.
    let modes: std::collections::BTreeSet<String> = variants
        .iter()
        .map(|spec| match spec.quantization {
            VariantQuantization::None => "none".to_owned(),
            VariantQuantization::Affine { bits, .. } => format!("affine{bits}"),
            VariantQuantization::Mxfp4 => "mxfp4".to_owned(),
        })
        .collect();
    for required in ["none", "affine2", "affine3", "affine4", "mxfp4"] {
        assert!(
            modes.contains(required),
            "quantization axis {required} must appear"
        );
    }
    // Lookup naming: both schemes, plus the manifest form.
    assert!(
        variants
            .iter()
            .any(|spec| spec.ngram_storage == NgramStorageForm::InlineUnderscoreNaming)
    );
    assert!(
        variants
            .iter()
            .any(|spec| spec.ngram_storage == NgramStorageForm::InlineDotNamingWithManifest)
    );
    assert!(
        variants
            .iter()
            .any(|spec| spec.ngram_storage == NgramStorageForm::InlineDotNamingWithoutBiases)
    );
    // A variant that declares a prediction head without tensors.
    assert!(
        variants
            .iter()
            .any(|spec| spec.subsystems.multi_token_prediction_declared
                && !spec.multi_token_prediction_tensors)
    );
    // A text-only variant with the optional subsystem groups absent.
    assert!(variants.iter().any(|spec| !spec.subsystems.linear_attention
        && !spec.subsystems.sparse_attention
        && !spec.subsystems.hyper_connections));
    // Both convolution axis orders.
    assert!(
        variants
            .iter()
            .any(|spec| spec.convolution_axes_channel_last)
    );
    assert!(
        variants
            .iter()
            .any(|spec| !spec.convolution_axes_channel_last)
    );
    // Quantized and BF16 indexer projections.
    assert!(
        variants
            .iter()
            .any(|spec| spec.indexer_projection_quantized)
    );
    assert!(
        variants
            .iter()
            .any(|spec| !spec.indexer_projection_quantized)
    );
}

#[test]
fn generated_artifacts_are_kilobytes_not_gigabytes() {
    let parent = tempfile::tempdir().expect("temporary directory should be created");
    let spec = variant_matrix()
        .into_iter()
        .next()
        .expect("the matrix is non-empty");
    let generated = generate_variant(parent.path(), &spec).expect("variant generates");
    let mut total_bytes = 0_u64;
    for entry in fs::read_dir(&generated.model_directory).expect("variant directory reads") {
        let entry = entry.expect("directory entry reads");
        total_bytes += entry.metadata().expect("entry metadata reads").len();
    }
    assert!(
        total_bytes < 1024 * 1024,
        "a generated variant must stay kilobyte-scale for routine runs, got {total_bytes} bytes"
    );
    assert!(
        generated.shard_count >= 2,
        "the writer must split tensors across shards at the size ceiling"
    );
}

#[test]
fn lookup_geometry_stays_consistent_with_the_pinned_rule() {
    let spec = variant_matrix()
        .into_iter()
        .next()
        .expect("the matrix is non-empty");
    let generated = generate_variant(
        &tempfile::tempdir().expect("temporary directory").path(),
        &spec,
    )
    .expect("variant generates");
    let (_, _, padded_rows) = spec.ngram_layout().expect("lookup geometry derives");
    assert_eq!(
        generated.ngram_row_count, padded_rows,
        "the writer must emit the row count the pinned rule derives"
    );
    assert_eq!(padded_rows % 8, 0, "rows must respect the divisor padding");
}

#[test]
fn the_manifest_form_publishes_ple_store_json() {
    let parent = tempfile::tempdir().expect("temporary directory should be created");
    let spec = variant_matrix()
        .into_iter()
        .find(|spec| spec.ngram_storage == NgramStorageForm::InlineDotNamingWithManifest)
        .expect("the matrix carries the manifest form");
    let generated = generate_variant(parent.path(), &spec).expect("variant generates");
    assert!(
        generated.model_directory.join("ple-store.json").is_file(),
        "the manifest form must publish ple-store.json"
    );
    let plain = variant_matrix()
        .into_iter()
        .find(|spec| spec.ngram_storage == NgramStorageForm::InlineDotNaming)
        .expect("the matrix carries the plain form");
    let plain_generated = generate_variant(parent.path(), &plain).expect("variant generates");
    assert!(
        !plain_generated
            .model_directory
            .join("ple-store.json")
            .is_file(),
        "the plain form must not publish a manifest, matching the six artifacts that omit it"
    );
}
