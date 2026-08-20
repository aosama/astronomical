//! Hermetic rejection and comparison coverage for the independent FLUX reference contract.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use serde_json::{Value, json};

use crate::flux2_klein_reference_oracle::{ExpectedFluxReference, FluxReferenceOracle, sha256_hex};

const MODEL_ID: &str = "black-forest-labs/FLUX.2-klein-4B";
const MODEL_REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";
const PROMPT: &str = "A Romeo and Juliet-derived qualification prompt";

#[test]
fn should_reject_malformed_independent_reference_bundles() {
    for malformed_bundle in [
        b"not-json".as_slice(),
        br#"[]"#.as_slice(),
        br#"{"schema_version":1}"#.as_slice(),
    ] {
        assert!(FluxReferenceOracle::parse(malformed_bundle, &expected_reference()).is_err());
    }
}

#[test]
fn should_reject_reference_bundles_that_do_not_match_the_exact_input() {
    for (field_name, mismatched_value) in [
        (
            "bfl_source_repository",
            json!("https://example.invalid/flux2"),
        ),
        (
            "model_revision",
            json!("1111111111111111111111111111111111111111"),
        ),
        (
            "prompt_sha256",
            json!("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
        ),
        ("width", json!(32)),
        ("height", json!(32)),
        ("seed", json!(8)),
        ("steps", json!(5)),
        ("guidance", json!(1.5)),
    ] {
        let mut bundle = valid_bundle();
        bundle[field_name] = mismatched_value;
        let error = FluxReferenceOracle::parse(
            &serde_json::to_vec(&bundle).expect("fixture should serialize"),
            &expected_reference(),
        )
        .expect_err("mismatched reference metadata must fail before generation");
        assert!(error.contains(field_name));
    }
}

#[test]
fn should_reject_unpinned_sources_and_overly_loose_or_black_accepting_oracles() {
    let mut unpinned_source = valid_bundle();
    unpinned_source["bfl_source_revision"] = json!("main");
    assert!(
        FluxReferenceOracle::parse(
            &serde_json::to_vec(&unpinned_source).expect("fixture should serialize"),
            &expected_reference(),
        )
        .expect_err("a moving source revision must fail")
        .contains("pinned")
    );

    let mut unreviewed_source = valid_bundle();
    unreviewed_source["diffusers_source_revision"] =
        json!("1111111111111111111111111111111111111111");
    assert!(
        FluxReferenceOracle::parse(
            &serde_json::to_vec(&unreviewed_source).expect("fixture should serialize"),
            &expected_reference(),
        )
        .expect_err("an unreviewed source commit must fail")
        .contains("reviewed source commit")
    );

    let mut loose_tolerance = valid_bundle();
    loose_tolerance["tolerance"]["maximum_channel_error"] = json!(65);
    assert!(
        FluxReferenceOracle::parse(
            &serde_json::to_vec(&loose_tolerance).expect("fixture should serialize"),
            &expected_reference(),
        )
        .expect_err("an overly loose tolerance must fail")
        .contains("too loose")
    );

    let mut black_reference = valid_bundle();
    black_reference["reference"]["base64"] = json!(STANDARD.encode(vec![0_u8; 12]));
    assert!(
        FluxReferenceOracle::parse(
            &serde_json::to_vec(&black_reference).expect("fixture should serialize"),
            &expected_reference(),
        )
        .expect_err("an oracle that accepts black pixels must fail")
        .contains("all-black")
    );
}

#[test]
fn should_reject_incorrect_pixels_with_maximum_and_mean_error_diagnostics() {
    let oracle = parse_valid_bundle();
    let error = oracle
        .compare_generated_rgb(&[0_u8; 12])
        .expect_err("deterministic but incorrect pixels must not qualify");
    assert!(error.contains("max_error="));
    assert!(error.contains("mean_error="));
}

#[test]
fn should_reject_native_initial_noise_that_differs_from_the_reference_input() {
    let oracle = parse_valid_bundle();
    assert!(
        oracle
            .verify_initial_noise_sha256(
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
            )
            .is_err()
    );
    oracle
        .verify_initial_noise_sha256(
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        )
        .expect("the exact reference noise digest should match");
}

#[test]
fn should_accept_pixels_within_the_independent_reference_tolerance() {
    let oracle = parse_valid_bundle();
    let metrics = oracle
        .compare_generated_rgb(&[31, 64, 96, 128, 160, 192, 224, 200, 176, 152, 128, 104])
        .expect("adjacent RGB values should satisfy the explicit oracle tolerance");
    assert_eq!(metrics.maximum_channel_error, 1);
    assert_eq!(metrics.mean_channel_error, 1.0);
}

#[test]
fn should_decode_a_png_reference_to_the_same_rgb_comparison_boundary() {
    let reference_rgb = vec![32_u8, 65, 97, 129, 161, 193, 225, 201, 177, 153, 129, 105];
    let reference_image = ImageBuffer::<Rgb<u8>, _>::from_raw(2, 2, reference_rgb.clone())
        .expect("fixture RGB dimensions should match");
    let mut reference_png = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(reference_image)
        .write_to(&mut reference_png, ImageFormat::Png)
        .expect("fixture PNG should encode");
    let mut bundle = valid_bundle();
    bundle["reference"]["encoding"] = json!("png-base64");
    bundle["reference"]["base64"] = json!(STANDARD.encode(reference_png.into_inner()));
    let oracle = FluxReferenceOracle::parse(
        &serde_json::to_vec(&bundle).expect("fixture should serialize"),
        &expected_reference(),
    )
    .expect("PNG reference should decode through the RGB oracle boundary");

    oracle
        .compare_generated_rgb(&reference_rgb)
        .expect("decoded PNG RGB should compare exactly");
}

fn parse_valid_bundle() -> FluxReferenceOracle {
    FluxReferenceOracle::parse(
        &serde_json::to_vec(&valid_bundle()).expect("fixture should serialize"),
        &expected_reference(),
    )
    .expect("valid independent reference bundle should parse")
}

fn expected_reference() -> ExpectedFluxReference<'static> {
    ExpectedFluxReference {
        model_id: MODEL_ID,
        model_revision: MODEL_REVISION,
        prompt: PROMPT,
        width: 2,
        height: 2,
        seed: 7,
        steps: 4,
        guidance: 1.0,
    }
}

fn valid_bundle() -> Value {
    json!({
        "schema_version": 1,
        "reference_implementation": "black-forest-labs/diffusers",
        "bfl_source_repository": "https://github.com/black-forest-labs/flux2",
        "bfl_source_revision": "50fe5162777813d869182b139e83b10743caef15",
        "diffusers_source_repository": "https://github.com/huggingface/diffusers",
        "diffusers_source_revision": "2f7e0154a9db246e95c9ede43edba7db5b130805",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_sha256": sha256_hex(PROMPT.as_bytes()),
        "width": 2,
        "height": 2,
        "seed": 7,
        "steps": 4,
        "guidance": 1.0,
        "initial_noise": {
            "implementation": "mlx",
            "version": "0.32.1",
            "dtype": "bfloat16",
            "layout": "packed_batch_sequence_channels",
            "shape": [1, 1, 128],
            "float32_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        },
        "reference": {
            "encoding": "rgb8-base64",
            "base64": STANDARD.encode([32_u8, 65, 97, 129, 161, 193, 225, 201, 177, 153, 129, 105]),
        },
        "tolerance": {"maximum_channel_error": 1, "mean_channel_error": 1.0},
    })
}
