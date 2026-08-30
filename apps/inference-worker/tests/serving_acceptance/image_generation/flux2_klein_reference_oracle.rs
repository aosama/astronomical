//! Independent FLUX RGB oracle contract shared by hermetic and real-model acceptance.
//!
//! Set `ASTRONOMICAL_FLUX2_KLEIN_REFERENCE_BUNDLE` to a JSON bundle produced outside
//! Astronomical by the pinned Black Forest Labs and Diffusers implementations. The bundle
//! must contain exactly this shape; revisions and digests are lowercase 40/64-character hex:
//!
//! ```text
//! {
//!   "schema_version": 1,
//!   "reference_implementation": "black-forest-labs/diffusers",
//!   "bfl_source_repository": "https://github.com/black-forest-labs/flux2",
//!   "bfl_source_revision": "<40 hex>",
//!   "diffusers_source_repository": "https://github.com/huggingface/diffusers",
//!   "diffusers_source_revision": "<40 hex>",
//!   "model_id": "black-forest-labs/FLUX.2-klein-4B",
//!   "model_revision": "<40 hex>",
//!   "prompt_sha256": "<64 hex>",
//!   "width": 64, "height": 64, "seed": 7309, "steps": 4, "guidance": 1.0,
//!   "initial_noise": {"implementation": "mlx", "version": "0.32.1", "dtype": "bfloat16",
//!     "layout": "packed_batch_sequence_channels", "shape": [1, 16, 128], "float32_sha256": "<64 hex>"},
//!   "reference": {"encoding": "rgb8-base64|png-base64", "base64": "..."},
//!   "tolerance": {"maximum_channel_error": 64, "mean_channel_error": 3.0}
//! }
//! ```

use std::collections::BTreeSet;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{GenericImageView, ImageFormat};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

#[path = "flux2_klein_reference_oracle/initial_noise.rs"]
mod initial_noise;

use initial_noise::parse_initial_noise;

const REFERENCE_IMPLEMENTATION: &str = "black-forest-labs/diffusers";
const BFL_SOURCE_REPOSITORY: &str = "https://github.com/black-forest-labs/flux2";
pub(crate) const BFL_SOURCE_REVISION: &str = "50fe5162777813d869182b139e83b10743caef15";
const DIFFUSERS_SOURCE_REPOSITORY: &str = "https://github.com/huggingface/diffusers";
pub(crate) const DIFFUSERS_SOURCE_REVISION: &str = "2f7e0154a9db246e95c9ede43edba7db5b130805";
const MAXIMUM_ALLOWED_CHANNEL_ERROR: u8 = 64;
const MAXIMUM_ALLOWED_MEAN_CHANNEL_ERROR: f64 = 3.0;
const MAXIMUM_ALLOWED_P99_CHANNEL_ERROR: u8 = 24;
const MAXIMUM_ALLOWED_P999_CHANNEL_ERROR: u8 = 40;

#[derive(Debug)]
pub(crate) struct ExpectedFluxReference<'a> {
    pub(crate) model_id: &'a str,
    pub(crate) model_revision: &'a str,
    pub(crate) prompt: &'a str,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) seed: u64,
    pub(crate) steps: u32,
    pub(crate) guidance: f64,
}

#[derive(Debug)]
pub(crate) struct FluxReferenceOracle {
    reference_rgb: Vec<u8>,
    maximum_channel_error: u8,
    mean_channel_error: f64,
    initial_noise_sha256: String,
}

#[derive(Debug)]
pub(crate) struct PixelErrorMetrics {
    pub(crate) maximum_channel_error: u8,
    pub(crate) mean_channel_error: f64,
    pub(crate) p99_channel_error: u8,
    pub(crate) p999_channel_error: u8,
    pub(crate) channels_above_eight: usize,
}

impl FluxReferenceOracle {
    pub(crate) fn parse(
        bundle_bytes: &[u8],
        expected: &ExpectedFluxReference<'_>,
    ) -> Result<Self, String> {
        let bundle: Value = serde_json::from_slice(bundle_bytes)
            .map_err(|parse_error| format!("reference bundle is not valid JSON: {parse_error}"))?;
        let bundle_object = bundle
            .as_object()
            .ok_or_else(|| "reference bundle root must be an object".to_owned())?;
        require_exact_fields(
            bundle_object,
            &[
                "schema_version",
                "reference_implementation",
                "bfl_source_repository",
                "bfl_source_revision",
                "diffusers_source_repository",
                "diffusers_source_revision",
                "model_id",
                "model_revision",
                "prompt_sha256",
                "width",
                "height",
                "seed",
                "steps",
                "guidance",
                "initial_noise",
                "reference",
                "tolerance",
            ],
            "reference bundle",
        )?;
        if require_u64(bundle_object, "schema_version", "reference bundle")? != 1 {
            return Err("reference bundle schema_version must be 1".to_owned());
        }
        require_equal_string(
            bundle_object,
            "reference_implementation",
            REFERENCE_IMPLEMENTATION,
        )?;
        require_equal_string(
            bundle_object,
            "bfl_source_repository",
            BFL_SOURCE_REPOSITORY,
        )?;
        require_equal_string(
            bundle_object,
            "diffusers_source_repository",
            DIFFUSERS_SOURCE_REPOSITORY,
        )?;

        require_pinned_source_revision(bundle_object, "bfl_source_revision", BFL_SOURCE_REVISION)?;
        require_pinned_source_revision(
            bundle_object,
            "diffusers_source_revision",
            DIFFUSERS_SOURCE_REVISION,
        )?;
        require_equal_string(bundle_object, "model_id", expected.model_id)?;
        require_equal_string(bundle_object, "model_revision", expected.model_revision)?;
        require_equal_string(
            bundle_object,
            "prompt_sha256",
            &sha256_hex(expected.prompt.as_bytes()),
        )?;
        require_equal_u64(bundle_object, "width", u64::from(expected.width))?;
        require_equal_u64(bundle_object, "height", u64::from(expected.height))?;
        require_equal_u64(bundle_object, "seed", expected.seed)?;
        require_equal_u64(bundle_object, "steps", u64::from(expected.steps))?;
        require_equal_f64(bundle_object, "guidance", expected.guidance)?;
        let initial_noise_sha256 = parse_initial_noise(bundle_object, expected)?;

        let (maximum_channel_error, mean_channel_error) = parse_tolerance(bundle_object)?;
        let reference_rgb = parse_reference_rgb(bundle_object, expected.width, expected.height)?;
        reject_oracle_that_accepts_black(
            &reference_rgb,
            maximum_channel_error,
            mean_channel_error,
        )?;

        Ok(Self {
            reference_rgb,
            maximum_channel_error,
            mean_channel_error,
            initial_noise_sha256,
        })
    }

    pub(crate) fn compare_generated_rgb(
        &self,
        generated_rgb: &[u8],
    ) -> Result<PixelErrorMetrics, String> {
        if generated_rgb.len() != self.reference_rgb.len() {
            return Err(format!(
                "generated RGB length {} does not match reference RGB length {}",
                generated_rgb.len(),
                self.reference_rgb.len()
            ));
        }
        let metrics = pixel_error_metrics(generated_rgb, &self.reference_rgb);
        if metrics.maximum_channel_error > self.maximum_channel_error
            || metrics.mean_channel_error > self.mean_channel_error
            || metrics.p99_channel_error > MAXIMUM_ALLOWED_P99_CHANNEL_ERROR
            || metrics.p999_channel_error > MAXIMUM_ALLOWED_P999_CHANNEL_ERROR
        {
            return Err(format!(
                "independent FLUX RGB mismatch: max_error={} allowed_max={} mean_error={:.6} allowed_mean={:.6} p99_error={} p999_error={} channels_above_eight={}",
                metrics.maximum_channel_error,
                self.maximum_channel_error,
                metrics.mean_channel_error,
                self.mean_channel_error,
                metrics.p99_channel_error,
                metrics.p999_channel_error,
                metrics.channels_above_eight,
            ));
        }
        Ok(metrics)
    }

    pub(crate) fn verify_initial_noise_sha256(
        &self,
        actual_initial_noise_sha256: &str,
    ) -> Result<(), String> {
        if actual_initial_noise_sha256 != self.initial_noise_sha256 {
            return Err(
                "native MLX initial noise does not match the independent reference bundle"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    const LOWERCASE_HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut digest_hex = String::with_capacity(digest.len() * 2);
    for digest_byte in digest {
        digest_hex.push(char::from(
            LOWERCASE_HEX_DIGITS[usize::from(digest_byte >> 4)],
        ));
        digest_hex.push(char::from(
            LOWERCASE_HEX_DIGITS[usize::from(digest_byte & 0x0f)],
        ));
    }
    digest_hex
}

fn parse_reference_rgb(
    bundle_object: &Map<String, Value>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let reference_object = require_object(bundle_object, "reference", "reference bundle")?;
    require_exact_fields(reference_object, &["encoding", "base64"], "reference")?;
    let encoding = require_string(reference_object, "encoding", "reference")?;
    let encoded_bytes = require_string(reference_object, "base64", "reference")?;
    let decoded_bytes = STANDARD
        .decode(encoded_bytes)
        .map_err(|decode_error| format!("reference base64 is invalid: {decode_error}"))?;
    let expected_rgb_length = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixel_count| pixel_count.checked_mul(3))
        .ok_or_else(|| "reference dimensions overflow RGB length".to_owned())?;
    match encoding {
        "rgb8-base64" => {
            if decoded_bytes.len() != expected_rgb_length {
                return Err(format!(
                    "reference RGB length {} does not match expected length {expected_rgb_length}",
                    decoded_bytes.len()
                ));
            }
            Ok(decoded_bytes)
        }
        "png-base64" => {
            let decoded_image =
                image::load_from_memory_with_format(&decoded_bytes, ImageFormat::Png)
                    .map_err(|decode_error| format!("reference PNG is invalid: {decode_error}"))?;
            if decoded_image.dimensions() != (width, height) {
                return Err(format!(
                    "reference PNG dimensions {:?} do not match expected dimensions ({width}, {height})",
                    decoded_image.dimensions()
                ));
            }
            Ok(decoded_image.into_rgb8().into_raw())
        }
        unsupported => Err(format!("reference encoding {unsupported:?} is unsupported")),
    }
}

fn parse_tolerance(bundle_object: &Map<String, Value>) -> Result<(u8, f64), String> {
    let tolerance_object = require_object(bundle_object, "tolerance", "reference bundle")?;
    require_exact_fields(
        tolerance_object,
        &["maximum_channel_error", "mean_channel_error"],
        "tolerance",
    )?;
    let maximum_channel_error_u64 =
        require_u64(tolerance_object, "maximum_channel_error", "tolerance")?;
    let maximum_channel_error = u8::try_from(maximum_channel_error_u64)
        .map_err(|_| "maximum_channel_error must fit in one RGB channel".to_owned())?;
    let mean_channel_error = require_f64(tolerance_object, "mean_channel_error", "tolerance")?;
    if maximum_channel_error > MAXIMUM_ALLOWED_CHANNEL_ERROR
        || !mean_channel_error.is_finite()
        || mean_channel_error < 0.0
        || mean_channel_error > MAXIMUM_ALLOWED_MEAN_CHANNEL_ERROR
    {
        return Err(format!(
            "tolerance is too loose; maximum_channel_error must be <= {MAXIMUM_ALLOWED_CHANNEL_ERROR} and mean_channel_error must be within 0..={MAXIMUM_ALLOWED_MEAN_CHANNEL_ERROR}"
        ));
    }
    Ok((maximum_channel_error, mean_channel_error))
}

fn reject_oracle_that_accepts_black(
    reference_rgb: &[u8],
    maximum_channel_error: u8,
    mean_channel_error: f64,
) -> Result<(), String> {
    let black_metrics = pixel_error_metrics(&vec![0_u8; reference_rgb.len()], reference_rgb);
    if black_metrics.maximum_channel_error <= maximum_channel_error
        && black_metrics.mean_channel_error <= mean_channel_error
    {
        return Err("reference and tolerance would accept an all-black generated image".to_owned());
    }
    Ok(())
}

fn pixel_error_metrics(actual_rgb: &[u8], reference_rgb: &[u8]) -> PixelErrorMetrics {
    let mut maximum_channel_error = 0_u8;
    let mut total_channel_error = 0_u64;
    let mut channel_errors = Vec::with_capacity(actual_rgb.len());
    for (actual_channel, reference_channel) in actual_rgb.iter().zip(reference_rgb) {
        let channel_error = actual_channel.abs_diff(*reference_channel);
        maximum_channel_error = maximum_channel_error.max(channel_error);
        total_channel_error = total_channel_error.saturating_add(u64::from(channel_error));
        channel_errors.push(channel_error);
    }
    channel_errors.sort_unstable();
    let mean_channel_error = if actual_rgb.is_empty() {
        0.0
    } else {
        total_channel_error as f64 / actual_rgb.len() as f64
    };
    PixelErrorMetrics {
        maximum_channel_error,
        mean_channel_error,
        p99_channel_error: percentile_error(&channel_errors, 990),
        p999_channel_error: percentile_error(&channel_errors, 999),
        channels_above_eight: channel_errors
            .iter()
            .filter(|channel_error| **channel_error > 8)
            .count(),
    }
}

fn percentile_error(sorted_channel_errors: &[u8], percentile_thousandths: usize) -> u8 {
    if sorted_channel_errors.is_empty() {
        return 0;
    }
    let index = (sorted_channel_errors.len() - 1).saturating_mul(percentile_thousandths) / 1_000;
    sorted_channel_errors[index]
}

fn require_pinned_source_revision(
    object: &Map<String, Value>,
    field_name: &str,
    expected_revision: &str,
) -> Result<String, String> {
    let revision = require_string(object, field_name, "reference bundle")?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{field_name} must be a pinned lowercase 40-character commit hash"
        ));
    }
    if revision != expected_revision {
        return Err(format!(
            "{field_name} does not match the reviewed source commit"
        ));
    }
    Ok(revision.to_owned())
}

fn require_exact_fields(
    object: &Map<String, Value>,
    expected_fields: &[&str],
    object_name: &str,
) -> Result<(), String> {
    let actual_fields = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected_fields = expected_fields.iter().copied().collect::<BTreeSet<_>>();
    if actual_fields != expected_fields {
        return Err(format!(
            "{object_name} fields do not match the required contract"
        ));
    }
    Ok(())
}

fn require_equal_string(
    object: &Map<String, Value>,
    field_name: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = require_string(object, field_name, "reference bundle")?;
    if actual != expected {
        return Err(format!(
            "reference bundle {field_name} does not match the acceptance input"
        ));
    }
    Ok(())
}

fn require_equal_u64(
    object: &Map<String, Value>,
    field_name: &str,
    expected: u64,
) -> Result<(), String> {
    let actual = require_u64(object, field_name, "reference bundle")?;
    if actual != expected {
        return Err(format!(
            "reference bundle {field_name} does not match the acceptance input"
        ));
    }
    Ok(())
}

fn require_equal_f64(
    object: &Map<String, Value>,
    field_name: &str,
    expected: f64,
) -> Result<(), String> {
    let actual = require_f64(object, field_name, "reference bundle")?;
    if actual != expected {
        return Err(format!(
            "reference bundle {field_name} does not match the acceptance input"
        ));
    }
    Ok(())
}

fn require_object<'a>(
    object: &'a Map<String, Value>,
    field_name: &str,
    object_name: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(field_name)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{object_name} {field_name} must be an object"))
}

fn require_string<'a>(
    object: &'a Map<String, Value>,
    field_name: &str,
    object_name: &str,
) -> Result<&'a str, String> {
    object
        .get(field_name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{object_name} {field_name} must be a string"))
}

fn require_u64(
    object: &Map<String, Value>,
    field_name: &str,
    object_name: &str,
) -> Result<u64, String> {
    object
        .get(field_name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{object_name} {field_name} must be an unsigned integer"))
}

fn require_f64(
    object: &Map<String, Value>,
    field_name: &str,
    object_name: &str,
) -> Result<f64, String> {
    object
        .get(field_name)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{object_name} {field_name} must be a number"))
}
