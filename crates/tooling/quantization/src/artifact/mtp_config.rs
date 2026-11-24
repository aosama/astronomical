//! The `config.json` registration of a written multi-token-prediction head.
//!
//! OptiQ's runtime finds the head through these keys, in the convention the
//! MTPLX project established for a head shipped beside a model: where the
//! file is, how many tensors it holds, and how its projections were
//! quantized so a loader builds matching modules.
//!
//! Ported from OptiQ's `_register_mtp_sidecar` (`optiq/runtime/mtp_convert.py`).

use std::path::Path;

use serde_json::{Map, Value, json};

use crate::error::QuantizationError;
use crate::requantize::{MTP_SIDECAR_RELATIVE_PATH, MtpSidecarSummary};

use super::json_file::{read_json, write_pretty_json};

#[cfg(test)]
#[path = "mtp_config_tests.rs"]
mod mtp_config_tests;

/// The policy name for a head whose fusion layer stays at bfloat16 while
/// its projections are quantized, the only arrangement this crate writes.
const PROJECTIONS_ONLY_POLICY: &str = "cyankiwi";

/// Records the written head in the artifact's `config.json`. The source's
/// own `mtp_num_hidden_layers` is kept when it declares one at the top
/// level; otherwise the count seen in the head's tensors is recorded.
pub fn register_mtp_sidecar(
    config_path: &Path,
    mtp_sidecar: &MtpSidecarSummary,
) -> Result<(), QuantizationError> {
    let mut config_entries: Map<String, Value> = match read_json(config_path) {
        Ok(config_entries) => config_entries,
        Err(description) => {
            return Err(QuantizationError::ModelConfigUnreadable {
                config_path: config_path.to_path_buf(),
                description,
            });
        }
    };
    config_entries.insert("mtp_file".to_owned(), json!(MTP_SIDECAR_RELATIVE_PATH));
    config_entries.insert(
        "mtp_tensor_count".to_owned(),
        json!(mtp_sidecar.tensor_count),
    );
    config_entries.insert(
        "mtp_policy".to_owned(),
        json!(format!(
            "optiq-int{}-prequantized-gs{}",
            mtp_sidecar.bits, mtp_sidecar.group_size
        )),
    );
    config_entries.insert(
        "mtplx_mtp_quantization".to_owned(),
        json!({
            "bits": mtp_sidecar.bits,
            "group_size": mtp_sidecar.group_size,
            "mode": "affine",
            "policy": PROJECTIONS_ONLY_POLICY,
            "prequantized": true,
        }),
    );
    let mut extra_tensors = match config_entries.remove("mlx_lm_extra_tensors") {
        Some(Value::Object(extra_tensors)) => extra_tensors,
        _ => Map::new(),
    };
    extra_tensors.insert("mtp_file".to_owned(), json!(MTP_SIDECAR_RELATIVE_PATH));
    config_entries.insert(
        "mlx_lm_extra_tensors".to_owned(),
        Value::Object(extra_tensors),
    );
    if !config_entries.contains_key("mtp_num_hidden_layers") {
        config_entries.insert(
            "mtp_num_hidden_layers".to_owned(),
            json!(mtp_sidecar.layer_count),
        );
    }
    write_pretty_json(&config_entries, config_path)
}
