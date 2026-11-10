//! Manifest-level executable-identity gate that runs before payload transfer.
//!
//! Publication runs full executable discovery only after every byte is transferred and verified,
//! so an artifact that would never be discoverable used to cost a complete multi-gigabyte
//! transfer before failing. This gate reuses the same family classifiers and family-owned rules
//! on bounded fetched metadata to reject artifacts that discovery could never advertise, before
//! a byte is transferred. It is a cheap admission check, not the full publication backstop:
//! deeper family rules that need files on disk still run at publication.

use std::collections::BTreeSet;

use astronomical_config::{
    FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME, MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS,
    ModelFamily, classify_mflux_transformer_index_bytes, classify_pipeline_index_bytes,
    context_window_tokens, required_shard_file_names,
};
use serde_json::Value;
use thiserror::Error;

use crate::library::{HuggingFaceHub, HuggingFaceHubError, HuggingFaceManifest};

/// Diffusers pipelines carry both files; disk discovery classifies `model_index.json` first, and
/// the gate must apply that same precedence or it could reject an executable pipeline.
const PIPELINE_INDEX_FILE_PATH: &str = "model_index.json";
const CONFIG_FILE_PATH: &str = "config.json";
const QWEN_INDEX_FILE_PATH: &str = "model.safetensors.index.json";
const QWEN_TOKENIZER_FILE_PATH: &str = "tokenizer.json";
const QWEN_STREAMING_MANIFEST_FILE_PATH: &str = "manifest.json";
const QWEN_STREAMING_RESIDENT_FILE_PATH: &str = "resident.safetensors";
const PRISM_HADAMARD_CONTRACT_FILE_PATH: &str = "hadamard.json";
const PRISM_TOKENIZER_FILE_PATH: &str = "tokenizer.json";
const PRISM_WEIGHTS_FILE_PATH: &str = "model.safetensors";
const PRISM_SHARD_INDEX_FILE_PATH: &str = "model.safetensors.index.json";

/// Why one manifest does not describe an executable artifact.
#[derive(Debug, Error)]
pub enum DownloadExecutablePreflightError {
    #[error("{0}")]
    NotExecutable(String),
    #[error("Hugging Face executable preflight retrieval failed: {0}")]
    Hub(#[from] HuggingFaceHubError),
}

/// Owns the manifest-level executable-identity admission check.
pub struct DownloadExecutablePreflight {
    hub: HuggingFaceHub,
}

impl DownloadExecutablePreflight {
    #[must_use]
    pub fn new(hub: HuggingFaceHub) -> Self {
        Self { hub }
    }

    /// Rejects artifacts that executable discovery could never advertise, before transfer.
    pub async fn validate(
        &self,
        manifest: &HuggingFaceManifest,
    ) -> Result<(), DownloadExecutablePreflightError> {
        let manifest_file_paths = manifest
            .files()
            .iter()
            .map(|manifest_file| manifest_file.relative_path().to_owned())
            .collect::<BTreeSet<_>>();
        if manifest_file_paths.contains(PIPELINE_INDEX_FILE_PATH) {
            return self.validate_pipeline_artifact(manifest).await;
        }
        self.validate_non_pipeline_artifact(manifest, &manifest_file_paths)
            .await
    }

    /// Artifacts without a pipeline index classify in disk discovery's order:
    /// the 4-bit FLUX.2 Klein transformer index first, then `config.json`.
    async fn validate_non_pipeline_artifact(
        &self,
        manifest: &HuggingFaceManifest,
        manifest_file_paths: &BTreeSet<String>,
    ) -> Result<(), DownloadExecutablePreflightError> {
        let has_quantized_transformer_index =
            manifest_file_paths.contains(FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME);
        if has_quantized_transformer_index && self.transformer_index_classifies(manifest).await? {
            return Ok(());
        }
        if manifest_file_paths.contains(CONFIG_FILE_PATH) {
            return self
                .validate_config_artifact(manifest, manifest_file_paths)
                .await;
        }
        Err(DownloadExecutablePreflightError::NotExecutable(
            unclassifiable_manifest_reason(has_quantized_transformer_index).to_owned(),
        ))
    }

    /// Whether the transformer weight index describes the reviewed 4-bit FLUX.2
    /// Klein 4B build. A document that does not parse is not that build, as on disk.
    async fn transformer_index_classifies(
        &self,
        manifest: &HuggingFaceManifest,
    ) -> Result<bool, DownloadExecutablePreflightError> {
        let transformer_index_bytes = self
            .hub
            .fetch_bounded_repository_file(
                manifest.repository_id(),
                manifest.revision(),
                FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME,
            )
            .await?;
        Ok(matches!(
            classify_mflux_transformer_index_bytes(&transformer_index_bytes),
            Ok(Some(_))
        ))
    }

    /// Pipeline artifacts classify from the index document alone; deeper directory evidence
    /// keeps publication as its backstop.
    async fn validate_pipeline_artifact(
        &self,
        manifest: &HuggingFaceManifest,
    ) -> Result<(), DownloadExecutablePreflightError> {
        let pipeline_index_bytes = self
            .hub
            .fetch_bounded_repository_file(
                manifest.repository_id(),
                manifest.revision(),
                PIPELINE_INDEX_FILE_PATH,
            )
            .await?;
        let classified_family =
            classify_pipeline_index_bytes(&pipeline_index_bytes).map_err(|parse_error| {
                DownloadExecutablePreflightError::NotExecutable(format!(
                    "model_index.json is not a classifiable pipeline document: {parse_error}"
                ))
            })?;
        match classified_family {
            Some(ModelFamily::Flux2Klein) => Ok(()),
            Some(ModelFamily::QwenImage21) => Ok(()),
            Some(_) => Ok(()),
            None => Err(DownloadExecutablePreflightError::NotExecutable(
                "model_index.json does not describe a supported text-to-image pipeline".to_owned(),
            )),
        }
    }

    async fn validate_config_artifact(
        &self,
        manifest: &HuggingFaceManifest,
        manifest_file_paths: &BTreeSet<String>,
    ) -> Result<(), DownloadExecutablePreflightError> {
        let config_bytes = self
            .hub
            .fetch_bounded_repository_file(
                manifest.repository_id(),
                manifest.revision(),
                CONFIG_FILE_PATH,
            )
            .await?;
        let config_document = serde_json::from_slice::<Value>(&config_bytes).map_err(|source| {
            DownloadExecutablePreflightError::NotExecutable(format!(
                "config.json is not valid JSON: {source}"
            ))
        })?;
        let model_family =
            ModelFamily::from_model_type(config_document.get("model_type").and_then(Value::as_str));
        let Some(model_family) = model_family else {
            return Err(DownloadExecutablePreflightError::NotExecutable(
                "config.json model_type is not a recognized model family".to_owned(),
            ));
        };
        match model_family {
            // Families whose shallow rules need files on disk keep publication as their backstop.
            ModelFamily::ModernBert
            | ModelFamily::Laya
            | ModelFamily::Flux2Klein
            | ModelFamily::QwenImage21 => Ok(()),
            ModelFamily::Qwen3_5 => {
                self.validate_qwen3_5_manifest_shape(
                    manifest,
                    manifest_file_paths,
                    &config_document,
                )
                .await
            }
            ModelFamily::PrismHadamardQwen35 => {
                validate_prism_hadamard_manifest_shape(manifest_file_paths, &config_document)
            }
        }
    }

    /// Applies the family-owned shallow Qwen admission rules against the manifest and fetched
    /// metadata: shard index plus tokenizer must be selected, every mandatory shard must be
    /// selected, and the declared context window must serve at least one user turn.
    async fn validate_qwen3_5_manifest_shape(
        &self,
        manifest: &HuggingFaceManifest,
        manifest_file_paths: &BTreeSet<String>,
        config_document: &Value,
    ) -> Result<(), DownloadExecutablePreflightError> {
        if manifest_file_paths.contains(QWEN_STREAMING_MANIFEST_FILE_PATH) {
            if !manifest_file_paths.contains(QWEN_STREAMING_RESIDENT_FILE_PATH)
                || !manifest_file_paths.contains(QWEN_TOKENIZER_FILE_PATH)
            {
                return Err(DownloadExecutablePreflightError::NotExecutable(
                    "Expert Atlas models require a resident weight bundle and a tokenizer"
                        .to_owned(),
                ));
            }
            return Ok(());
        }
        if !manifest_file_paths.contains(QWEN_INDEX_FILE_PATH)
            || !manifest_file_paths.contains(QWEN_TOKENIZER_FILE_PATH)
        {
            return Err(DownloadExecutablePreflightError::NotExecutable(
                "Qwen3.5 artifacts require a shard index and a tokenizer".to_owned(),
            ));
        }
        if context_window_tokens(config_document) < MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS {
            return Err(DownloadExecutablePreflightError::NotExecutable(
                "the declared context window cannot serve a single user turn".to_owned(),
            ));
        }
        let index_bytes = self
            .hub
            .fetch_bounded_repository_file(
                manifest.repository_id(),
                manifest.revision(),
                QWEN_INDEX_FILE_PATH,
            )
            .await?;
        validate_qwen_shard_inventory(manifest_file_paths, &index_bytes)
    }
}

fn unclassifiable_manifest_reason(has_quantized_transformer_index: bool) -> &'static str {
    if has_quantized_transformer_index {
        "the transformer weight index does not describe the reviewed 4-bit FLUX.2 Klein 4B build"
    } else {
        "the manifest has neither model_index.json, config.json, nor a 4-bit transformer index, so no model family can classify it"
    }
}

/// Applies the family-owned shallow folded-pack admission rules: the Hadamard
/// contract, tokenizer, and single-file weights must all be selected, and the
/// sharded layout the family validator refuses must be caught before download.
fn validate_prism_hadamard_manifest_shape(
    manifest_file_paths: &BTreeSet<String>,
    config_document: &Value,
) -> Result<(), DownloadExecutablePreflightError> {
    if context_window_tokens(config_document) < MINIMUM_SERVABLE_CONTEXT_WINDOW_TOKENS {
        return Err(DownloadExecutablePreflightError::NotExecutable(
            "the declared context window cannot serve a single user turn".to_owned(),
        ));
    }
    for required_file_path in [
        PRISM_HADAMARD_CONTRACT_FILE_PATH,
        PRISM_TOKENIZER_FILE_PATH,
        PRISM_WEIGHTS_FILE_PATH,
    ] {
        if !manifest_file_paths.contains(required_file_path) {
            return Err(DownloadExecutablePreflightError::NotExecutable(format!(
                "folded ternary packs require {required_file_path} in the repository"
            )));
        }
    }
    if manifest_file_paths.contains(PRISM_SHARD_INDEX_FILE_PATH) {
        return Err(DownloadExecutablePreflightError::NotExecutable(
            "folded ternary packs are single-file artifacts; a shard index is not executable"
                .to_owned(),
        ));
    }
    Ok(())
}

fn validate_qwen_shard_inventory(
    manifest_file_paths: &BTreeSet<String>,
    index_bytes: &[u8],
) -> Result<(), DownloadExecutablePreflightError> {
    let index_document = serde_json::from_slice::<Value>(index_bytes).map_err(|source| {
        DownloadExecutablePreflightError::NotExecutable(format!(
            "model.safetensors.index.json is not valid JSON: {source}"
        ))
    })?;
    let Some(weight_map) = index_document.get("weight_map").and_then(Value::as_object) else {
        return Err(DownloadExecutablePreflightError::NotExecutable(
            "model.safetensors.index.json has no weight_map".to_owned(),
        ));
    };
    for required_shard_file_name in required_shard_file_names(weight_map) {
        if !manifest_file_paths.contains(&required_shard_file_name) {
            return Err(DownloadExecutablePreflightError::NotExecutable(format!(
                "the shard index requires {required_shard_file_name} but the manifest does not select it"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{DownloadExecutablePreflightError, validate_prism_hadamard_manifest_shape};
    use serde_json::json;

    #[test]
    fn should_refuse_a_folded_pack_manifest_missing_required_files() {
        let manifest_file_paths = BTreeSet::from([
            "config.json".to_owned(),
            "tokenizer.json".to_owned(),
            "model.safetensors".to_owned(),
        ]);

        let config_document = json!({"text_config": {"max_position_embeddings": 4096}});
        let preflight_result =
            validate_prism_hadamard_manifest_shape(&manifest_file_paths, &config_document);

        assert!(matches!(
            preflight_result,
            Err(DownloadExecutablePreflightError::NotExecutable(reason))
                if reason.contains("hadamard.json")
        ));
    }

    #[test]
    fn should_accept_a_complete_single_file_folded_pack_manifest() {
        let manifest_file_paths = BTreeSet::from([
            "config.json".to_owned(),
            "hadamard.json".to_owned(),
            "tokenizer.json".to_owned(),
            "model.safetensors".to_owned(),
        ]);
        let config_document = json!({"text_config": {"max_position_embeddings": 4096}});

        assert!(
            validate_prism_hadamard_manifest_shape(&manifest_file_paths, &config_document).is_ok()
        );
    }

    #[test]
    fn should_refuse_a_folded_pack_with_an_unservable_context_window() {
        let manifest_file_paths = BTreeSet::from([
            "config.json".to_owned(),
            "hadamard.json".to_owned(),
            "tokenizer.json".to_owned(),
            "model.safetensors".to_owned(),
        ]);
        let config_document = json!({"text_config": {"max_position_embeddings": 1}});

        assert!(matches!(
            validate_prism_hadamard_manifest_shape(&manifest_file_paths, &config_document),
            Err(DownloadExecutablePreflightError::NotExecutable(reason))
                if reason.contains("context window")
        ));
    }
}
