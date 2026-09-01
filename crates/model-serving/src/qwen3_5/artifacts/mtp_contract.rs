use serde_json::{Map, Value};
use thiserror::Error;

use crate::memory::MtpDraftDepth;

pub const MAXIMUM_MTPLX_RUNTIME_BYTES: usize = 64 * 1024;
const MAXIMUM_SELECTED_STRING_BYTES: usize = 64;
const SUPPORTED_ARCHITECTURE_ID: &str = "qwen3-next-mtp";

/// Bounded selected MTPLX metadata needed to prove Qwen MTP execution semantics.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Qwen3_5MtpContract {
    architecture_id: Option<String>,
    schema_version: Option<String>,
    runtime_version: Option<String>,
    artifact_default_depth: Option<MtpDraftDepth>,
    artifact_maximum_depth: Option<MtpDraftDepth>,
}

impl Qwen3_5MtpContract {
    /// Parses only execution-relevant fields. Unknown publisher metadata is never retained.
    pub fn parse(
        config_bytes: &[u8],
        optional_runtime_bytes: Option<&[u8]>,
    ) -> Result<Self, Qwen3_5MtpContractError> {
        if optional_runtime_bytes.is_some_and(|bytes| bytes.len() > MAXIMUM_MTPLX_RUNTIME_BYTES) {
            return Err(Qwen3_5MtpContractError::RuntimeDocumentTooLarge);
        }
        let config_document = serde_json::from_slice::<Value>(config_bytes)
            .map_err(|_| Qwen3_5MtpContractError::Malformed)?;
        let runtime_document = optional_runtime_bytes
            .map(|bytes| {
                serde_json::from_slice::<Value>(bytes)
                    .map_err(|_| Qwen3_5MtpContractError::Malformed)
            })
            .transpose()?;

        let config_fields = SelectedContractFields::from_config(&config_document)?;
        let runtime_fields = runtime_document
            .as_ref()
            .map(SelectedContractFields::from_runtime)
            .transpose()?
            .unwrap_or_default();
        let selected_fields = config_fields.merge(runtime_fields)?;
        selected_fields.validate_supported_semantics()?;
        selected_fields.into_contract()
    }

    #[must_use]
    pub fn architecture_id(&self) -> Option<&str> {
        self.architecture_id.as_deref()
    }

    #[must_use]
    pub fn schema_version(&self) -> Option<&str> {
        self.schema_version.as_deref()
    }

    #[must_use]
    pub fn runtime_version(&self) -> Option<&str> {
        self.runtime_version.as_deref()
    }

    #[must_use]
    pub const fn artifact_default_depth(&self) -> Option<MtpDraftDepth> {
        self.artifact_default_depth
    }

    #[must_use]
    pub const fn artifact_maximum_depth(&self) -> Option<MtpDraftDepth> {
        self.artifact_maximum_depth
    }
}

/// Typed bounded reason why optional MTPLX metadata cannot authorize MTP execution.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Qwen3_5MtpContractError {
    #[error("optional MTP contract JSON is malformed")]
    Malformed,
    #[error("optional MTP runtime metadata exceeds 64 KB")]
    RuntimeDocumentTooLarge,
    #[error("duplicated optional MTP contract fields disagree")]
    FieldDisagreement,
    #[error("optional MTP contract uses unsupported execution semantics")]
    Incompatible,
}

#[derive(Clone, Debug, Default)]
struct SelectedContractFields {
    architecture_id: Option<String>,
    schema_version: Option<String>,
    runtime_version: Option<String>,
    base_hidden_variant: Option<String>,
    hidden_variant: Option<String>,
    concat_order: Option<String>,
    position_mode: Option<String>,
    default_depth: Option<u64>,
    maximum_depth: Option<u64>,
}

impl SelectedContractFields {
    fn from_config(config_document: &Value) -> Result<Self, Qwen3_5MtpContractError> {
        let root = json_object(config_document)?;
        let Some(contract_value) = root.get("mtplx_mtp_contract") else {
            return Ok(Self::default());
        };
        Self::from_contract_object(json_object(contract_value)?)
    }

    fn from_runtime(runtime_document: &Value) -> Result<Self, Qwen3_5MtpContractError> {
        let root = json_object(runtime_document)?;
        let mut fields = Self {
            architecture_id: selected_string(root, &["arch_id", "architecture"])?,
            schema_version: selected_string(root, &["schema", "schema_version"])?,
            runtime_version: selected_string(root, &["mtplx_version", "version"])?,
            default_depth: selected_integer(root, &["mtp_depth_default"])?,
            maximum_depth: selected_integer(root, &["mtp_depth_max"])?,
            ..Self::default()
        };
        if let Some(contract_value) = root.get("mtp_contract") {
            fields = fields.merge(Self::from_contract_object(json_object(contract_value)?)?)?;
        }
        Ok(fields)
    }

    fn from_contract_object(
        contract: &Map<String, Value>,
    ) -> Result<Self, Qwen3_5MtpContractError> {
        Ok(Self {
            architecture_id: selected_string(contract, &["arch_id", "architecture"])?,
            schema_version: selected_string(contract, &["schema", "schema_version"])?,
            runtime_version: selected_string(contract, &["mtplx_version", "version"])?,
            base_hidden_variant: selected_string(contract, &["base_hidden_variant"])?,
            hidden_variant: selected_string(contract, &["hidden_variant"])?,
            concat_order: selected_string(contract, &["concat_order"])?,
            position_mode: selected_string(contract, &["mtp_position_mode"])?,
            default_depth: selected_integer(contract, &["mtp_depth_default"])?,
            maximum_depth: selected_integer(contract, &["mtp_depth_max"])?,
        })
    }

    fn merge(self, other: Self) -> Result<Self, Qwen3_5MtpContractError> {
        Ok(Self {
            architecture_id: merge_selected(self.architecture_id, other.architecture_id)?,
            schema_version: merge_selected(self.schema_version, other.schema_version)?,
            runtime_version: merge_selected(self.runtime_version, other.runtime_version)?,
            base_hidden_variant: merge_selected(
                self.base_hidden_variant,
                other.base_hidden_variant,
            )?,
            hidden_variant: merge_selected(self.hidden_variant, other.hidden_variant)?,
            concat_order: merge_selected(self.concat_order, other.concat_order)?,
            position_mode: merge_selected(self.position_mode, other.position_mode)?,
            default_depth: merge_selected(self.default_depth, other.default_depth)?,
            maximum_depth: merge_selected(self.maximum_depth, other.maximum_depth)?,
        })
    }

    fn validate_supported_semantics(&self) -> Result<(), Qwen3_5MtpContractError> {
        // Missing vendor identity remains valid for physically complete known Qwen weights. Once
        // an identity is declared, however, it must name the contract whose arithmetic this
        // implementation actually executes; retaining an unknown label would falsely authorize it.
        if self
            .architecture_id
            .as_deref()
            .is_some_and(|architecture_id| architecture_id != SUPPORTED_ARCHITECTURE_ID)
        {
            return Err(Qwen3_5MtpContractError::Incompatible);
        }
        let declared_semantic_field_count = [
            self.base_hidden_variant.as_deref(),
            self.hidden_variant.as_deref(),
            self.concat_order.as_deref(),
            self.position_mode.as_deref(),
        ]
        .into_iter()
        .flatten()
        .count();
        if declared_semantic_field_count == 0 {
            return Ok(());
        }
        if declared_semantic_field_count != 4
            || self.base_hidden_variant.as_deref() != Some("post_norm")
            || self.hidden_variant.as_deref() != Some("post_norm")
            || self.concat_order.as_deref() != Some("embedding_hidden")
            || !matches!(
                self.position_mode.as_deref(),
                Some("local" | "cache_owned" | "mtp_cache_local")
            )
        {
            return Err(Qwen3_5MtpContractError::Incompatible);
        }
        Ok(())
    }

    fn into_contract(self) -> Result<Qwen3_5MtpContract, Qwen3_5MtpContractError> {
        let artifact_default_depth = self.default_depth.map(validated_depth).transpose()?;
        // Vendor maxima above three describe capability Astronomical does not execute;
        // cap them rather than allowing metadata to raise the runtime ceiling.
        let artifact_maximum_depth = self
            .maximum_depth
            .map(|depth| validated_depth(depth.min(u64::from(MtpDraftDepth::MAXIMUM))))
            .transpose()?
            .or(artifact_default_depth);
        if artifact_default_depth > artifact_maximum_depth {
            return Err(Qwen3_5MtpContractError::Incompatible);
        }
        Ok(Qwen3_5MtpContract {
            architecture_id: self.architecture_id,
            schema_version: self.schema_version,
            runtime_version: self.runtime_version,
            artifact_default_depth,
            artifact_maximum_depth,
        })
    }
}

fn json_object(value: &Value) -> Result<&Map<String, Value>, Qwen3_5MtpContractError> {
    value.as_object().ok_or(Qwen3_5MtpContractError::Malformed)
}

fn selected_string(
    object: &Map<String, Value>,
    field_names: &[&str],
) -> Result<Option<String>, Qwen3_5MtpContractError> {
    let mut selected = None;
    for field_name in field_names {
        let Some(value) = object.get(*field_name) else {
            continue;
        };
        let text = value.as_str().ok_or(Qwen3_5MtpContractError::Malformed)?;
        if text.is_empty() || text.len() > MAXIMUM_SELECTED_STRING_BYTES {
            return Err(Qwen3_5MtpContractError::Malformed);
        }
        selected = merge_selected(selected, Some(text.to_owned()))?;
    }
    Ok(selected)
}

fn selected_integer(
    object: &Map<String, Value>,
    field_names: &[&str],
) -> Result<Option<u64>, Qwen3_5MtpContractError> {
    let mut selected = None;
    for field_name in field_names {
        if let Some(value) = object.get(*field_name) {
            selected = merge_selected(
                selected,
                Some(value.as_u64().ok_or(Qwen3_5MtpContractError::Malformed)?),
            )?;
        }
    }
    Ok(selected)
}

fn merge_selected<T: Eq>(
    first: Option<T>,
    second: Option<T>,
) -> Result<Option<T>, Qwen3_5MtpContractError> {
    match (first, second) {
        (Some(first), Some(second)) if first != second => {
            Err(Qwen3_5MtpContractError::FieldDisagreement)
        }
        (Some(first), _) => Ok(Some(first)),
        (_, Some(second)) => Ok(Some(second)),
        (None, None) => Ok(None),
    }
}

fn validated_depth(depth: u64) -> Result<MtpDraftDepth, Qwen3_5MtpContractError> {
    let depth = u8::try_from(depth).map_err(|_| Qwen3_5MtpContractError::Incompatible)?;
    MtpDraftDepth::new(depth).map_err(|_| Qwen3_5MtpContractError::Incompatible)
}
