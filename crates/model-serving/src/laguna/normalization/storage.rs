use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::{
    compressed_storage::normalize_compressed_storage,
    document::{bounded_text_label, bounded_value_label},
    error::LagunaNormalizationError,
    storage_descriptor::{
        LagunaAffineProfile, LagunaDirectAffineStorageDescriptor, LagunaNvfp4Profile,
        LagunaStorageDescriptor,
    },
};

const PROFILE_FIELD_NAMES: [&str; 4] = ["bits", "group_size", "mode", "quant_method"];

pub(super) fn normalize_storage(
    quantization_documents: &[Value],
) -> Result<LagunaStorageDescriptor, LagunaNormalizationError> {
    if quantization_documents.is_empty() {
        return Ok(LagunaStorageDescriptor::Unquantized);
    }
    let mut canonical_document: Option<LagunaStorageDescriptor> = None;
    for (document_index, quantization_value) in quantization_documents.iter().enumerate() {
        let parsed_document = parse_storage_document(quantization_value, document_index)?;
        if let Some(existing_document) = &canonical_document
            && existing_document != &parsed_document
        {
            return Err(LagunaNormalizationError::ConflictingQuantizationDocuments);
        }
        canonical_document = Some(parsed_document);
    }
    canonical_document.ok_or_else(|| LagunaNormalizationError::MissingRequiredField {
        field_name: "quantization".to_owned(),
    })
}

fn parse_storage_document(
    quantization_value: &Value,
    document_index: usize,
) -> Result<LagunaStorageDescriptor, LagunaNormalizationError> {
    let quantization_fields =
        quantization_value
            .as_object()
            .ok_or_else(|| LagunaNormalizationError::ExpectedObject {
                field_name: format!("quantization[{document_index}]"),
            })?;
    if quantization_fields
        .get("quant_method")
        .and_then(Value::as_str)
        == Some("compressed-tensors")
    {
        return normalize_compressed_storage(quantization_fields);
    }
    if quantization_fields.get("mode").and_then(Value::as_str) == Some("nvfp4") {
        reject_unknown_fields(
            quantization_fields,
            &["bits", "group_size", "mode"],
            "quantization",
        )?;
        validate_exact_u32(quantization_fields, "bits", 4)?;
        validate_exact_u32(quantization_fields, "group_size", 16)?;
        return Ok(LagunaStorageDescriptor::NativeNvfp4(
            LagunaNvfp4Profile::native(),
        ));
    }
    parse_affine_document(quantization_fields).map(LagunaStorageDescriptor::DirectAffine)
}

fn parse_affine_document(
    quantization_fields: &Map<String, Value>,
) -> Result<LagunaDirectAffineStorageDescriptor, LagunaNormalizationError> {
    reject_inventory_dependent_encoding(quantization_fields)?;
    let default_profile = parse_profile(quantization_fields, "quantization")?;
    let mut module_overrides = BTreeMap::new();
    for (artifact_module_name, override_value) in quantization_fields {
        if PROFILE_FIELD_NAMES.contains(&artifact_module_name.as_str()) {
            continue;
        }
        let override_location =
            format!("quantization.{}", bounded_text_label(artifact_module_name));
        let override_fields = override_value.as_object().ok_or_else(|| {
            LagunaNormalizationError::UnsupportedQuantizationValue {
                location: override_location.clone(),
                description: "module override",
                actual_value: bounded_value_label(override_value),
            }
        })?;
        let canonical_module_name = canonicalize_module_name(artifact_module_name)?;
        let override_profile = parse_profile(override_fields, &override_location)?;
        if let Some(existing_profile) =
            module_overrides.insert(canonical_module_name.clone(), override_profile)
            && existing_profile != override_profile
        {
            return Err(LagunaNormalizationError::ConflictingModuleOverride {
                module_name: bounded_text_label(&canonical_module_name),
            });
        }
    }
    Ok(LagunaDirectAffineStorageDescriptor::new(
        default_profile,
        module_overrides,
    ))
}

fn reject_unknown_fields(
    fields: &Map<String, Value>,
    supported_fields: &[&str],
    location: &str,
) -> Result<(), LagunaNormalizationError> {
    if let Some(unknown_field) = fields
        .keys()
        .find(|field_name| !supported_fields.contains(&field_name.as_str()))
    {
        return Err(LagunaNormalizationError::UnsupportedQuantizationValue {
            location: location.to_owned(),
            description: "field",
            actual_value: unknown_field.clone(),
        });
    }
    Ok(())
}

fn validate_exact_u32(
    fields: &Map<String, Value>,
    field_name: &str,
    expected_value: u32,
) -> Result<(), LagunaNormalizationError> {
    if fields.get(field_name).and_then(Value::as_u64) != Some(u64::from(expected_value)) {
        return Err(LagunaNormalizationError::UnsupportedQuantizationValue {
            location: format!("quantization.{field_name}"),
            description: "exact evidenced value",
            actual_value: fields
                .get(field_name)
                .map(value_label)
                .unwrap_or_else(|| "missing".to_owned()),
        });
    }
    Ok(())
}

fn reject_inventory_dependent_encoding(
    quantization_fields: &Map<String, Value>,
) -> Result<(), LagunaNormalizationError> {
    if let Some(format_value) = quantization_fields.get("format") {
        return Err(LagunaNormalizationError::UnsupportedStorageEncoding {
            encoding: value_label(format_value),
        });
    }
    if let Some(quantization_method) = quantization_fields.get("quant_method") {
        let method_name = quantization_method.as_str().unwrap_or("non-string");
        if !matches!(method_name, "mlx" | "affine") {
            return Err(LagunaNormalizationError::UnsupportedStorageEncoding {
                encoding: value_label(quantization_method),
            });
        }
    }
    Ok(())
}

fn parse_profile(
    profile_fields: &Map<String, Value>,
    location: &str,
) -> Result<LagunaAffineProfile, LagunaNormalizationError> {
    reject_inventory_dependent_encoding(profile_fields)?;
    if let Some(mode_value) = profile_fields.get("mode")
        && mode_value.as_str() != Some("affine")
    {
        return Err(LagunaNormalizationError::UnsupportedQuantizationValue {
            location: format!("{location}.mode"),
            description: "mode",
            actual_value: value_label(mode_value),
        });
    }
    let bits = profile_u32(profile_fields, "bits", location)?;
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return Err(LagunaNormalizationError::UnsupportedQuantizationValue {
            location: format!("{location}.bits"),
            description: "bit width",
            actual_value: bits.to_string(),
        });
    }
    let group_size = profile_u32(profile_fields, "group_size", location)?;
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(LagunaNormalizationError::UnsupportedQuantizationValue {
            location: format!("{location}.group_size"),
            description: "group size",
            actual_value: group_size.to_string(),
        });
    }
    Ok(LagunaAffineProfile::new(bits, group_size))
}

fn profile_u32(
    profile_fields: &Map<String, Value>,
    field_name: &str,
    location: &str,
) -> Result<u32, LagunaNormalizationError> {
    let field_value = profile_fields.get(field_name).ok_or_else(|| {
        LagunaNormalizationError::UnsupportedQuantizationValue {
            location: location.to_owned(),
            description: "missing field",
            actual_value: field_name.to_owned(),
        }
    })?;
    let unsigned_value = field_value.as_u64().ok_or_else(|| {
        LagunaNormalizationError::UnsupportedQuantizationValue {
            location: format!("{location}.{field_name}"),
            description: "integer",
            actual_value: value_label(field_value),
        }
    })?;
    u32::try_from(unsigned_value).map_err(|_| {
        LagunaNormalizationError::UnsupportedQuantizationValue {
            location: format!("{location}.{field_name}"),
            description: "integer range",
            actual_value: unsigned_value.to_string(),
        }
    })
}

fn canonicalize_module_name(
    artifact_module_name: &str,
) -> Result<String, LagunaNormalizationError> {
    let canonical_module_name = artifact_module_name
        .strip_prefix("language_model.")
        .unwrap_or(artifact_module_name);
    if canonical_module_name.starts_with("language_model.")
        || !(canonical_module_name == "lm_head" || canonical_module_name.starts_with("model."))
    {
        return Err(LagunaNormalizationError::UnsupportedQuantizationValue {
            location: "quantization override".to_owned(),
            description: "module name",
            actual_value: bounded_text_label(artifact_module_name),
        });
    }
    let canonical_module_name = if canonical_module_name.ends_with(".mlp.gate") {
        format!("{canonical_module_name}.proj")
    } else {
        canonical_module_name.to_owned()
    };
    Ok(canonical_module_name)
}

fn value_label(field_value: &Value) -> String {
    bounded_value_label(field_value)
}
