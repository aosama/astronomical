use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::compressed_storage_descriptor::{
    LagunaCompressedFeedForwardProjection, LagunaCompressedIgnoreScope,
    LagunaCompressedInputActivationDescriptor, LagunaCompressedModuleScope,
    LagunaCompressedStorageDescriptor, LagunaCompressedWeightEncoding,
    LagunaFp8InputActivationDescriptor, LagunaFp8KvCacheDescriptor,
    LagunaNvfp4InputActivationDescriptor,
};
use super::compressed_storage_validation::*;
use super::error::LagunaNormalizationError;
use super::storage_descriptor::{
    LagunaBlockFp8Profile, LagunaNvfp4Profile, LagunaStorageDescriptor,
    LagunaSymmetricPackedAffineProfile,
};

const DENSE_FEED_FORWARD_SELECTOR: &str = "re:.*mlp\\.(gate_proj|up_proj|down_proj)$";
const ROUTED_EXPERT_SELECTOR: &str = "re:.*experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)$";
const SHARED_EXPERT_SELECTOR: &str = "re:.*shared_expert\\.(gate_proj|up_proj|down_proj)$";
const FUSED_ROUTED_EXPERT_SELECTOR: &str = "re:.*mlp\\.(gate_up_proj|down_proj)$";
const FUSED_SHARED_EXPERT_SELECTOR: &str = "re:.*shared_expert\\.(gate_up_proj|down_proj)$";

const OUTPUT_HEAD_IGNORE: &str = "lm_head";
const ATTENTION_QUERY_IGNORE: &str = "re:.*\\.self_attn\\.q_proj$";
const ATTENTION_KEY_IGNORE: &str = "re:.*\\.self_attn\\.k_proj$";
const ATTENTION_VALUE_IGNORE: &str = "re:.*\\.self_attn\\.v_proj$";
const ATTENTION_OUTPUT_IGNORE: &str = "re:.*\\.self_attn\\.o_proj$";
const ATTENTION_GATE_IGNORE: &str = "re:.*\\.self_attn\\.g_proj$";
const ROUTER_IGNORE: &str = "re:.*\\.mlp\\.gate$";
const SHARED_EXPERT_GATE_IGNORE: &str = "re:.*\\.mlp\\.shared_expert\\.gate_proj$";
const SHARED_EXPERT_UP_IGNORE: &str = "re:.*\\.mlp\\.shared_expert\\.up_proj$";
const SHARED_EXPERT_DOWN_IGNORE: &str = "re:.*\\.mlp\\.shared_expert\\.down_proj$";

const TOP_LEVEL_FIELDS: &[&str] = &[
    "config_groups",
    "format",
    "global_compression_ratio",
    "ignore",
    "kv_cache_scheme",
    "quant_method",
    "quantization_status",
    "sparsity_config",
    "transform_config",
    "version",
];
const GROUP_FIELDS: &[&str] = &[
    "format",
    "input_activations",
    "output_activations",
    "targets",
    "weights",
];
const SCHEME_FIELDS: &[&str] = &[
    "actorder",
    "block_structure",
    "dynamic",
    "group_size",
    "num_bits",
    "observer",
    "observer_kwargs",
    "scale_dtype",
    "strategy",
    "symmetric",
    "type",
    "zp_dtype",
];

pub(super) fn normalize_compressed_storage(
    quantization_fields: &Map<String, Value>,
) -> Result<LagunaStorageDescriptor, LagunaNormalizationError> {
    reject_unknown_fields(quantization_fields, TOP_LEVEL_FIELDS, "quantization")?;
    validate_exact_string(
        quantization_fields,
        "quant_method",
        "compressed-tensors",
        "quantization",
    )?;
    validate_optional_string(
        quantization_fields,
        "quantization_status",
        "compressed",
        "quantization",
    )?;
    validate_optional_null(
        quantization_fields,
        "global_compression_ratio",
        "quantization",
    )?;
    validate_optional_empty_object(quantization_fields, "sparsity_config", "quantization")?;
    validate_optional_empty_object(quantization_fields, "transform_config", "quantization")?;
    validate_optional_nonempty_string(quantization_fields, "version", "quantization")?;

    let group_fields = required_single_group(quantization_fields)?;
    reject_unknown_fields(
        group_fields,
        GROUP_FIELDS,
        "quantization.config_groups.group_0",
    )?;
    validate_optional_null(
        group_fields,
        "output_activations",
        "quantization.config_groups.group_0",
    )?;
    let format = reconcile_format(quantization_fields, group_fields)?;
    let weight_fields = required_object(
        group_fields,
        "weights",
        "quantization.config_groups.group_0",
    )?;
    reject_unknown_fields(
        weight_fields,
        SCHEME_FIELDS,
        "quantization.config_groups.group_0.weights",
    )?;

    let weight_encoding = normalize_weight_encoding(format, weight_fields)?;
    let input_activations = normalize_input_activations(format, group_fields)?;
    let target_scopes = normalize_target_scopes(group_fields)?;
    let ignored_scopes = normalize_ignored_scopes(quantization_fields)?;
    let kv_cache = normalize_kv_cache(quantization_fields)?;
    Ok(LagunaStorageDescriptor::Compressed(
        LagunaCompressedStorageDescriptor::new(
            weight_encoding,
            target_scopes,
            ignored_scopes,
            input_activations,
            kv_cache,
        ),
    ))
}

fn normalize_weight_encoding(
    format: &str,
    fields: &Map<String, Value>,
) -> Result<LagunaCompressedWeightEncoding, LagunaNormalizationError> {
    let location = "quantization.config_groups.group_0.weights";
    match format {
        "pack-quantized" => {
            validate_exact_u32(fields, "num_bits", 4, location)?;
            validate_exact_u32(fields, "group_size", 32, location)?;
            validate_exact_string(fields, "type", "int", location)?;
            validate_optional_true(fields, "symmetric", location)?;
            validate_optional_false(fields, "dynamic", location)?;
            validate_optional_string(fields, "strategy", "group", location)?;
            validate_optional_string(fields, "observer", "memoryless_minmax", location)?;
            validate_optional_null(fields, "scale_dtype", location)?;
            validate_optional_null(fields, "block_structure", location)?;
            validate_common_null_defaults(fields, location)?;
            Ok(LagunaCompressedWeightEncoding::SymmetricPackedAffine(
                LagunaSymmetricPackedAffineProfile::evidenced(),
            ))
        }
        "nvfp4-pack-quantized" => {
            validate_exact_u32(fields, "num_bits", 4, location)?;
            validate_exact_u32(fields, "group_size", 16, location)?;
            validate_optional_string(fields, "type", "float", location)?;
            validate_optional_true(fields, "symmetric", location)?;
            validate_optional_false(fields, "dynamic", location)?;
            validate_optional_string(fields, "strategy", "tensor_group", location)?;
            validate_optional_string(fields, "observer", "memoryless_minmax", location)?;
            validate_optional_string(fields, "scale_dtype", "torch.float8_e4m3fn", location)?;
            validate_optional_null(fields, "block_structure", location)?;
            validate_common_null_defaults(fields, location)?;
            Ok(LagunaCompressedWeightEncoding::TwoLevelNvfp4(
                LagunaNvfp4Profile::two_level(),
            ))
        }
        "float-quantized" => {
            validate_exact_u32(fields, "num_bits", 8, location)?;
            validate_exact_string(fields, "type", "float", location)?;
            validate_optional_true(fields, "symmetric", location)?;
            validate_optional_false(fields, "dynamic", location)?;
            validate_optional_string(fields, "strategy", "block", location)?;
            validate_optional_string(fields, "observer", "memoryless_minmax", location)?;
            validate_optional_null(fields, "group_size", location)?;
            validate_optional_null(fields, "scale_dtype", location)?;
            validate_optional_block_128(fields, location)?;
            validate_common_null_defaults(fields, location)?;
            Ok(LagunaCompressedWeightEncoding::BlockFp8(
                LagunaBlockFp8Profile::evidenced(),
            ))
        }
        unsupported_format => Err(LagunaNormalizationError::UnsupportedStorageEncoding {
            encoding: super::document::bounded_text_label(unsupported_format),
        }),
    }
}

fn normalize_input_activations(
    format: &str,
    group_fields: &Map<String, Value>,
) -> Result<Option<LagunaCompressedInputActivationDescriptor>, LagunaNormalizationError> {
    let Some(activation_value) = group_fields.get("input_activations") else {
        return Ok(None);
    };
    if activation_value.is_null() {
        return Ok(None);
    }
    let fields = activation_value.as_object().ok_or_else(|| {
        unsupported(
            "quantization.config_groups.group_0.input_activations",
            "object or null",
            activation_value,
        )
    })?;
    let location = "quantization.config_groups.group_0.input_activations";
    reject_unknown_fields(fields, SCHEME_FIELDS, location)?;
    match format {
        "nvfp4-pack-quantized" => {
            validate_exact_u32(fields, "num_bits", 4, location)?;
            validate_exact_u32(fields, "group_size", 16, location)?;
            validate_exact_string(fields, "type", "float", location)?;
            validate_exact_string(fields, "dynamic", "local", location)?;
            validate_exact_string(fields, "strategy", "tensor_group", location)?;
            validate_exact_string(fields, "observer", "static_minmax", location)?;
            validate_exact_string(fields, "scale_dtype", "torch.float8_e4m3fn", location)?;
            validate_exact_true(fields, "symmetric", location)?;
            validate_optional_null(fields, "block_structure", location)?;
            validate_common_null_defaults(fields, location)?;
            Ok(Some(
                LagunaCompressedInputActivationDescriptor::Nvfp4TensorGroup(
                    LagunaNvfp4InputActivationDescriptor::evidenced(),
                ),
            ))
        }
        "float-quantized" => {
            validate_exact_u32(fields, "num_bits", 8, location)?;
            validate_exact_u32(fields, "group_size", 128, location)?;
            validate_exact_string(fields, "type", "float", location)?;
            validate_exact_true(fields, "dynamic", location)?;
            validate_exact_string(fields, "strategy", "group", location)?;
            validate_optional_null(fields, "observer", location)?;
            validate_optional_null(fields, "scale_dtype", location)?;
            validate_exact_true(fields, "symmetric", location)?;
            validate_optional_null(fields, "block_structure", location)?;
            validate_common_null_defaults(fields, location)?;
            Ok(Some(LagunaCompressedInputActivationDescriptor::Fp8Group(
                LagunaFp8InputActivationDescriptor::evidenced(),
            )))
        }
        "pack-quantized" => Err(unsupported(
            location,
            "null input activations",
            activation_value,
        )),
        unsupported_format => Err(LagunaNormalizationError::UnsupportedStorageEncoding {
            encoding: super::document::bounded_text_label(unsupported_format),
        }),
    }
}

fn normalize_kv_cache(
    quantization_fields: &Map<String, Value>,
) -> Result<Option<LagunaFp8KvCacheDescriptor>, LagunaNormalizationError> {
    let Some(kv_cache_value) = quantization_fields.get("kv_cache_scheme") else {
        return Ok(None);
    };
    if kv_cache_value.is_null() {
        return Ok(None);
    }
    let fields = kv_cache_value.as_object().ok_or_else(|| {
        unsupported(
            "quantization.kv_cache_scheme",
            "object or null",
            kv_cache_value,
        )
    })?;
    let location = "quantization.kv_cache_scheme";
    reject_unknown_fields(fields, SCHEME_FIELDS, location)?;
    validate_exact_u32(fields, "num_bits", 8, location)?;
    validate_exact_string(fields, "type", "float", location)?;
    validate_exact_false(fields, "dynamic", location)?;
    validate_exact_string(fields, "strategy", "tensor", location)?;
    validate_exact_string(fields, "observer", "minmax", location)?;
    validate_exact_true(fields, "symmetric", location)?;
    validate_optional_null(fields, "group_size", location)?;
    validate_optional_null(fields, "scale_dtype", location)?;
    validate_optional_null(fields, "block_structure", location)?;
    validate_common_null_defaults(fields, location)?;
    Ok(Some(LagunaFp8KvCacheDescriptor::evidenced()))
}

fn normalize_target_scopes(
    group_fields: &Map<String, Value>,
) -> Result<BTreeSet<LagunaCompressedModuleScope>, LagunaNormalizationError> {
    let Some(targets_value) = group_fields.get("targets") else {
        return Ok(BTreeSet::from([LagunaCompressedModuleScope::AllMatrices]));
    };
    let target_values = targets_value.as_array().ok_or_else(|| {
        unsupported(
            "quantization.config_groups.group_0.targets",
            "array",
            targets_value,
        )
    })?;
    let mut target_scopes = BTreeSet::new();
    for target_value in target_values {
        let target = target_value.as_str().ok_or_else(|| {
            unsupported(
                "quantization.config_groups.group_0.targets",
                "string selector",
                target_value,
            )
        })?;
        let scope = match target {
            "Linear" => LagunaCompressedModuleScope::AllLinear,
            DENSE_FEED_FORWARD_SELECTOR => LagunaCompressedModuleScope::DenseFeedForward,
            ROUTED_EXPERT_SELECTOR | FUSED_ROUTED_EXPERT_SELECTOR => {
                LagunaCompressedModuleScope::RoutedExperts
            }
            SHARED_EXPERT_SELECTOR | FUSED_SHARED_EXPERT_SELECTOR => {
                LagunaCompressedModuleScope::SharedExpert
            }
            _ => {
                return Err(unsupported(
                    "quantization.config_groups.group_0.targets",
                    "evidenced structured selector",
                    target_value,
                ));
            }
        };
        target_scopes.insert(scope);
    }
    if target_scopes.is_empty() {
        return Err(unsupported(
            "quantization.config_groups.group_0.targets",
            "non-empty selector array",
            targets_value,
        ));
    }
    Ok(target_scopes)
}

fn normalize_ignored_scopes(
    quantization_fields: &Map<String, Value>,
) -> Result<BTreeSet<LagunaCompressedIgnoreScope>, LagunaNormalizationError> {
    let Some(ignore_value) = quantization_fields.get("ignore") else {
        return Ok(BTreeSet::new());
    };
    let ignored_values = ignore_value
        .as_array()
        .ok_or_else(|| unsupported("quantization.ignore", "array", ignore_value))?;
    let mut ignored_scopes = BTreeSet::new();
    for ignored_value in ignored_values {
        let ignored_selector = ignored_value
            .as_str()
            .ok_or_else(|| unsupported("quantization.ignore", "string selector", ignored_value))?;
        let scope = match ignored_selector {
            OUTPUT_HEAD_IGNORE => LagunaCompressedIgnoreScope::OutputHead,
            ATTENTION_QUERY_IGNORE => LagunaCompressedIgnoreScope::AttentionQuery,
            ATTENTION_KEY_IGNORE => LagunaCompressedIgnoreScope::AttentionKey,
            ATTENTION_VALUE_IGNORE => LagunaCompressedIgnoreScope::AttentionValue,
            ATTENTION_OUTPUT_IGNORE => LagunaCompressedIgnoreScope::AttentionOutput,
            ATTENTION_GATE_IGNORE => LagunaCompressedIgnoreScope::AttentionGate,
            ROUTER_IGNORE => LagunaCompressedIgnoreScope::Router,
            SHARED_EXPERT_GATE_IGNORE | SHARED_EXPERT_UP_IGNORE | SHARED_EXPERT_DOWN_IGNORE => {
                LagunaCompressedIgnoreScope::SharedExpert
            }
            _ => parse_dense_feed_forward_ignore(ignored_selector).ok_or_else(|| {
                unsupported(
                    "quantization.ignore",
                    "evidenced structured selector",
                    ignored_value,
                )
            })?,
        };
        ignored_scopes.insert(scope);
    }
    Ok(ignored_scopes)
}

fn parse_dense_feed_forward_ignore(ignored_selector: &str) -> Option<LagunaCompressedIgnoreScope> {
    let path_parts = ignored_selector.split('.').collect::<Vec<_>>();
    let ["model", "layers", layer_index, "mlp", projection_name] = path_parts.as_slice() else {
        return None;
    };
    let layer_index = layer_index.parse::<usize>().ok()?;
    let projection = match *projection_name {
        "gate_proj" => LagunaCompressedFeedForwardProjection::Gate,
        "up_proj" => LagunaCompressedFeedForwardProjection::Up,
        "down_proj" => LagunaCompressedFeedForwardProjection::Down,
        _ => return None,
    };
    Some(LagunaCompressedIgnoreScope::DenseFeedForward {
        layer_index,
        projection,
    })
}

fn required_single_group(
    quantization_fields: &Map<String, Value>,
) -> Result<&Map<String, Value>, LagunaNormalizationError> {
    let groups = required_object(quantization_fields, "config_groups", "quantization")?;
    if groups.len() != 1 || !groups.contains_key("group_0") {
        return Err(unsupported(
            "quantization.config_groups",
            "exactly group_0",
            quantization_fields
                .get("config_groups")
                .unwrap_or(&Value::Null),
        ));
    }
    required_object(groups, "group_0", "quantization.config_groups")
}

fn reconcile_format<'a>(
    quantization_fields: &'a Map<String, Value>,
    group_fields: &'a Map<String, Value>,
) -> Result<&'a str, LagunaNormalizationError> {
    let top_level_format = optional_string(quantization_fields, "format", "quantization")?;
    let group_format =
        optional_string(group_fields, "format", "quantization.config_groups.group_0")?;
    if top_level_format.is_some() && group_format.is_some() && top_level_format != group_format {
        return Err(LagunaNormalizationError::ConflictingQuantizationDocuments);
    }
    top_level_format.or(group_format).ok_or_else(|| {
        LagunaNormalizationError::UnsupportedStorageEncoding {
            encoding: "missing compressed-tensors format".to_owned(),
        }
    })
}
