//! Per-tensor shape contract for the folded pack inventory.
//!
//! Every language-side tensor in the pack must be classified as either a
//! packed module member (rotated ternary projection) or a recognized
//! auxiliary weight, with exact dtype and shape rules derived from the
//! validated configuration. Anything else is a refusal.

use std::collections::{BTreeMap, BTreeSet};

use ::safetensors::Dtype;

use crate::artifact_validation::RawSafetensorsInventory;

use super::super::configuration::{FoldedPackConfig, FoldedTextDimensions, PackedModule};
use super::super::hadamard::HadamardContract;
use super::FoldedPackValidationError;

/// The language-side tensor prefix every classified tensor must carry.
const LANGUAGE_TENSOR_PREFIX: &str = "language_model.";
/// The vision-side tensor prefix, required only when the pack declares vision.
const VISION_TENSOR_PREFIX: &str = "vision_tower.";

/// Which packed module a record path declares, fixing both packed shapes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedModuleKind {
    LanguageModelHead,
    TokenEmbedding,
    GatedQuery,
    Key,
    Value,
    AttentionOutput,
    GdnQueryKeyValue,
    GdnGate,
    GdnOutput,
    MlpGate,
    MlpUp,
    MlpDown,
}

fn classify_packed_module(
    module_path: &str,
) -> Result<PackedModuleKind, FoldedPackValidationError> {
    let kind = match module_path {
        "lm_head" => PackedModuleKind::LanguageModelHead,
        "model.embed_tokens" => PackedModuleKind::TokenEmbedding,
        _ => {
            let path_segments: Vec<&str> = module_path.split('.').collect();
            match path_segments.as_slice() {
                [.., "self_attn", "q_proj"] => PackedModuleKind::GatedQuery,
                [.., "self_attn", "k_proj"] => PackedModuleKind::Key,
                [.., "self_attn", "v_proj"] => PackedModuleKind::Value,
                [.., "self_attn", "o_proj"] => PackedModuleKind::AttentionOutput,
                [.., "linear_attn", "in_proj_qkv"] => PackedModuleKind::GdnQueryKeyValue,
                [.., "linear_attn", "in_proj_z"] => PackedModuleKind::GdnGate,
                [.., "linear_attn", "out_proj"] => PackedModuleKind::GdnOutput,
                [.., "mlp", "gate_proj"] => PackedModuleKind::MlpGate,
                [.., "mlp", "up_proj"] => PackedModuleKind::MlpUp,
                [.., "mlp", "down_proj"] => PackedModuleKind::MlpDown,
                _ => {
                    return Err(FoldedPackValidationError::ContractModuleDisagreement {
                        detail: format!(
                            "module path {module_path:?} is not a recognized packed projection"
                        ),
                    });
                }
            }
        }
    };
    Ok(kind)
}

/// The logical `[rows, input_width]` of a packed module before ternary packing.
fn packed_module_dimensions(kind: PackedModuleKind, dims: &FoldedTextDimensions) -> (u32, u32) {
    match kind {
        PackedModuleKind::LanguageModelHead | PackedModuleKind::TokenEmbedding => {
            (dims.vocab_size, dims.hidden_size)
        }
        // The packed query carries the swish output gate alongside the
        // queries, doubling its row count.
        PackedModuleKind::GatedQuery => (
            2 * dims.num_attention_heads * dims.head_dim,
            dims.hidden_size,
        ),
        PackedModuleKind::Key | PackedModuleKind::Value => {
            (dims.num_key_value_heads * dims.head_dim, dims.hidden_size)
        }
        PackedModuleKind::AttentionOutput => {
            (dims.hidden_size, dims.num_attention_heads * dims.head_dim)
        }
        PackedModuleKind::GdnQueryKeyValue => (
            2 * dims.linear_num_key_heads * dims.linear_key_head_dim
                + dims.linear_num_value_heads * dims.linear_value_head_dim,
            dims.hidden_size,
        ),
        PackedModuleKind::GdnGate => (
            dims.linear_num_value_heads * dims.linear_value_head_dim,
            dims.hidden_size,
        ),
        // The grouped GDN value layout means the output projection consumes
        // the value-head width directly, matching the shipped gdn_v_grouped
        // contract.
        PackedModuleKind::GdnOutput => (
            dims.hidden_size,
            dims.linear_num_value_heads * dims.linear_value_head_dim,
        ),
        PackedModuleKind::MlpGate | PackedModuleKind::MlpUp => {
            (dims.intermediate_size, dims.hidden_size)
        }
        PackedModuleKind::MlpDown => (dims.hidden_size, dims.intermediate_size),
    }
}

/// Validates the complete pack inventory: packed modules, auxiliary weights,
/// manifest agreement, and the strict no-unrecognized-tensor rule.
pub(super) fn validate_pack_inventory(
    config: &FoldedPackConfig,
    hadamard_contract: &HadamardContract,
    weights_inventory: &RawSafetensorsInventory,
) -> Result<(), FoldedPackValidationError> {
    validate_complete_packed_module_inventory(config)?;
    let tensor_declarations = weights_inventory
        .tensor_descriptors
        .iter()
        .map(|descriptor| {
            (
                descriptor.tensor_name.as_str(),
                (descriptor.dtype, descriptor.shape.as_slice()),
            )
        })
        .collect::<BTreeMap<&str, (Dtype, &[usize])>>();

    let mut recognized_tensor_names = BTreeSet::new();
    for module in config.packed_modules() {
        validate_packed_module(
            module,
            config.text(),
            hadamard_contract,
            &tensor_declarations,
        )?;
        for suffix in ["weight", "scales", "biases", "signs"] {
            recognized_tensor_names.insert(format!("{}.{}", module.tensor_stem(), suffix));
        }
    }
    for (auxiliary_name, (expected_dtype, expected_shape)) in
        expected_auxiliary_tensor_shapes(config.text())
    {
        validate_tensor_declaration(
            &auxiliary_name,
            expected_dtype,
            &expected_shape,
            &tensor_declarations,
        )?;
        recognized_tensor_names.insert(auxiliary_name);
    }

    validate_manifest_module_agreement(config, hadamard_contract)?;

    let is_unrecognized_language_tensor = |tensor_name: &str| -> bool {
        tensor_name.starts_with(LANGUAGE_TENSOR_PREFIX)
            && !recognized_tensor_names.contains(tensor_name)
    };
    if let Some(first_unrecognized) = tensor_declarations
        .keys()
        .copied()
        .find(|tensor_name| is_unrecognized_language_tensor(tensor_name))
    {
        return Err(FoldedPackValidationError::UnrecognizedLanguageTensor {
            tensor_name: first_unrecognized.to_owned(),
        });
    }

    let vision_tensor_count = tensor_declarations
        .keys()
        .filter(|tensor_name| tensor_name.starts_with(VISION_TENSOR_PREFIX))
        .count();
    if config.vision_declared() && vision_tensor_count == 0 {
        return Err(FoldedPackValidationError::ContractModuleDisagreement {
            detail: "the pack declares the vision component but ships no vision tower tensors"
                .to_owned(),
        });
    }
    if !config.vision_declared() && vision_tensor_count > 0 {
        return Err(FoldedPackValidationError::UnrecognizedLanguageTensor {
            tensor_name: VISION_TENSOR_PREFIX.to_owned() + "…",
        });
    }
    Ok(())
}

fn validate_complete_packed_module_inventory(
    config: &FoldedPackConfig,
) -> Result<(), FoldedPackValidationError> {
    let mut expected_module_paths =
        BTreeSet::from(["lm_head".to_owned(), "model.embed_tokens".to_owned()]);
    for (layer_index, layer_type) in config.text().layer_types.iter().enumerate() {
        let layer_prefix = format!("model.layers.{layer_index}");
        match layer_type.as_str() {
            "linear_attention" => {
                for projection_name in [
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.out_proj",
                ] {
                    expected_module_paths.insert(format!("{layer_prefix}.{projection_name}"));
                }
            }
            "full_attention" => {
                for projection_name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    expected_module_paths
                        .insert(format!("{layer_prefix}.self_attn.{projection_name}"));
                }
            }
            _ => unreachable!("configuration validation already pins the layer vocabulary"),
        }
        for projection_name in ["gate_proj", "up_proj", "down_proj"] {
            expected_module_paths.insert(format!("{layer_prefix}.mlp.{projection_name}"));
        }
    }
    let declared_module_paths = config
        .packed_modules()
        .iter()
        .map(|packed_module| packed_module.path().to_owned())
        .collect::<BTreeSet<_>>();
    if expected_module_paths == declared_module_paths {
        return Ok(());
    }

    let missing_module_paths = expected_module_paths
        .difference(&declared_module_paths)
        .take(3)
        .collect::<Vec<_>>();
    let unexpected_module_paths = declared_module_paths
        .difference(&expected_module_paths)
        .take(3)
        .collect::<Vec<_>>();
    Err(FoldedPackValidationError::ContractModuleDisagreement {
        detail: format!(
            "missing canonical modules: {missing_module_paths:?}; unexpected modules: {unexpected_module_paths:?}"
        ),
    })
}

fn validate_packed_module(
    module: &PackedModule,
    dims: &FoldedTextDimensions,
    hadamard_contract: &HadamardContract,
    tensor_declarations: &BTreeMap<&str, (Dtype, &[usize])>,
) -> Result<(), FoldedPackValidationError> {
    let kind = classify_packed_module(module.path())?;
    let (rows, input_width) = packed_module_dimensions(kind, dims);
    let words_per_row = input_width / 16;
    let groups_per_row = input_width / 128;
    if input_width % 128 != 0 || input_width % 16 != 0 {
        return Err(FoldedPackValidationError::ContractModuleDisagreement {
            detail: format!(
                "module {} input width {input_width} does not divide into ternary groups",
                module.path()
            ),
        });
    }
    let block_size = hadamard_contract.block_size();
    if input_width % block_size != 0 {
        // The runtime reshapes each activation row into whole Hadamard blocks;
        // refuse this contract here rather than defer it to a serving-time reshape.
        return Err(FoldedPackValidationError::ContractModuleDisagreement {
            detail: format!(
                "module {} input width {input_width} does not divide into Hadamard blocks of {block_size}",
                module.path()
            ),
        });
    }
    if hadamard_contract.sign_vector(input_width).is_none() {
        return Err(FoldedPackValidationError::ContractModuleDisagreement {
            detail: format!(
                "module {} input width {input_width} has no declared sign vector",
                module.path()
            ),
        });
    }

    let weight_tensor_name = format!("{}.weight", module.tensor_stem());
    validate_tensor_declaration(
        &weight_tensor_name,
        Dtype::U32,
        &[rows as usize, words_per_row as usize],
        tensor_declarations,
    )?;
    let scales_tensor_name = format!("{}.scales", module.tensor_stem());
    validate_tensor_declaration(
        &scales_tensor_name,
        Dtype::F16,
        &[rows as usize, groups_per_row as usize],
        tensor_declarations,
    )?;
    let biases_tensor_name = format!("{}.biases", module.tensor_stem());
    validate_tensor_declaration(
        &biases_tensor_name,
        Dtype::F16,
        &[rows as usize, groups_per_row as usize],
        tensor_declarations,
    )?;
    let signs_tensor_name = format!("{}.signs", module.tensor_stem());
    validate_tensor_declaration(
        &signs_tensor_name,
        Dtype::F32,
        &[input_width as usize],
        tensor_declarations,
    )?;

    let weight_tensor_is_inverse = hadamard_contract.is_inverse(&weight_tensor_name);
    let weight_tensor_is_rotated = hadamard_contract.is_rotated(&weight_tensor_name);
    if module.is_embedding() != weight_tensor_is_inverse
        || weight_tensor_is_inverse == weight_tensor_is_rotated
    {
        return Err(FoldedPackValidationError::ContractModuleDisagreement {
            detail: format!(
                "module {} transform direction disagrees with the hadamard manifest",
                module.path()
            ),
        });
    }
    if module.block() != hadamard_contract.block_size() {
        return Err(FoldedPackValidationError::ModuleBlockMismatch {
            module_path: module.path().to_owned(),
            module_block: module.block(),
            manifest_block: hadamard_contract.block_size(),
        });
    }
    Ok(())
}

/// Every auxiliary (non-packed) language tensor the family requires, with
/// exact dtype and shape, keyed by full tensor name.
fn expected_auxiliary_tensor_shapes(
    dims: &FoldedTextDimensions,
) -> BTreeMap<String, (Dtype, Vec<usize>)> {
    let mut auxiliary_tensors = BTreeMap::new();
    auxiliary_tensors.insert(
        "language_model.model.norm.weight".to_owned(),
        (Dtype::F32, vec![dims.hidden_size as usize]),
    );
    for layer_index in 0..dims.num_hidden_layers {
        let layer_prefix = format!("language_model.model.layers.{layer_index}.");
        auxiliary_tensors.insert(
            format!("{layer_prefix}input_layernorm.weight"),
            (Dtype::F32, vec![dims.hidden_size as usize]),
        );
        auxiliary_tensors.insert(
            format!("{layer_prefix}post_attention_layernorm.weight"),
            (Dtype::F32, vec![dims.hidden_size as usize]),
        );
        match dims.layer_types[layer_index as usize].as_str() {
            "full_attention" => {
                auxiliary_tensors.insert(
                    format!("{layer_prefix}self_attn.q_norm.weight"),
                    (Dtype::F32, vec![dims.head_dim as usize]),
                );
                auxiliary_tensors.insert(
                    format!("{layer_prefix}self_attn.k_norm.weight"),
                    (Dtype::F32, vec![dims.head_dim as usize]),
                );
            }
            _ => {
                auxiliary_tensors.insert(
                    format!("{layer_prefix}linear_attn.A_log"),
                    (Dtype::F32, vec![dims.linear_num_value_heads as usize]),
                );
                auxiliary_tensors.insert(
                    format!("{layer_prefix}linear_attn.dt_bias"),
                    (Dtype::F32, vec![dims.linear_num_value_heads as usize]),
                );
                auxiliary_tensors.insert(
                    format!("{layer_prefix}linear_attn.conv1d.weight"),
                    (
                        Dtype::F32,
                        vec![
                            (2 * dims.linear_num_key_heads * dims.linear_key_head_dim
                                + dims.linear_num_value_heads * dims.linear_value_head_dim)
                                as usize,
                            dims.linear_conv_kernel_dim as usize,
                            1,
                        ],
                    ),
                );
                auxiliary_tensors.insert(
                    format!("{layer_prefix}linear_attn.in_proj_a.weight"),
                    (
                        Dtype::F32,
                        vec![
                            dims.linear_num_value_heads as usize,
                            dims.hidden_size as usize,
                        ],
                    ),
                );
                auxiliary_tensors.insert(
                    format!("{layer_prefix}linear_attn.in_proj_b.weight"),
                    (
                        Dtype::F32,
                        vec![
                            dims.linear_num_value_heads as usize,
                            dims.hidden_size as usize,
                        ],
                    ),
                );
                auxiliary_tensors.insert(
                    format!("{layer_prefix}linear_attn.norm.weight"),
                    (Dtype::F32, vec![dims.linear_key_head_dim as usize]),
                );
            }
        }
    }
    auxiliary_tensors
}

fn validate_tensor_declaration(
    tensor_name: &str,
    expected_dtype: Dtype,
    expected_shape: &[usize],
    tensor_declarations: &BTreeMap<&str, (Dtype, &[usize])>,
) -> Result<(), FoldedPackValidationError> {
    let Some(&(actual_dtype, actual_shape)) = tensor_declarations.get(tensor_name) else {
        return Err(FoldedPackValidationError::MissingPackedTensor {
            module_path: tensor_name.to_owned(),
            suffix: String::new(),
        });
    };
    if actual_dtype != expected_dtype {
        return Err(FoldedPackValidationError::TensorDtypeMismatch {
            tensor_name: tensor_name.to_owned(),
            expected_dtype: dtype_label(expected_dtype),
            actual_dtype: dtype_label(actual_dtype).to_owned(),
        });
    }
    if actual_shape != expected_shape {
        return Err(FoldedPackValidationError::TensorShapeMismatch {
            tensor_name: tensor_name.to_owned(),
            expected_shape: format!("{expected_shape:?}"),
            actual_shape: format!("{actual_shape:?}"),
        });
    }
    Ok(())
}

/// Human-readable dtype labels for contract failure messages.
fn dtype_label(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::U32 => "U32",
        Dtype::F16 => "F16",
        Dtype::F32 => "F32",
        Dtype::BF16 => "BF16",
        _ => "other",
    }
}

fn validate_manifest_module_agreement(
    config: &FoldedPackConfig,
    hadamard_contract: &HadamardContract,
) -> Result<(), FoldedPackValidationError> {
    let mut module_weight_names = BTreeSet::new();
    for module in config.packed_modules() {
        module_weight_names.insert(format!("{}.weight", module.tensor_stem()));
    }
    let mut manifest_names = hadamard_contract.rotated_tensor_names().clone();
    manifest_names.extend(hadamard_contract.inverse_tensor_names().iter().cloned());
    let modules_without_manifest_entries: Vec<&String> = module_weight_names
        .difference(&manifest_names)
        .take(3)
        .collect();
    let manifest_entries_without_modules: Vec<&String> = manifest_names
        .difference(&module_weight_names)
        .take(3)
        .collect();
    if !modules_without_manifest_entries.is_empty() || !manifest_entries_without_modules.is_empty()
    {
        return Err(FoldedPackValidationError::ContractModuleDisagreement {
            detail: format!(
                "modules missing from the manifest: {modules_without_manifest_entries:?}; manifest entries missing from the modules: {manifest_entries_without_modules:?}"
            ),
        });
    }
    Ok(())
}
