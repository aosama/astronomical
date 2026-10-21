//! The on-disk tensor inventory contract for jangtq2 modules.
//!
//! A jangtq2 bundle stores every module in one of three physical layouts,
//! and this contract is what the artifact validator checks a shard index
//! against — the family owns its inventory shapes end to end:
//!
//! * **Stacked codebook** (routed-expert projections): one
//!   `{module}.tq2_packed` `uint32` tensor of shape
//!   `(experts, outputs, inputs · bits / 32)` plus one
//!   `{module}.tq2_scales` `float16` tensor of shape `(experts, outputs)`.
//!   The pair covers *all* experts of the layer — experts are stacked along
//!   the leading axis, never one tensor per expert.
//! * **Affine triple**: `{module}.weight` `uint32`, plus `.scales` and
//!   `.biases` companions — the standard MLX layout.
//! * **Native weight**: bare `{module}.weight` in `bfloat16`.
//!
//! Shapes are derived from the module geometry (hidden size, MoE
//! intermediate size, expert count) resolved from the text config by the
//! caller, so this module stays free of model-architecture knowledge.

use super::codebook;
use super::configuration::{
    Jangtq2CodebookRotation, Jangtq2ConfigurationError, Jangtq2WeightOverrideEntry,
};

/// The tensor suffix carrying an expert-stacked codebook's packed codes.
pub const CODEBOOK_PACKED_TENSOR_SUFFIX: &str = "tq2_packed";

/// The tensor suffix carrying an expert-stacked codebook's row scales.
pub const CODEBOOK_SCALES_TENSOR_SUFFIX: &str = "tq2_scales";

/// The physical element type a contract tensor is stored in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractTensorElementKind {
    /// Packed quantization codes (and affine packed weights): `uint32`.
    PackedUnsignedInteger32,
    /// The codebook row scales: `float16`.
    CodebookScaleFloat16,
    /// Affine per-group scales and biases: `bfloat16`.
    AffineCompanionBfloat16,
    /// Unquantized module weights: `bfloat16`.
    NativeWeightBfloat16,
}

/// One expected tensor: its on-disk name, element kind, and shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractTensorExpectation {
    /// The tensor name as it appears in the safetensors index.
    pub tensor_name: String,
    /// The stored element kind.
    pub element_kind: ContractTensorElementKind,
    /// The stored shape.
    pub shape: Vec<usize>,
}

/// The geometry of one quantizable module, resolved from the model config.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Jangtq2ModuleGeometry {
    /// The module's input width (`K`).
    pub input_width: u32,
    /// The module's output width (`N`).
    pub output_width: u32,
    /// The number of experts stacked along the leading axis. Non-expert
    /// modules pass `1` and are never given stacked codebook tensors.
    pub expert_count: u32,
    /// The affine group size used by affine modules of this bundle; the
    /// affine contract requires the input width to divide into whole
    /// groups of exactly this size.
    pub affine_group_size: u32,
}

/// The full tensor inventory one module must provide.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Jangtq2ModuleTensorInventory {
    /// Every tensor the module must carry on disk.
    pub expected_tensors: Vec<ContractTensorExpectation>,
    /// The bit width when the module is packed (codebook or affine);
    /// `None` for native floating point.
    pub packed_bit_width: Option<u32>,
    /// The activation-side rotation the decode path must apply.
    pub activation_rotation: Jangtq2CodebookRotation,
}

/// Builds the tensor inventory for one module from its override entry and
/// geometry.
///
/// The entry must already be validated
/// ([`Jangtq2WeightOverrideEntry::validate`]); this function owns the
/// geometry-dependent rules. Every packed layout requires the input width
/// to divide into whole 32-code groups — which is also exactly the
/// Hadamard-32 block width, so a packing-aligned width is rotation-blockable
/// by construction; a future rotation wider than the code group would need
/// its own check added here.
pub fn tensor_inventory_for_module(
    module_name: &str,
    override_entry: &Jangtq2WeightOverrideEntry,
    geometry: &Jangtq2ModuleGeometry,
) -> Result<Jangtq2ModuleTensorInventory, Jangtq2ConfigurationError> {
    let packing_alignment_multiple = codebook::codes_per_group();
    if geometry.input_width % packing_alignment_multiple != 0 {
        return Err(Jangtq2ConfigurationError::InputWidthNotPackingAligned {
            actual_input_width: geometry.input_width,
            required_alignment_multiple: packing_alignment_multiple,
        });
    }
    match override_entry {
        Jangtq2WeightOverrideEntry::Codebook { bits, rotation, .. } => {
            let codebook_bit_width = *bits;
            let packed_last_dimension = codebook::packed_word_count_for_input_width(
                geometry.input_width,
                codebook_bit_width,
            ) as usize;
            Ok(Jangtq2ModuleTensorInventory {
                expected_tensors: vec![
                    ContractTensorExpectation {
                        tensor_name: format!("{module_name}.{CODEBOOK_PACKED_TENSOR_SUFFIX}"),
                        element_kind: ContractTensorElementKind::PackedUnsignedInteger32,
                        shape: vec![
                            geometry.expert_count as usize,
                            geometry.output_width as usize,
                            packed_last_dimension,
                        ],
                    },
                    ContractTensorExpectation {
                        tensor_name: format!("{module_name}.{CODEBOOK_SCALES_TENSOR_SUFFIX}"),
                        element_kind: ContractTensorElementKind::CodebookScaleFloat16,
                        shape: vec![
                            geometry.expert_count as usize,
                            geometry.output_width as usize,
                        ],
                    },
                ],
                packed_bit_width: Some(codebook_bit_width),
                activation_rotation: *rotation,
            })
        }
        Jangtq2WeightOverrideEntry::Affine { bits, group_size } => {
            let affine_bit_width = *bits;
            let affine_group_size = *group_size;
            // Affine companions run along whole groups of the input width:
            // a width that does not divide evenly would leave the contract
            // describing fewer companion columns than the tensor carries.
            if geometry.input_width % affine_group_size != 0 {
                return Err(Jangtq2ConfigurationError::InputWidthNotPackingAligned {
                    actual_input_width: geometry.input_width,
                    required_alignment_multiple: affine_group_size,
                });
            }
            let packed_last_dimension =
                codebook::packed_word_count_for_input_width(geometry.input_width, affine_bit_width)
                    as usize;
            let companion_last_dimension = (geometry.input_width / affine_group_size) as usize;
            Ok(Jangtq2ModuleTensorInventory {
                expected_tensors: vec![
                    ContractTensorExpectation {
                        tensor_name: format!("{module_name}.weight"),
                        element_kind: ContractTensorElementKind::PackedUnsignedInteger32,
                        shape: vec![geometry.output_width as usize, packed_last_dimension],
                    },
                    ContractTensorExpectation {
                        tensor_name: format!("{module_name}.scales"),
                        element_kind: ContractTensorElementKind::AffineCompanionBfloat16,
                        shape: vec![geometry.output_width as usize, companion_last_dimension],
                    },
                    ContractTensorExpectation {
                        tensor_name: format!("{module_name}.biases"),
                        element_kind: ContractTensorElementKind::AffineCompanionBfloat16,
                        shape: vec![geometry.output_width as usize, companion_last_dimension],
                    },
                ],
                packed_bit_width: Some(affine_bit_width),
                activation_rotation: Jangtq2CodebookRotation::NoRotation,
            })
        }
        Jangtq2WeightOverrideEntry::LoaderFlag(false) => Ok(Jangtq2ModuleTensorInventory {
            expected_tensors: vec![ContractTensorExpectation {
                tensor_name: format!("{module_name}.weight"),
                element_kind: ContractTensorElementKind::NativeWeightBfloat16,
                shape: vec![
                    geometry.output_width as usize,
                    geometry.input_width as usize,
                ],
            }],
            packed_bit_width: None,
            activation_rotation: Jangtq2CodebookRotation::NoRotation,
        }),
        // A `true` flag defers to the document defaults, which the caller
        // resolves before consulting this contract; reaching here with the
        // raw flag is a caller bug, and failing closed is the safe outcome.
        Jangtq2WeightOverrideEntry::LoaderFlag(true) => {
            Err(Jangtq2ConfigurationError::UnresolvedDefaultQuantizationFlag)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jangtq2_qwen35::configuration::Jangtq2CodebookMode;
    use crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation;
    use crate::jangtq2_qwen35::configuration::Jangtq2ConfigurationError;
    use crate::jangtq2_qwen35::configuration::Jangtq2WeightOverrideEntry;

    const JANGH2_GATE_PROJECTION_MODULE_NAME: &str =
        "language_model.model.layers.5.mlp.switch_mlp.gate_proj";

    fn jangh2_gate_geometry() -> Jangtq2ModuleGeometry {
        // Measured from the bundle: hidden 3072 in, MoE intermediate 1024
        // out, 256 stacked experts.
        Jangtq2ModuleGeometry {
            input_width: 3072,
            output_width: 1024,
            expert_count: 256,
            affine_group_size: 64,
        }
    }

    #[test]
    fn should_expect_stacked_codebook_tensors_for_a_two_bit_projection() {
        let codebook_entry = Jangtq2WeightOverrideEntry::Codebook {
            mode: Jangtq2CodebookMode,
            bits: 2,
            rotation: Jangtq2CodebookRotation::Hadamard32,
        };
        let inventory = tensor_inventory_for_module(
            JANGH2_GATE_PROJECTION_MODULE_NAME,
            &codebook_entry,
            &jangh2_gate_geometry(),
        )
        .expect("the JANGH2 gate projection geometry must satisfy the contract");
        assert_eq!(inventory.packed_bit_width, Some(2));
        assert_eq!(
            inventory.activation_rotation,
            Jangtq2CodebookRotation::Hadamard32
        );
        assert_eq!(
            inventory.expected_tensors,
            vec![
                ContractTensorExpectation {
                    tensor_name:
                        "language_model.model.layers.5.mlp.switch_mlp.gate_proj.tq2_packed"
                            .to_owned(),
                    element_kind: ContractTensorElementKind::PackedUnsignedInteger32,
                    // 256 experts stacked, 1024 outputs, 3072 inputs at 2
                    // bits pack into 192 uint32 words per row.
                    shape: vec![256, 1024, 192],
                },
                ContractTensorExpectation {
                    tensor_name:
                        "language_model.model.layers.5.mlp.switch_mlp.gate_proj.tq2_scales"
                            .to_owned(),
                    element_kind: ContractTensorElementKind::CodebookScaleFloat16,
                    shape: vec![256, 1024],
                },
            ]
        );
    }

    #[test]
    fn should_expect_codebook_tensors_for_an_unrotated_projection() {
        let unrotated_entry = Jangtq2WeightOverrideEntry::Codebook {
            mode: Jangtq2CodebookMode,
            bits: 3,
            rotation: Jangtq2CodebookRotation::NoRotation,
        };
        let inventory = tensor_inventory_for_module(
            "language_model.model.layers.40.mlp.switch_mlp.down_proj",
            &unrotated_entry,
            &Jangtq2ModuleGeometry {
                // down_proj: MoE intermediate 1024 in, hidden 3072 out.
                input_width: 1024,
                output_width: 3072,
                expert_count: 256,
                affine_group_size: 64,
            },
        )
        .expect("an unrotated codebook entry maps to the same tensor pair");
        assert_eq!(inventory.packed_bit_width, Some(3));
        assert_eq!(
            inventory.activation_rotation,
            Jangtq2CodebookRotation::NoRotation
        );
        // 1024 inputs at 3 bits pack into 96 uint32 words per row.
        assert_eq!(inventory.expected_tensors[0].shape, vec![256, 3072, 96]);
    }

    #[test]
    fn should_expect_affine_triple_for_affine_entry() {
        let affine_entry = Jangtq2WeightOverrideEntry::Affine {
            bits: 8,
            group_size: 64,
        };
        let inventory = tensor_inventory_for_module(
            "language_model.model.layers.0.linear_attn.out_proj",
            &affine_entry,
            &Jangtq2ModuleGeometry {
                input_width: 2048,
                output_width: 3072,
                expert_count: 1,
                affine_group_size: 64,
            },
        )
        .expect("affine entries map to the standard triple");
        assert_eq!(inventory.packed_bit_width, Some(8));
        assert_eq!(
            inventory.activation_rotation,
            Jangtq2CodebookRotation::NoRotation
        );
        assert_eq!(
            inventory.expected_tensors,
            vec![
                ContractTensorExpectation {
                    tensor_name: "language_model.model.layers.0.linear_attn.out_proj.weight"
                        .to_owned(),
                    element_kind: ContractTensorElementKind::PackedUnsignedInteger32,
                    shape: vec![3072, 512],
                },
                ContractTensorExpectation {
                    tensor_name: "language_model.model.layers.0.linear_attn.out_proj.scales"
                        .to_owned(),
                    element_kind: ContractTensorElementKind::AffineCompanionBfloat16,
                    shape: vec![3072, 32],
                },
                ContractTensorExpectation {
                    tensor_name: "language_model.model.layers.0.linear_attn.out_proj.biases"
                        .to_owned(),
                    element_kind: ContractTensorElementKind::AffineCompanionBfloat16,
                    shape: vec![3072, 32],
                },
            ]
        );
    }

    #[test]
    fn should_expect_native_weight_for_native_floating_point_flag() {
        let native_entry = Jangtq2WeightOverrideEntry::LoaderFlag(false);
        let inventory = tensor_inventory_for_module(
            "language_model.model.layers.0.mlp.gate",
            &native_entry,
            &Jangtq2ModuleGeometry {
                input_width: 3072,
                output_width: 256,
                expert_count: 1,
                affine_group_size: 64,
            },
        )
        .expect("false flags map to a bare native weight");
        assert_eq!(inventory.packed_bit_width, None);
        assert_eq!(
            inventory.expected_tensors,
            vec![ContractTensorExpectation {
                tensor_name: "language_model.model.layers.0.mlp.gate.weight".to_owned(),
                element_kind: ContractTensorElementKind::NativeWeightBfloat16,
                shape: vec![256, 3072],
            }]
        );
    }

    #[test]
    fn should_reject_input_width_that_is_not_packing_aligned() {
        // 3070 divides by neither the 32-code group nor the affine group:
        // every packed layout is unavailable, rotated or not.
        let codebook_entry = Jangtq2WeightOverrideEntry::Codebook {
            mode: Jangtq2CodebookMode,
            bits: 2,
            rotation: Jangtq2CodebookRotation::Hadamard32,
        };
        let unaligned_geometry = Jangtq2ModuleGeometry {
            input_width: 3070,
            output_width: 1024,
            expert_count: 256,
            affine_group_size: 64,
        };
        assert_eq!(
            tensor_inventory_for_module("some.module", &codebook_entry, &unaligned_geometry),
            Err(Jangtq2ConfigurationError::InputWidthNotPackingAligned {
                actual_input_width: 3070,
                required_alignment_multiple: 32,
            })
        );
        // The same width must fall at the affine gate too, naming the
        // packing alignment rather than a rotation it never declared.
        let affine_entry = Jangtq2WeightOverrideEntry::Affine {
            bits: 8,
            group_size: 64,
        };
        assert_eq!(
            tensor_inventory_for_module("some.module", &affine_entry, &unaligned_geometry),
            Err(Jangtq2ConfigurationError::InputWidthNotPackingAligned {
                actual_input_width: 3070,
                required_alignment_multiple: 32,
            })
        );
    }

    #[test]
    fn should_reject_affine_input_width_that_leaves_a_partial_companion_group() {
        // 2080 divides by 32 but not by the affine group 64: the companion
        // columns could not be described honestly for the trailing group.
        let affine_entry = Jangtq2WeightOverrideEntry::Affine {
            bits: 8,
            group_size: 64,
        };
        let partial_group_geometry = Jangtq2ModuleGeometry {
            input_width: 2080,
            output_width: 1024,
            expert_count: 1,
            affine_group_size: 64,
        };
        assert_eq!(
            tensor_inventory_for_module("some.module", &affine_entry, &partial_group_geometry),
            Err(Jangtq2ConfigurationError::InputWidthNotPackingAligned {
                actual_input_width: 2080,
                required_alignment_multiple: 64,
            })
        );
    }

    #[test]
    fn should_fail_closed_when_a_default_quantization_flag_is_unresolved() {
        let unresolved_entry = Jangtq2WeightOverrideEntry::LoaderFlag(true);
        assert_eq!(
            tensor_inventory_for_module("some.module", &unresolved_entry, &jangh2_gate_geometry()),
            Err(Jangtq2ConfigurationError::UnresolvedDefaultQuantizationFlag)
        );
    }
}
