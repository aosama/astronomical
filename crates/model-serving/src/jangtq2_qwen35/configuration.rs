//! Parses the per-module quantization override entries found in a jangtq2
//! bundle's `config.json` `quantization` map.
//!
//! Three entry shapes coexist in one map (measured on the
//! Ornith-1.5-122B-JANGH2 bundle: 144 codebook, 1252 affine, 125 boolean
//! entries):
//!
//! * `{"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"}` — routed
//!   expert projections stored as codebook-packed tensors.
//! * `{"bits": 8, "group_size": 64}` — standard MLX affine quantization.
//!   Older OptiQ documents repeat the document-level `"mode": "affine"`
//!   inside every entry; that restatement is accepted as redundant.
//! * `false` — the module is stored as native floating point (routers,
//!   shared-expert gates, norms); the MLX loader convention treats a missing
//!   or `false` entry as unquantized.
//!
//! Deserialization is dispatched by hand rather than through an untagged
//! enum: untagged matching erases per-variant errors and would let a
//! malformed entry fall through to a different shape (for example
//! `{"mode": "jangtq3", "bits": 2, "group_size": 64}` silently parsing as
//! affine). Every mis-shaped entry is rejected with a message naming the
//! offending field, so a future format revision is refused by old runtimes
//! instead of mis-decoded — the same reason the reference chose
//! `tq2_`-prefixed tensor names (old runtimes fail on unknown tensors).

use serde::Deserialize;
use serde::de::Error as _;

use super::codebook;

/// Discriminator string routing an entry to the JANGTQ v2 codebook format.
const CODEBOOK_MODE_WIRE_NAME: &str = "jangtq2";

/// Mode string older OptiQ documents repeat inside every affine entry. The
/// document-level `mode` is the authority; per-module affine entries may
/// restate it (historical artifacts do) or omit it, and both parse.
const AFFINE_MODE_WIRE_NAME: &str = "affine";

/// Wire name of the `none` rotation.
const NO_ROTATION_WIRE_NAME: &str = "none";

/// Wire name of the blockwise Hadamard-32 rotation.
const HADAMARD_32_WIRE_NAME: &str = "hadamard32";

/// Marker for entries whose `mode` field carried the exact codebook
/// discriminator. Constructing the codebook variant is only possible through
/// the dispatch below, which verified the mode string first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Jangtq2CodebookMode;

/// The activation-side rotation declared by a codebook entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Jangtq2CodebookRotation {
    /// Weights were quantized as stored; no rotation anywhere.
    NoRotation,
    /// Weights were quantized as `W · Rᵀ` with the blockwise normalized
    /// Walsh-Hadamard over 32-wide input blocks. Decode rotates activations.
    Hadamard32,
}

impl Jangtq2CodebookRotation {
    /// Resolves a rotation from its wire name, rejecting unknown values.
    pub fn from_wire_name(wire_name: &str) -> Result<Self, String> {
        match wire_name {
            NO_ROTATION_WIRE_NAME => Ok(Self::NoRotation),
            HADAMARD_32_WIRE_NAME => Ok(Self::Hadamard32),
            _ => Err(format!(
                "unknown jangtq2 rotation `{wire_name}`; only `{NO_ROTATION_WIRE_NAME}` and `{HADAMARD_32_WIRE_NAME}` are recognized"
            )),
        }
    }
}

/// Declared language RMSNorm convention for a jangtq2 bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Jangtq2RmsNormConvention {
    /// Apply the stored RMSNorm weight directly.
    DirectWeight,
    /// The stored weight is a deviation from one; apply `1 + weight`.
    LanguageMlxPlusOne,
}

impl Jangtq2RmsNormConvention {
    /// Resolves the bundle-level convention string, rejecting unknown values.
    pub fn from_wire_name(wire_name: &str) -> Result<Self, Jangtq2ConfigurationError> {
        match wire_name {
            "direct" => Ok(Self::DirectWeight),
            "qwen3_5_language_mlx_plus_one" => Ok(Self::LanguageMlxPlusOne),
            _ => Err(Jangtq2ConfigurationError::UnsupportedRmsNormConvention {
                actual_convention: wire_name.to_owned(),
            }),
        }
    }
}

/// One per-module quantization override entry, in any of its three shapes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Jangtq2WeightOverrideEntry {
    /// A JANGTQ v2 codebook entry for a routed-expert projection.
    Codebook {
        /// The `"jangtq2"` discriminator (see [`Jangtq2CodebookMode`]).
        mode: Jangtq2CodebookMode,
        /// Codebook bit width (2, 3, or 4 — see [`codebook`]).
        bits: u32,
        /// The activation-side rotation.
        rotation: Jangtq2CodebookRotation,
    },
    /// A standard MLX affine override.
    Affine {
        /// Affine bit width.
        bits: u32,
        /// Affine group size.
        group_size: u32,
    },
    /// An MLX loader boolean flag. `false` marks a module stored as native
    /// floating point; `true` means "quantize with the document defaults".
    LoaderFlag(bool),
}

impl<'de> Deserialize<'de> for Jangtq2WeightOverrideEntry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire_value = serde_json::Value::deserialize(deserializer)?;
        if let Some(loader_flag) = wire_value.as_bool() {
            return Ok(Self::LoaderFlag(loader_flag));
        }
        let Some(wire_object) = wire_value.as_object() else {
            return Err(D::Error::custom(
                "a quantization override entry must be an object or a boolean flag",
            ));
        };
        let mode_wire_name = wire_object.get("mode").and_then(serde_json::Value::as_str);
        let rotation_wire_name = wire_object
            .get("rotation")
            .and_then(serde_json::Value::as_str);
        let bits_wire_value = wire_object.get("bits").and_then(serde_json::Value::as_u64);
        let group_size_wire_value = wire_object
            .get("group_size")
            .and_then(serde_json::Value::as_u64);

        let dispatch_error = |message: String| Err(D::Error::custom(message));
        match mode_wire_name {
            Some(CODEBOOK_MODE_WIRE_NAME) => {
                if group_size_wire_value.is_some() {
                    return dispatch_error(
                        "a jangtq2 codebook entry carries no group_size; remove it".to_owned(),
                    );
                }
                let Some(bits_wire_value) = bits_wire_value else {
                    return dispatch_error("a jangtq2 codebook entry requires bits".to_owned());
                };
                let Some(rotation_wire_name) = rotation_wire_name else {
                    return dispatch_error(
                        "a jangtq2 codebook entry requires its rotation to be declared".to_owned(),
                    );
                };
                let rotation = Jangtq2CodebookRotation::from_wire_name(rotation_wire_name)
                    .map_err(D::Error::custom)?;
                let bits = u32::try_from(bits_wire_value).map_err(|_| {
                    D::Error::custom("a jangtq2 codebook entry requires a small bit width")
                })?;
                Ok(Self::Codebook {
                    mode: Jangtq2CodebookMode,
                    bits,
                    rotation,
                })
            }
            // An affine entry restating its mode is accepted: the repeated
            // string is redundant with the document-level mode, and older
            // OptiQ artifacts carry it in every entry. This arm must precede
            // the unknown-mode catch-all below it.
            Some(AFFINE_MODE_WIRE_NAME) | None => {
                if rotation_wire_name.is_some() {
                    return dispatch_error(
                        "a rotation is only valid inside a jangtq2 codebook entry".to_owned(),
                    );
                }
                let (Some(bits_wire_value), Some(group_size_wire_value)) =
                    (bits_wire_value, group_size_wire_value)
                else {
                    return dispatch_error(
                        "an affine quantization entry requires bits and group_size".to_owned(),
                    );
                };
                let bits = u32::try_from(bits_wire_value).map_err(|_| {
                    D::Error::custom("an affine quantization entry requires a small bit width")
                })?;
                let group_size = u32::try_from(group_size_wire_value).map_err(|_| {
                    D::Error::custom("an affine quantization entry requires a small group size")
                })?;
                Ok(Self::Affine { bits, group_size })
            }
            Some(unknown_mode) => Err(D::Error::custom(format!(
                "unknown quantization mode `{unknown_mode}`; only `{AFFINE_MODE_WIRE_NAME}` and `{CODEBOOK_MODE_WIRE_NAME}` are recognized"
            ))),
        }
    }
}

/// Errors rejected while resolving a jangtq2 module's storage contract.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum Jangtq2ConfigurationError {
    /// A bundle declared an RMSNorm storage convention the loader does not know.
    #[error("jangtq2 RMSNorm convention `{actual_convention}` is not supported")]
    UnsupportedRmsNormConvention {
        /// The rejected convention string.
        actual_convention: String,
    },
    /// A codebook entry declared a bit width outside the codebook family.
    #[error(
        "jangtq2 codebook bit width {actual_bit_width} is not supported (expected one of {supported_bit_widths:?})"
    )]
    UnsupportedCodebookBitWidth {
        /// The rejected bit width.
        actual_bit_width: u32,
        /// The widths the codebook family defines.
        supported_bit_widths: &'static [u32],
    },
    /// A packed layout was requested for an input width that cannot be
    /// packed into whole words or whole companion groups. Both the codebook
    /// bitstream (32-code groups) and the affine layout (whole groups along
    /// the input width) need the width to divide evenly; a width that does
    /// not cannot be described by any honest tensor shape.
    #[error(
        "jangtq2 packed storage requires an input width divisible by {required_alignment_multiple}, got {actual_input_width}"
    )]
    InputWidthNotPackingAligned {
        /// The rejected input width.
        actual_input_width: u32,
        /// The alignment the layout needs (the 32-code group width, or the
        /// affine group size).
        required_alignment_multiple: u32,
    },
    /// The tensor contract was consulted for a raw `true` loader flag whose
    /// document-default resolution never happened. This is a caller bug;
    /// failing closed is the safe outcome because no shape can be derived
    /// from an unresolved flag.
    #[error(
        "jangtq2 tensor contract reached before a default-quantization flag was resolved to a concrete entry"
    )]
    UnresolvedDefaultQuantizationFlag,
}

impl Jangtq2WeightOverrideEntry {
    /// Validates the entry against the format's own rules.
    ///
    /// Affine bit widths and group sizes are deliberately not checked here:
    /// those belong to the MLX affine grid, which the caller (the OptiQ
    /// quantization config) already validates. This method owns exactly the
    /// jangtq2-specific constraints.
    pub fn validate(&self) -> Result<(), Jangtq2ConfigurationError> {
        match self {
            Self::Codebook { bits, .. } => {
                if codebook::is_supported_bit_width(*bits) {
                    Ok(())
                } else {
                    Err(Jangtq2ConfigurationError::UnsupportedCodebookBitWidth {
                        actual_bit_width: *bits,
                        supported_bit_widths: codebook::SUPPORTED_CODEBOOK_BIT_WIDTHS,
                    })
                }
            }
            Self::Affine { .. } | Self::LoaderFlag(_) => Ok(()),
        }
    }

    /// Returns whether the entry marks a module stored as native floating
    /// point (`false` flags only). Such modules resolve to unquantized
    /// profiles regardless of the document's default bits.
    #[must_use]
    pub const fn is_native_floating_point(&self) -> bool {
        matches!(self, Self::LoaderFlag(false))
    }
}

#[cfg(test)]
mod tests {
    use super::Jangtq2CodebookMode;
    use super::Jangtq2CodebookRotation;
    use super::Jangtq2RmsNormConvention;
    use super::Jangtq2WeightOverrideEntry;

    const TWO_BIT_CODEBOOK_ENTRY_FROM_THE_JANGH2_BUNDLE: &str =
        r#"{"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"}"#;
    const UNROTATED_CODEBOOK_ENTRY: &str = r#"{"mode": "jangtq2", "bits": 3, "rotation": "none"}"#;
    const THREE_BIT_CODEBOOK_ENTRY: &str =
        r#"{"mode": "jangtq2", "bits": 3, "rotation": "hadamard32"}"#;
    const CODEBOOK_ENTRY_WITHOUT_ROTATION: &str = r#"{"mode": "jangtq2", "bits": 2}"#;
    const CODEBOOK_ENTRY_WITHOUT_BITS: &str = r#"{"mode": "jangtq2", "rotation": "hadamard32"}"#;
    const CODEBOOK_ENTRY_CARRYING_GROUP_SIZE: &str =
        r#"{"mode": "jangtq2", "bits": 2, "rotation": "hadamard32", "group_size": 64}"#;
    const AFFINE_EIGHT_BIT_GROUP_SIXTY_FOUR_ENTRY: &str = r#"{"bits": 8, "group_size": 64}"#;
    const AFFINE_ENTRY_REPEATING_THE_AFFINE_MODE: &str =
        r#"{"bits": 8, "group_size": 64, "mode": "affine"}"#;
    const AFFINE_ENTRY_WITHOUT_GROUP_SIZE: &str = r#"{"bits": 8}"#;
    const AFFINE_ENTRY_CARRYING_ROTATION: &str =
        r#"{"bits": 8, "group_size": 64, "rotation": "hadamard32"}"#;
    const NATIVE_FLOATING_POINT_FLAG: &str = r#"false"#;
    const DEFAULT_QUANTIZATION_FLAG: &str = r#"true"#;
    const ENTRY_WITH_UNKNOWN_MODE: &str =
        r#"{"mode": "jangtq3", "bits": 2, "rotation": "hadamard32"}"#;
    // The exact fall-through shape an untagged enum would have silently
    // accepted as affine; the manual dispatch must reject it.
    const UNKNOWN_MODE_ENTRY_SHAPED_LIKE_AFFINE: &str =
        r#"{"mode": "jangtq3", "bits": 2, "group_size": 64}"#;
    const ENTRY_WITH_UNDECLARED_MODE: &str = r#"{"bits": 2, "rotation": "hadamard32"}"#;
    const ENTRY_WITH_UNKNOWN_ROTATION: &str =
        r#"{"mode": "jangtq2", "bits": 2, "rotation": "hadamard64"}"#;
    const ENTRY_THAT_IS_NEITHER_OBJECT_NOR_FLAG: &str = r#""jangtq2""#;

    fn parse_entry(entry_json: &str) -> Result<Jangtq2WeightOverrideEntry, serde_json::Error> {
        serde_json::from_str(entry_json)
    }

    #[test]
    fn should_resolve_the_jangtq2_language_mlx_plus_one_norm_convention() {
        assert_eq!(
            Jangtq2RmsNormConvention::from_wire_name("qwen3_5_language_mlx_plus_one"),
            Ok(Jangtq2RmsNormConvention::LanguageMlxPlusOne)
        );
        assert_eq!(
            Jangtq2RmsNormConvention::from_wire_name("direct"),
            Ok(Jangtq2RmsNormConvention::DirectWeight)
        );
    }

    #[test]
    fn should_reject_an_unknown_jangtq2_language_norm_convention() {
        assert!(matches!(
            Jangtq2RmsNormConvention::from_wire_name("unknown"),
            Err(super::Jangtq2ConfigurationError::UnsupportedRmsNormConvention {
                actual_convention
            }) if actual_convention == "unknown"
        ));
    }

    #[test]
    fn should_parse_codebook_entry_from_the_jangh2_bundle() {
        let parsed_entry = parse_entry(TWO_BIT_CODEBOOK_ENTRY_FROM_THE_JANGH2_BUNDLE)
            .expect("the exact layer-entry shape from the bundle must parse");
        assert_eq!(
            parsed_entry,
            Jangtq2WeightOverrideEntry::Codebook {
                mode: Jangtq2CodebookMode,
                bits: 2,
                rotation: Jangtq2CodebookRotation::Hadamard32,
            }
        );
        parsed_entry
            .validate()
            .expect("2-bit codebook is supported");
    }

    #[test]
    fn should_parse_three_bit_codebook_entry() {
        let parsed_entry = parse_entry(THREE_BIT_CODEBOOK_ENTRY).expect("3-bit entries must parse");
        assert_eq!(
            parsed_entry,
            Jangtq2WeightOverrideEntry::Codebook {
                mode: Jangtq2CodebookMode,
                bits: 3,
                rotation: Jangtq2CodebookRotation::Hadamard32,
            }
        );
    }

    #[test]
    fn should_parse_unrotated_codebook_entry() {
        let parsed_entry =
            parse_entry(UNROTATED_CODEBOOK_ENTRY).expect("rotation none entries must parse");
        assert_eq!(
            parsed_entry,
            Jangtq2WeightOverrideEntry::Codebook {
                mode: Jangtq2CodebookMode,
                bits: 3,
                rotation: Jangtq2CodebookRotation::NoRotation,
            }
        );
    }

    #[test]
    fn should_parse_affine_entry() {
        let parsed_entry = parse_entry(AFFINE_EIGHT_BIT_GROUP_SIXTY_FOUR_ENTRY)
            .expect("affine entries must keep parsing");
        assert_eq!(
            parsed_entry,
            Jangtq2WeightOverrideEntry::Affine {
                bits: 8,
                group_size: 64,
            }
        );
    }

    #[test]
    fn should_parse_affine_entry_that_repeats_the_affine_mode() {
        // Older OptiQ artifacts restate the document-level mode inside every
        // affine entry; the extra string is redundant, never a conflict.
        let parsed_entry = parse_entry(AFFINE_ENTRY_REPEATING_THE_AFFINE_MODE)
            .expect("affine entries restating their mode must keep parsing");
        assert_eq!(
            parsed_entry,
            Jangtq2WeightOverrideEntry::Affine {
                bits: 8,
                group_size: 64,
            }
        );
    }

    #[test]
    fn should_parse_native_floating_point_flag() {
        let parsed_entry = parse_entry(NATIVE_FLOATING_POINT_FLAG).expect("false flags must parse");
        assert_eq!(parsed_entry, Jangtq2WeightOverrideEntry::LoaderFlag(false));
        assert!(parsed_entry.is_native_floating_point());
    }

    #[test]
    fn should_parse_default_quantization_flag() {
        let parsed_entry = parse_entry(DEFAULT_QUANTIZATION_FLAG).expect("true flags must parse");
        assert_eq!(parsed_entry, Jangtq2WeightOverrideEntry::LoaderFlag(true));
        assert!(!parsed_entry.is_native_floating_point());
    }

    #[test]
    fn should_refuse_unknown_mode_string() {
        assert!(parse_entry(ENTRY_WITH_UNKNOWN_MODE).is_err());
    }

    #[test]
    fn should_refuse_unknown_mode_even_when_the_entry_is_shaped_like_affine() {
        // This is the fall-through hole an untagged enum would open: an
        // affine-shaped entry whose mode is a future format revision must be
        // rejected, never silently accepted as affine.
        assert!(parse_entry(UNKNOWN_MODE_ENTRY_SHAPED_LIKE_AFFINE).is_err());
    }

    #[test]
    fn should_refuse_entry_with_undeclared_mode() {
        assert!(parse_entry(ENTRY_WITH_UNDECLARED_MODE).is_err());
    }

    #[test]
    fn should_refuse_unknown_rotation_string() {
        assert!(parse_entry(ENTRY_WITH_UNKNOWN_ROTATION).is_err());
    }

    #[test]
    fn should_refuse_codebook_entry_missing_rotation() {
        // Rotation is a required field of the codebook shape: v2 bundles
        // always declare it, and guessing would silently mis-decode.
        assert!(parse_entry(CODEBOOK_ENTRY_WITHOUT_ROTATION).is_err());
    }

    #[test]
    fn should_refuse_codebook_entry_missing_bits() {
        assert!(parse_entry(CODEBOOK_ENTRY_WITHOUT_BITS).is_err());
    }

    #[test]
    fn should_refuse_codebook_entry_carrying_group_size() {
        // Codebook entries have no group concept; an entry mixing both
        // shapes is malformed, not half of each.
        assert!(parse_entry(CODEBOOK_ENTRY_CARRYING_GROUP_SIZE).is_err());
    }

    #[test]
    fn should_refuse_affine_entry_missing_group_size() {
        assert!(parse_entry(AFFINE_ENTRY_WITHOUT_GROUP_SIZE).is_err());
    }

    #[test]
    fn should_refuse_affine_entry_carrying_rotation() {
        assert!(parse_entry(AFFINE_ENTRY_CARRYING_ROTATION).is_err());
    }

    #[test]
    fn should_refuse_entry_that_is_neither_object_nor_flag() {
        assert!(parse_entry(ENTRY_THAT_IS_NEITHER_OBJECT_NOR_FLAG).is_err());
    }

    #[test]
    fn should_reject_unsupported_codebook_bit_width() {
        let unsupported_entry = Jangtq2WeightOverrideEntry::Codebook {
            mode: Jangtq2CodebookMode,
            bits: 5,
            rotation: Jangtq2CodebookRotation::Hadamard32,
        };
        assert_eq!(
            unsupported_entry.validate(),
            Err(
                super::Jangtq2ConfigurationError::UnsupportedCodebookBitWidth {
                    actual_bit_width: 5,
                    supported_bit_widths: super::codebook::SUPPORTED_CODEBOOK_BIT_WIDTHS,
                }
            )
        );
    }
}
