//! Affine quantization widths declared by the artifact, including router overrides.

use std::collections::BTreeMap;

use super::{K2HorizonMoVAConfigError, document::K2HorizonMoVAQuantizationDocument};

const LEGAL_AFFINE_BITS: [u32; 6] = [2, 3, 4, 5, 6, 8];
const LEGAL_AFFINE_GROUP_SIZES: [u32; 3] = [32, 64, 128];

/// One affine width/group pair used by a linear module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct K2HorizonMoVAAffineProfile {
    bits: u32,
    group_size: u32,
}

impl K2HorizonMoVAAffineProfile {
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.bits
    }

    #[must_use]
    pub const fn group_size(self) -> u32 {
        self.group_size
    }
}

/// Default affine profile plus per-module overrides from `quantization`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct K2HorizonMoVAQuantizationContract {
    default_profile: K2HorizonMoVAAffineProfile,
    module_overrides: BTreeMap<String, K2HorizonMoVAAffineProfile>,
}

impl K2HorizonMoVAQuantizationContract {
    pub(super) fn from_document(
        document: &K2HorizonMoVAQuantizationDocument,
    ) -> Result<Self, K2HorizonMoVAConfigError> {
        if document.mode != "affine" {
            return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
                description: "K2 Horizon MoVA serving requires affine quantization".to_owned(),
            });
        }
        let default_profile = parse_affine_profile(document.bits, document.group_size)?;
        let mut module_overrides = BTreeMap::new();
        for (module_path, override_value) in &document.module_overrides {
            let Some(override_object) = override_value.as_object() else {
                continue;
            };
            let Some(bits) = override_object
                .get("bits")
                .and_then(serde_json::Value::as_u64)
            else {
                continue;
            };
            let Some(group_size) = override_object
                .get("group_size")
                .and_then(serde_json::Value::as_u64)
            else {
                continue;
            };
            module_overrides.insert(
                module_path.clone(),
                parse_affine_profile(bits as u32, group_size as u32)?,
            );
        }
        Ok(Self {
            default_profile,
            module_overrides,
        })
    }

    #[must_use]
    pub const fn default_profile(&self) -> K2HorizonMoVAAffineProfile {
        self.default_profile
    }

    /// Returns the affine profile for one module path, falling back to the default.
    #[must_use]
    pub fn profile_for_module(&self, module_path: &str) -> K2HorizonMoVAAffineProfile {
        self.module_overrides
            .get(module_path)
            .copied()
            .unwrap_or(self.default_profile)
    }
}

fn parse_affine_profile(
    bits: u32,
    group_size: u32,
) -> Result<K2HorizonMoVAAffineProfile, K2HorizonMoVAConfigError> {
    if !LEGAL_AFFINE_BITS.contains(&bits) {
        return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
            description: format!("affine bits {bits} are not an MLX-supported width"),
        });
    }
    if !LEGAL_AFFINE_GROUP_SIZES.contains(&group_size) {
        return Err(K2HorizonMoVAConfigError::InvalidConfigValue {
            description: format!("affine group size {group_size} is not an MLX-supported group"),
        });
    }
    Ok(K2HorizonMoVAAffineProfile { bits, group_size })
}
