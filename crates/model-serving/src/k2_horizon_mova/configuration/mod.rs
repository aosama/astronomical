//! Typed K2 Horizon MoVA configuration parsed from family `config.json` knobs.

mod affine_profile;
mod config;
mod document;
mod error;
mod layer_schedule;

pub use affine_profile::{K2HorizonMoVAAffineProfile, K2HorizonMoVAQuantizationContract};
pub use config::{K2HorizonMoVAAttentionGateFunc, K2HorizonMoVAConfig};
pub use error::K2HorizonMoVAConfigError;
pub use layer_schedule::K2HorizonMoVALayerKind;
