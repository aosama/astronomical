use astronomical_model_serving::{
    ModelWeightStorage, OptiQMetadata, OptiQMetadataError, Qwen3_5Config, Qwen3_5ConfigError,
    qwen3_5_language_tensor_profiles,
};
use serde_json::{Value, json};

use crate::common::qwen3_5_moe::{
    frozen_optiq_metadata_bytes, frozen_ornith_1_0_config_bytes,
    frozen_ornith_1_0_optiq_config_bytes,
};

mod compatibility_fallbacks;
mod model_contract;
mod optiq_metadata;
mod quantization;
pub(crate) mod support;

use support::minimal_valid_config_json;
