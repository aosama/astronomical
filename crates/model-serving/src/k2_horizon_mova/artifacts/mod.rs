//! Stacked MLX affine artifact validation for the K2 Horizon MoVA family.

mod dialect;
mod error;
mod expected_tensors;
mod shard_index;
mod validator;

pub use dialect::K2HorizonMoVAWeightDialect;
pub use error::K2HorizonMoVAArtifactValidationError;
pub use expected_tensors::expected_stacked_affine_tensor_names;
pub use shard_index::K2HorizonMoVAShardIndex;
pub use validator::{K2HorizonMoVAArtifactValidator, ValidatedK2HorizonMoVAArtifact};
