//! Stacked MLX affine artifact validation for the K2 Horizon MoVA family.

pub(crate) mod dialect;
mod error;
mod expected_tensors;
mod shard_index;
pub(crate) mod validator;

pub use error::K2HorizonMoVAArtifactValidationError;
pub use expected_tensors::expected_stacked_affine_tensor_names;
pub use shard_index::K2HorizonMoVAShardIndex;
pub use validator::ValidatedK2HorizonMoVAArtifact;

#[cfg(feature = "direct-mlx")]
pub use validator::K2HorizonMoVAArtifactValidator;
