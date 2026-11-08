//! Everything specific to the 4-bit MLX build of FLUX.2 Klein 4B
//! (`mlx-community/flux2-klein-4b-4bit`, issue #1904): its identity, how it
//! names its tensors, the configuration it does not ship, and how its files are
//! validated. The rest of the engine stays in Diffusers terms: the validator
//! here hands back the same validated artifact the BF16 build produces, with
//! inventories keyed by Diffusers names.

mod artifact;
mod provenance;
mod shard_index;
mod tensor_names;
mod tensor_profile;
mod upstream_configs;

pub(in crate::flux2_klein) use artifact::validate_mlx_4bit_artifact;
pub(crate) use provenance::identifies_mlx_4bit_build;
pub use provenance::{
    FLUX2_KLEIN_MLX_4BIT_MODEL_ID, FLUX2_KLEIN_MLX_4BIT_PROVIDER_MODEL_ID,
    FLUX2_KLEIN_MLX_4BIT_REVISION,
};
pub(crate) use tensor_names::mlx_4bit_tensor_name;
