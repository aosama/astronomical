//! Identity of the reviewed 4-bit MLX build.

/// The build's model identifier: its repository name, as discovery and the
/// catalog record it. Distinct from the BF16 build's so both can be installed.
pub const FLUX2_KLEIN_MLX_4BIT_MODEL_ID: &str = "flux2-klein-4b-4bit";
/// The provider-prefixed identity the build is published under.
pub const FLUX2_KLEIN_MLX_4BIT_PROVIDER_MODEL_ID: &str = "mlx-community/flux2-klein-4b-4bit";
/// The snapshot reviewed in issue #1904. Content checks, not this SHA, validate
/// a package; the constant lets callers build matching provenance once.
pub const FLUX2_KLEIN_MLX_4BIT_REVISION: &str = "860e87183ceb29e39627c0612ebd66d8ea66e68c";

/// Whether `model_id` names the 4-bit build, with or without its provider.
#[must_use]
pub(crate) fn identifies_mlx_4bit_build(model_id: &str) -> bool {
    model_id == FLUX2_KLEIN_MLX_4BIT_PROVIDER_MODEL_ID
        || model_id.rsplit('/').next() == Some(FLUX2_KLEIN_MLX_4BIT_MODEL_ID)
}

#[cfg(test)]
mod tests {
    use super::identifies_mlx_4bit_build;

    #[test]
    fn should_identify_the_build_by_provider_or_repository_name() {
        assert!(identifies_mlx_4bit_build(
            "mlx-community/flux2-klein-4b-4bit"
        ));
        assert!(identifies_mlx_4bit_build("flux2-klein-4b-4bit"));
        assert!(identifies_mlx_4bit_build("someone/flux2-klein-4b-4bit"));
    }

    #[test]
    fn should_not_identify_the_bf16_build_or_a_near_name() {
        assert!(!identifies_mlx_4bit_build(
            "black-forest-labs/FLUX.2-klein-4B"
        ));
        assert!(!identifies_mlx_4bit_build(
            "mlx-community/flux2-klein-9b-4bit"
        ));
        assert!(!identifies_mlx_4bit_build(""));
    }
}
