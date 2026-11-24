//! The Qwen 3.5 layout builder's handling of per-layer execution dtypes,
//! checked on the frozen Ornith 1.0 configuration.

use engine_core::astro_decoder_state_cache::{
    AstroDecoderStateCacheLayoutError, AstroDecoderStateCacheTensorDtype,
};

use super::{Qwen3_5AstroDecoderStateCacheLayerDtypes, qwen3_5_astro_decoder_state_cache_layout};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config;

#[test]
fn should_preserve_mixed_execution_dtypes_in_astro_ssd_prompt_cache_geometry() {
    let ornith_config = frozen_ornith_1_0_config();
    let mut astro_decoder_state_cache_layer_dtypes =
        bfloat16_astro_decoder_state_cache_layer_dtypes(&ornith_config);
    astro_decoder_state_cache_layer_dtypes[0] =
        Qwen3_5AstroDecoderStateCacheLayerDtypes::LinearAttention {
            convolution: AstroDecoderStateCacheTensorDtype::Float32,
        };
    astro_decoder_state_cache_layer_dtypes[3] =
        Qwen3_5AstroDecoderStateCacheLayerDtypes::FullAttention {
            keys: AstroDecoderStateCacheTensorDtype::Float32,
            values: AstroDecoderStateCacheTensorDtype::Float32,
        };

    let astro_decoder_state_cache_layout = qwen3_5_astro_decoder_state_cache_layout(
        &ornith_config,
        256,
        &astro_decoder_state_cache_layer_dtypes,
    )
    .expect("mixed execution dtypes should build an exact astro-decoder-state-cache layout");

    let first_boundary_tensor = astro_decoder_state_cache_layout
        .boundary_tensor_layouts()
        .into_iter()
        .find(|persisted_tensor_layout| persisted_tensor_layout.decoder_layer_index() == 0)
        .expect("the first linear-attention layer should have boundary state");
    assert_eq!(
        first_boundary_tensor.tensor_layout().dtype(),
        AstroDecoderStateCacheTensorDtype::Float32
    );
    let first_full_attention_tensors = astro_decoder_state_cache_layout
        .sequence_tensor_layouts()
        .into_iter()
        .filter(|persisted_tensor_layout| persisted_tensor_layout.decoder_layer_index() == 3)
        .collect::<Vec<_>>();
    assert_eq!(first_full_attention_tensors.len(), 2);
    assert!(
        first_full_attention_tensors
            .iter()
            .all(|persisted_tensor_layout| {
                persisted_tensor_layout.tensor_layout().dtype()
                    == AstroDecoderStateCacheTensorDtype::Float32
            })
    );
}

#[test]
fn should_reject_astro_decoder_state_cache_execution_dtypes_with_a_missing_layer() {
    let ornith_config = frozen_ornith_1_0_config();
    let mut astro_decoder_state_cache_layer_dtypes =
        bfloat16_astro_decoder_state_cache_layer_dtypes(&ornith_config);
    astro_decoder_state_cache_layer_dtypes.pop();

    let rejection = qwen3_5_astro_decoder_state_cache_layout(
        &ornith_config,
        256,
        &astro_decoder_state_cache_layer_dtypes,
    );

    assert!(matches!(
        rejection,
        Err(
            AstroDecoderStateCacheLayoutError::ExecutionDtypeLayerCountMismatch {
                expected_layer_count: 40,
                actual_layer_count: 39,
            }
        )
    ));
}

#[test]
fn should_reject_astro_decoder_state_cache_execution_dtypes_for_the_wrong_attention_family() {
    let ornith_config = frozen_ornith_1_0_config();
    let mut astro_decoder_state_cache_layer_dtypes =
        bfloat16_astro_decoder_state_cache_layer_dtypes(&ornith_config);
    astro_decoder_state_cache_layer_dtypes[0] =
        Qwen3_5AstroDecoderStateCacheLayerDtypes::FullAttention {
            keys: AstroDecoderStateCacheTensorDtype::BFloat16,
            values: AstroDecoderStateCacheTensorDtype::BFloat16,
        };

    let rejection = qwen3_5_astro_decoder_state_cache_layout(
        &ornith_config,
        256,
        &astro_decoder_state_cache_layer_dtypes,
    );

    assert!(matches!(
        rejection,
        Err(
            AstroDecoderStateCacheLayoutError::ExecutionDtypeLayerFamilyMismatch { layer_index: 0 }
        )
    ));
}

fn bfloat16_astro_decoder_state_cache_layer_dtypes(
    ornith_config: &Qwen3_5Config,
) -> Vec<Qwen3_5AstroDecoderStateCacheLayerDtypes> {
    (0..ornith_config.layer_count() as usize)
        .map(|decoder_layer_index| {
            if ornith_config.decoder_layer_is_full_attention(decoder_layer_index) {
                Qwen3_5AstroDecoderStateCacheLayerDtypes::FullAttention {
                    keys: AstroDecoderStateCacheTensorDtype::BFloat16,
                    values: AstroDecoderStateCacheTensorDtype::BFloat16,
                }
            } else {
                Qwen3_5AstroDecoderStateCacheLayerDtypes::LinearAttention {
                    convolution: AstroDecoderStateCacheTensorDtype::BFloat16,
                }
            }
        })
        .collect()
}
