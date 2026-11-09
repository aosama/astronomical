use crate::decoder_cache::{DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::decoder::cache_layout::{
    Qwen3_5DecoderLayerCacheDtypes, qwen3_5_decoder_cache_layout,
};
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::MlxArray;
use serial_test::serial;

use super::{RequestDecoderStateStack, composite_first_use_bytes};
use crate::decoder_cache::{ConvolutionState, GatedDeltaRecurrentState};

const KV_SLAB_TOKENS: usize = 256;

fn ornith_decoder_cache_layout() -> DecoderCacheLayout {
    let ornith_config = frozen_ornith_1_0_config();
    qwen3_5_decoder_cache_layout(
        &ornith_config,
        KV_SLAB_TOKENS,
        &bfloat16_decoder_layer_cache_dtypes(&ornith_config),
    )
    .expect("the frozen configuration builds a decoder-cache layout")
}

fn bfloat16_decoder_layer_cache_dtypes(
    ornith_config: &Qwen3_5Config,
) -> Vec<Qwen3_5DecoderLayerCacheDtypes> {
    let mut decoder_layer_cache_dtypes = Vec::new();
    for decoder_layer_index in 0..ornith_config.layer_count() as usize {
        decoder_layer_cache_dtypes.push(
            if ornith_config.decoder_layer_is_full_attention(decoder_layer_index) {
                Qwen3_5DecoderLayerCacheDtypes::FullAttention {
                    keys: DecoderCacheTensorDtype::BFloat16,
                    values: DecoderCacheTensorDtype::BFloat16,
                }
            } else {
                Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
                    convolution: DecoderCacheTensorDtype::BFloat16,
                }
            },
        );
    }
    decoder_layer_cache_dtypes
}

/// Independent expectation for an empty stack: every full-attention layer
/// allocates whole slabs covering the update, and every linear-attention
/// layer allocates its fixed convolution and recurrent state once.
fn expected_first_forward_bytes(
    decoder_cache_layout: &DecoderCacheLayout,
    update_token_count: usize,
) -> usize {
    let slab_count = update_token_count.div_ceil(KV_SLAB_TOKENS);
    let mut expected_bytes = 0;
    for decoder_layer_index in 0..decoder_cache_layout.layer_count() {
        expected_bytes += match decoder_cache_layout.layer(decoder_layer_index) {
            Some(DecoderCacheLayerLayout::AppendOnlyAttention { keys, values, .. }) => {
                slab_count
                    * KV_SLAB_TOKENS
                    * (keys
                        .sequence_payload_byte_count_per_token()
                        .expect("key bytes")
                        + values
                            .sequence_payload_byte_count_per_token()
                            .expect("value bytes"))
            }
            Some(DecoderCacheLayerLayout::Composite { components }) => {
                components.iter().map(fixed_component_bytes).sum()
            }
            unexpected_layout => panic!("unexpected Ornith layer layout {unexpected_layout:?}"),
        };
    }
    expected_bytes
}

fn fixed_component_bytes(component_layout: &DecoderCacheLayerLayout) -> usize {
    match component_layout {
        DecoderCacheLayerLayout::RecurrentTensor { tensor } => {
            tensor.fixed_payload_byte_count().expect("fixed bytes")
        }
        unexpected_component => panic!("unexpected composite component {unexpected_component:?}"),
    }
}

fn layout_layers(decoder_cache_layout: &DecoderCacheLayout) -> Vec<DecoderCacheLayerLayout> {
    let mut decoder_cache_layer_layouts = Vec::new();
    for decoder_layer_index in 0..decoder_cache_layout.layer_count() {
        decoder_cache_layer_layouts.push(
            decoder_cache_layout
                .layer(decoder_layer_index)
                .expect("a declared layer")
                .clone(),
        );
    }
    decoder_cache_layer_layouts
}

fn projected_growth_bytes(update_token_count: usize) -> usize {
    let decoder_cache_layout = ornith_decoder_cache_layout();
    RequestDecoderStateStack::empty_from_decoder_cache_layout(&decoder_cache_layout)
        .expect("an empty stack")
        .projected_persistent_state_growth_bytes(&decoder_cache_layout, update_token_count)
        .expect("a projection for an empty stack")
}

#[test]
fn should_project_one_slab_and_every_fixed_state_for_a_first_token() {
    assert_eq!(
        projected_growth_bytes(1),
        expected_first_forward_bytes(&ornith_decoder_cache_layout(), 1)
    );
}

#[test]
fn should_project_the_same_growth_up_to_a_full_slab_and_more_one_token_past_it() {
    let one_slab_bytes = projected_growth_bytes(KV_SLAB_TOKENS);
    assert_eq!(projected_growth_bytes(1), one_slab_bytes);
    let two_slab_bytes = projected_growth_bytes(KV_SLAB_TOKENS + 1);
    assert!(two_slab_bytes > one_slab_bytes);
    assert_eq!(
        two_slab_bytes,
        expected_first_forward_bytes(&ornith_decoder_cache_layout(), KV_SLAB_TOKENS + 1)
    );
}

#[test]
fn should_refuse_a_layout_with_a_different_layer_count() {
    let decoder_cache_layout = ornith_decoder_cache_layout();
    let empty_stack =
        RequestDecoderStateStack::empty_from_decoder_cache_layout(&decoder_cache_layout)
            .expect("an empty stack");
    let mut shorter_layers = layout_layers(&decoder_cache_layout);
    shorter_layers.pop();
    let shorter_layout =
        DecoderCacheLayout::new(shorter_layers).expect("a shorter layout is itself valid");
    let refusal = empty_stack
        .projected_persistent_state_growth_bytes(&shorter_layout, 1)
        .expect_err("a mismatched layout must be refused");
    assert!(refusal.to_string().contains("layer count"), "{refusal}");
}

#[test]
fn should_refuse_a_layout_whose_layer_families_are_swapped() {
    let decoder_cache_layout = ornith_decoder_cache_layout();
    let empty_stack =
        RequestDecoderStateStack::empty_from_decoder_cache_layout(&decoder_cache_layout)
            .expect("an empty stack");
    let mut reversed_layers = layout_layers(&decoder_cache_layout);
    reversed_layers.reverse();
    let reversed_layout =
        DecoderCacheLayout::new(reversed_layers).expect("a reversed layout is itself valid");
    let refusal = empty_stack
        .projected_persistent_state_growth_bytes(&reversed_layout, 1)
        .expect_err("swapped families must be refused");
    assert!(refusal.to_string().contains("family"), "{refusal}");
}

/// The convolution and recurrent components of the first linear-attention
/// layer of the Ornith layout.
fn first_composite_components() -> Vec<DecoderCacheLayerLayout> {
    for decoder_cache_layer_layout in layout_layers(&ornith_decoder_cache_layout()) {
        if let DecoderCacheLayerLayout::Composite { components } = decoder_cache_layer_layout {
            return components;
        }
    }
    panic!("the Ornith layout has linear-attention layers");
}

/// MLX arrays need an initialized runtime; small limits suffice because the
/// tests only hold one-element arrays.
fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the state stack test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// A one-element state array; only whether a state is present matters to
/// the projection, never its contents.
fn present_state_array() -> MlxArray {
    MlxArray::from_f32(&[0.0], &[1]).expect("a one-element state array")
}

#[test]
fn should_project_the_fixed_state_bytes_of_an_unallocated_linear_attention_layer() {
    let composite_components = first_composite_components();
    let expected_bytes: usize = composite_components.iter().map(fixed_component_bytes).sum();
    assert!(expected_bytes > 0);
    assert_eq!(
        composite_first_use_bytes(
            &ConvolutionState::empty(),
            &GatedDeltaRecurrentState::empty(),
            &composite_components,
        )
        .expect("an unallocated layer projects its fixed state"),
        expected_bytes
    );
}

#[test]
#[serial]
fn should_project_nothing_for_an_already_allocated_linear_attention_layer() {
    let _runtime = test_runtime();
    let mut convolution = ConvolutionState::empty();
    convolution.restore_from_snapshot(present_state_array());
    let mut recurrent = GatedDeltaRecurrentState::empty();
    recurrent.set_next(present_state_array());
    assert_eq!(
        composite_first_use_bytes(&convolution, &recurrent, &first_composite_components())
            .expect("an allocated layer projects no growth"),
        0
    );
}

#[test]
#[serial]
fn should_refuse_a_partially_allocated_linear_attention_layer() {
    let _runtime = test_runtime();
    let mut convolution = ConvolutionState::empty();
    convolution.restore_from_snapshot(present_state_array());
    let refusal = composite_first_use_bytes(
        &convolution,
        &GatedDeltaRecurrentState::empty(),
        &first_composite_components(),
    )
    .expect_err("half-allocated state must be refused");
    assert!(
        refusal.to_string().contains("partially allocated"),
        "{refusal}"
    );
}

#[test]
fn should_refuse_composite_components_that_are_not_convolution_then_recurrent() {
    let mut single_component = first_composite_components();
    single_component.truncate(1);
    let refusal = composite_first_use_bytes(
        &ConvolutionState::empty(),
        &GatedDeltaRecurrentState::empty(),
        &single_component,
    )
    .expect_err("a one-component layout must be refused");
    assert!(
        refusal.to_string().contains("convolution and recurrent"),
        "{refusal}"
    );
}
