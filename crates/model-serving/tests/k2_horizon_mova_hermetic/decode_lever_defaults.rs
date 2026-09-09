//! The served decode configuration must be the measured-most-performant one.
//!
//! Two experimental decode levers were implemented and measured against the
//! gathered bfloat16 baseline (issue #424): the 8-bit KV slab lost ~1 tok/s
//! and the fused single-token expert kernels lost ~2 tok/s at warm JIT. Both
//! stay merged-in behind opt-in flags. These tests pin the defaults so an
//! accidental default flip is caught before a slower configuration serves.

use astronomical_model_serving::K2HorizonMoVAServingSettings;

#[test]
fn should_default_the_k2_serving_settings_to_the_measured_fast_path() {
    let serving_settings = K2HorizonMoVAServingSettings::default_fixed();
    assert!(
        !serving_settings.quantized_kv_cache_enabled,
        "the 8-bit KV slab measured slower than bfloat16 and must stay opt-in"
    );
    assert!(
        !serving_settings.fused_expert_decode_enabled,
        "the fused expert decode kernels measured slower than the gathered chain and must stay opt-in"
    );
    assert!(
        !serving_settings.decode_stage_attribution_enabled,
        "attribution-gated stage evaluations roughly double decode time and must stay diagnostic-only"
    );
}
