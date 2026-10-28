//! Unit coverage for the shared decoder dtype replay: promotion rules, affine
//! and embedding contracts, per-attention and per-MLP flows, and the routed
//! expert plan lookups — positive, boundary, and negative cases. The replay
//! previously lived as two untested copies inside the streaming and resident
//! engines; these tests are the coverage both copies always lacked.

use std::path::PathBuf;

use mlx_c_rust::MlxDtype;

use crate::decoder_cache::DecoderCacheTensorDtype;
use crate::qwen3_5_core::artifacts::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype,
};
use crate::qwen3_5_core::decoder::Qwen3_5DecoderLayerCacheDtypes;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

use super::decoder_cache_dtype_flow::{
    AffineDtypeContract, DenseMlpDtypeFlow, FullAttentionDtypeFlow, LinearAttentionDtypeFlow,
    MixtureOfExpertsDtypeFlow, affine_output_dtype, decoder_cache_tensor_dtype,
    dense_mlp_output_dtype, embedding_output_dtype, expert_parameter_dtype,
    expert_projection_output_dtype, full_attention_dtypes, linear_attention_dtypes,
    mixture_of_experts_output_dtype, promote_floating_dtypes,
};

fn native_affine_contract() -> AffineDtypeContract {
    AffineDtypeContract::NativeBfloat16
}

fn codebook_affine_contract() -> AffineDtypeContract {
    AffineDtypeContract::Jangtq2Codebook
}

fn quantized_affine_contract(scales: MlxDtype, biases: MlxDtype) -> AffineDtypeContract {
    AffineDtypeContract::Quantized { scales, biases }
}

fn expert_tensor_source(
    projection_name: &str,
    parameter_name: &str,
    dtype: SafetensorsDtype,
) -> QuantizedTensorSource {
    QuantizedTensorSource {
        tensor_name: format!("{projection_name}_{parameter_name}"),
        projection_name: projection_name.to_owned(),
        parameter_name: parameter_name.to_owned(),
        quantization_bits: 4,
        quantization_group_size: 64,
        source_file: PathBuf::new(),
        source_file_size_bytes: 0,
        dtype,
        full_shape: Vec::new(),
        tensor_payload_offset: 0,
        bytes_per_expert: 0,
        expert_capacity: 1,
    }
}

/// An affine expert plan: scales and biases companions for every projection.
fn affine_expert_layer_plan(
    scale_dtype: SafetensorsDtype,
    bias_dtype: SafetensorsDtype,
) -> QuantizedExpertLayerPlan {
    let mut tensor_sources = Vec::new();
    for projection_name in ["gate_proj", "up_proj", "down_proj"] {
        tensor_sources.push(expert_tensor_source(projection_name, "scales", scale_dtype));
        tensor_sources.push(expert_tensor_source(projection_name, "biases", bias_dtype));
    }
    QuantizedExpertLayerPlan {
        layer_prefix: "model.layers.7.mlp.experts.0".to_owned(),
        tensor_sources,
        jangtq2_projection_rotations: Vec::new(),
        expert_capacity: 1,
        quantization_bits: 4,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}

/// A plan whose every projection carries a native bfloat16 contract.
fn native_bfloat16_expert_layer_plan() -> QuantizedExpertLayerPlan {
    QuantizedExpertLayerPlan {
        layer_prefix: "model.layers.7.mlp.experts.0".to_owned(),
        tensor_sources: Vec::new(),
        jangtq2_projection_rotations: Vec::new(),
        expert_capacity: 1,
        quantization_bits: 0,
        quantization_group_size: 0,
        quantization_mode: QuantizationMode::NativeBfloat16,
    }
}

fn assert_invalid_input_description(
    result: Result<MlxDtype, Qwen3_5ExecutionError>,
    expected_description: &str,
) {
    let Err(Qwen3_5ExecutionError::InvalidInput { description }) = result else {
        panic!("expected an InvalidInput error, got {result:?}");
    };
    assert_eq!(description, expected_description);
}

#[test]
fn should_promote_identical_float_dtypes_to_the_same_dtype() {
    assert_eq!(
        promote_floating_dtypes(MlxDtype::Float16, MlxDtype::Float16).unwrap(),
        MlxDtype::Float16
    );
    assert_eq!(
        promote_floating_dtypes(MlxDtype::BFloat16, MlxDtype::BFloat16).unwrap(),
        MlxDtype::BFloat16
    );
    assert_eq!(
        promote_floating_dtypes(MlxDtype::Float32, MlxDtype::Float32).unwrap(),
        MlxDtype::Float32
    );
}

#[test]
fn should_promote_mixed_sixteen_bit_dtypes_to_float32_in_both_orders() {
    assert_eq!(
        promote_floating_dtypes(MlxDtype::Float16, MlxDtype::BFloat16).unwrap(),
        MlxDtype::Float32
    );
    assert_eq!(
        promote_floating_dtypes(MlxDtype::BFloat16, MlxDtype::Float16).unwrap(),
        MlxDtype::Float32
    );
}

#[test]
fn should_let_float32_dominate_every_promotion() {
    assert_eq!(
        promote_floating_dtypes(MlxDtype::Float32, MlxDtype::BFloat16).unwrap(),
        MlxDtype::Float32
    );
    assert_eq!(
        promote_floating_dtypes(MlxDtype::Float16, MlxDtype::Float32).unwrap(),
        MlxDtype::Float32
    );
}

#[test]
fn should_reject_non_floating_dtypes_in_promotion() {
    let result = promote_floating_dtypes(MlxDtype::Int32, MlxDtype::Float16);
    assert_invalid_input_description(
        result,
        "decoder dtype flow encountered an unsupported MLX dtype",
    );
}

#[test]
fn should_derive_embedding_dtype_from_the_scale_contract() {
    assert_eq!(
        embedding_output_dtype(&native_affine_contract()).unwrap(),
        MlxDtype::BFloat16
    );
    assert_eq!(
        embedding_output_dtype(&codebook_affine_contract()).unwrap(),
        MlxDtype::BFloat16
    );
    // The embedding output follows the scale dtype alone, without bias
    // promotion: the dequantize primitive declares its output that way.
    assert_eq!(
        embedding_output_dtype(&quantized_affine_contract(
            MlxDtype::Float32,
            MlxDtype::Float16
        ))
        .unwrap(),
        MlxDtype::Float32
    );
}

#[test]
fn should_promote_affine_output_over_the_quantized_parameter_contract() {
    assert_eq!(
        affine_output_dtype(
            MlxDtype::Float16,
            &quantized_affine_contract(MlxDtype::Float16, MlxDtype::Float16),
        )
        .unwrap(),
        MlxDtype::Float16
    );
    // Mixed scale and bias dtypes widen the parameter contract to FP32.
    assert_eq!(
        affine_output_dtype(
            MlxDtype::Float16,
            &quantized_affine_contract(MlxDtype::Float16, MlxDtype::BFloat16),
        )
        .unwrap(),
        MlxDtype::Float32
    );
    assert_eq!(
        affine_output_dtype(MlxDtype::BFloat16, &native_affine_contract()).unwrap(),
        MlxDtype::BFloat16
    );
    assert_eq!(
        affine_output_dtype(MlxDtype::Float16, &codebook_affine_contract()).unwrap(),
        MlxDtype::Float32
    );
}

#[test]
fn should_reject_non_floating_affine_scale_dtypes() {
    let result = affine_output_dtype(
        MlxDtype::Float16,
        &quantized_affine_contract(MlxDtype::Int32, MlxDtype::Int32),
    );
    assert_invalid_input_description(
        result,
        "decoder dtype flow encountered an unsupported MLX dtype",
    );
}

#[test]
fn should_derive_linear_attention_dtypes_across_all_bfloat16_layers() {
    let flow = LinearAttentionDtypeFlow {
        normalized_input_dtype: MlxDtype::BFloat16,
        input_queries_keys_values_projection: native_affine_contract(),
        convolution_weight_dtype: MlxDtype::BFloat16,
        normalization_weight_dtype: MlxDtype::BFloat16,
        output_gate_projection: native_affine_contract(),
        output_projection: native_affine_contract(),
    };

    let (cache_dtypes, output_dtype) = linear_attention_dtypes(&flow).unwrap();

    assert_eq!(
        cache_dtypes,
        Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
            convolution: DecoderCacheTensorDtype::BFloat16,
        }
    );
    assert_eq!(output_dtype, MlxDtype::BFloat16);
}

#[test]
fn should_widen_linear_attention_through_the_convolution_and_gate_promotions() {
    // Float16 activations over a float16 qkv contract keep the convolution
    // state at float16, but the bfloat16 convolution weight and native output
    // gate widen every later stage to float32.
    let flow = LinearAttentionDtypeFlow {
        normalized_input_dtype: MlxDtype::Float16,
        input_queries_keys_values_projection: quantized_affine_contract(
            MlxDtype::Float16,
            MlxDtype::Float16,
        ),
        convolution_weight_dtype: MlxDtype::BFloat16,
        normalization_weight_dtype: MlxDtype::BFloat16,
        output_gate_projection: native_affine_contract(),
        output_projection: native_affine_contract(),
    };

    let (cache_dtypes, output_dtype) = linear_attention_dtypes(&flow).unwrap();

    assert_eq!(
        cache_dtypes,
        Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
            convolution: DecoderCacheTensorDtype::Float16,
        }
    );
    assert_eq!(output_dtype, MlxDtype::Float32);
}

#[test]
fn should_derive_full_attention_cache_dtypes_from_key_and_value_projections() {
    // Keys carry the promoted normalized key dtype; values carry the raw
    // value-projection dtype before any output gating.
    let flow = FullAttentionDtypeFlow {
        normalized_input_dtype: MlxDtype::BFloat16,
        query_projection: native_affine_contract(),
        query_normalization_weight_dtype: MlxDtype::BFloat16,
        key_projection: quantized_affine_contract(MlxDtype::Float32, MlxDtype::Float32),
        key_normalization_weight_dtype: MlxDtype::BFloat16,
        value_projection: native_affine_contract(),
        output_projection: native_affine_contract(),
    };

    let (cache_dtypes, output_dtype) = full_attention_dtypes(&flow).unwrap();

    assert_eq!(
        cache_dtypes,
        Qwen3_5DecoderLayerCacheDtypes::FullAttention {
            keys: DecoderCacheTensorDtype::Float32,
            values: DecoderCacheTensorDtype::BFloat16,
        }
    );
    assert_eq!(output_dtype, MlxDtype::Float32);
}

#[test]
fn should_derive_dense_mlp_output_through_gate_and_up_promotion() {
    let flow = DenseMlpDtypeFlow {
        normalized_attention_dtype: MlxDtype::Float16,
        gate_projection: native_affine_contract(),
        up_projection: native_affine_contract(),
        down_projection: quantized_affine_contract(MlxDtype::Float32, MlxDtype::Float32),
    };

    // gate stays float16, the native up projection widens to float32, and the
    // activated mix plus the float32 down contract pins float32.
    let output_dtype = dense_mlp_output_dtype(&flow).unwrap();

    assert_eq!(output_dtype, MlxDtype::Float32);
}

#[test]
fn should_combine_sparse_and_shared_expert_dtypes_in_mixture_of_experts() {
    let flow = MixtureOfExpertsDtypeFlow {
        normalized_attention_dtype: MlxDtype::Float16,
        shared_expert_gate_projection: native_affine_contract(),
        shared_expert_up_projection: native_affine_contract(),
        shared_expert_down_projection: native_affine_contract(),
        shared_expert_output_gate_projection: native_affine_contract(),
    };
    let expert_layer_plan =
        affine_expert_layer_plan(SafetensorsDtype::Float16, SafetensorsDtype::Float16);

    // The sparse leg stays float16 end to end; the shared leg widens to
    // float32 through its native bfloat16 projections, and the final
    // combination follows the wider side.
    let output_dtype = mixture_of_experts_output_dtype(&flow, &expert_layer_plan).unwrap();

    assert_eq!(output_dtype, MlxDtype::Float32);
}

#[test]
fn should_treat_native_and_codebook_expert_plans_as_bfloat16_contracts() {
    let flow = MixtureOfExpertsDtypeFlow {
        normalized_attention_dtype: MlxDtype::BFloat16,
        shared_expert_gate_projection: native_affine_contract(),
        shared_expert_up_projection: native_affine_contract(),
        shared_expert_down_projection: native_affine_contract(),
        shared_expert_output_gate_projection: native_affine_contract(),
    };
    let expert_layer_plan = native_bfloat16_expert_layer_plan();

    let output_dtype = mixture_of_experts_output_dtype(&flow, &expert_layer_plan).unwrap();

    assert_eq!(output_dtype, MlxDtype::BFloat16);
}

#[test]
fn should_reject_a_missing_expert_affine_parameter() {
    let mut expert_layer_plan =
        affine_expert_layer_plan(SafetensorsDtype::Float16, SafetensorsDtype::Float16);
    // Drop every biases companion: the scales lookups still succeed, but the
    // bias lookup must fail with the exact contract error.
    expert_layer_plan
        .tensor_sources
        .retain(|tensor_source| tensor_source.parameter_name != "biases");

    let result = expert_parameter_dtype(&expert_layer_plan, "gate_proj", "biases");
    let Err(Qwen3_5ExecutionError::InvalidInput { description }) = result else {
        panic!("expected an InvalidInput error");
    };
    assert_eq!(
        description,
        "expert layer plan is missing an affine parameter dtype"
    );
}

#[test]
fn should_reject_non_floating_expert_parameters() {
    let expert_layer_plan =
        affine_expert_layer_plan(SafetensorsDtype::Int32, SafetensorsDtype::Float16);

    let result = expert_parameter_dtype(&expert_layer_plan, "gate_proj", "scales");
    let Err(Qwen3_5ExecutionError::InvalidInput { description }) = result else {
        panic!("expected an InvalidInput error");
    };
    assert_eq!(
        description,
        "expert affine parameter has a non-floating dtype"
    );
}

#[test]
fn should_promote_expert_projection_output_over_affine_companions() {
    let expert_layer_plan =
        affine_expert_layer_plan(SafetensorsDtype::Float16, SafetensorsDtype::Float16);
    assert_eq!(
        expert_projection_output_dtype(MlxDtype::Float16, &expert_layer_plan, "gate_proj").unwrap(),
        MlxDtype::Float16
    );
}

#[test]
fn should_narrow_only_floating_cache_dtypes() {
    assert_eq!(
        decoder_cache_tensor_dtype(MlxDtype::Float16).unwrap(),
        DecoderCacheTensorDtype::Float16
    );
    assert_eq!(
        decoder_cache_tensor_dtype(MlxDtype::Float32).unwrap(),
        DecoderCacheTensorDtype::Float32
    );
    let result = decoder_cache_tensor_dtype(MlxDtype::Int32);
    let Err(Qwen3_5ExecutionError::InvalidInput { description }) = result else {
        panic!("expected an InvalidInput error");
    };
    assert_eq!(
        description,
        "decoder cache state has an unsupported MLX dtype"
    );
}
