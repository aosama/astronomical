use crate::artifact_validation::TensorProfile;
use crate::qwen3_5::Qwen3_5Config;
use crate::qwen3_5::artifacts::tensor_spec::append_qwen3_5_quantized_affine_tensor_profiles;

pub(crate) fn append_qwen3_5_dense_mlp_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    decoder_layer_prefix: &str,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) {
    let dense_mlp_prefix = format!("{decoder_layer_prefix}.mlp");
    for (projection_name, output_dimension, input_dimension) in [
        (
            "gate_proj",
            qwen3_5_config.dense_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "up_proj",
            qwen3_5_config.dense_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "down_proj",
            hidden_size,
            qwen3_5_config.dense_intermediate_size() as usize,
        ),
    ] {
        let projection_module_name = format!("{dense_mlp_prefix}.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &projection_module_name,
            &[output_dimension],
            input_dimension,
            qwen3_5_config.quantization_profile_for_module(&projection_module_name),
        );
    }
}
