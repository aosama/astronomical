use crate::qwen3_5::Qwen3_5Config;
use crate::qwen3_5::artifacts::tensor_spec::{
    append_qwen3_5_quantized_affine_tensor_profiles, qwen3_5_tensor_profile,
};
use crate::{TensorDtype, TensorProfile};

pub(crate) fn append_qwen3_5_moe_feed_forward_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    decoder_layer_prefix: &str,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) {
    let mixture_of_experts_prefix = format!("{decoder_layer_prefix}.mlp");
    let router_module_name = format!("{mixture_of_experts_prefix}.gate");
    let router_quantization_profile =
        qwen3_5_config.quantization_profile_for_module(&router_module_name);
    if router_quantization_profile.is_unquantized() {
        tensor_profiles.push(qwen3_5_tensor_profile(
            format!("{router_module_name}.weight"),
            TensorDtype::BFloat16,
            vec![qwen3_5_config.expert_count() as usize, hidden_size],
        ));
    } else {
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &router_module_name,
            &[qwen3_5_config.expert_count() as usize],
            hidden_size,
            router_quantization_profile,
        );
    }
    for (projection_name, output_dimension, input_dimension) in [
        (
            "gate_proj",
            qwen3_5_config.expert_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "up_proj",
            qwen3_5_config.expert_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "down_proj",
            hidden_size,
            qwen3_5_config.expert_intermediate_size() as usize,
        ),
    ] {
        let projection_module_name =
            format!("{mixture_of_experts_prefix}.switch_mlp.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &projection_module_name,
            &[qwen3_5_config.expert_count() as usize, output_dimension],
            input_dimension,
            qwen3_5_config.quantization_profile_for_module(&projection_module_name),
        );
    }
    for (projection_name, output_dimension, input_dimension) in [
        (
            "gate_proj",
            qwen3_5_config.shared_expert_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "up_proj",
            qwen3_5_config.shared_expert_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "down_proj",
            hidden_size,
            qwen3_5_config.shared_expert_intermediate_size() as usize,
        ),
    ] {
        let projection_module_name =
            format!("{mixture_of_experts_prefix}.shared_expert.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &projection_module_name,
            &[output_dimension],
            input_dimension,
            qwen3_5_config.quantization_profile_for_module(&projection_module_name),
        );
    }
    let shared_expert_gate_module_name = format!("{mixture_of_experts_prefix}.shared_expert_gate");
    append_qwen3_5_quantized_affine_tensor_profiles(
        tensor_profiles,
        &shared_expert_gate_module_name,
        &[1],
        hidden_size,
        qwen3_5_config.quantization_profile_for_module(&shared_expert_gate_module_name),
    );
}

pub(crate) fn is_sparse_selected_expert_tensor_name(tensor_name: &str) -> bool {
    tensor_name.contains(".mlp.switch_mlp.")
}
