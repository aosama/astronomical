use astronomical_model_serving::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertSourceInterval,
    QuantizedTensorSource, SafetensorsDtype,
};

pub(super) fn make_source_interval(
    tensor_name: &str,
    source_file_offset: u64,
    source_byte_count: usize,
    virtual_payload_offset: u64,
) -> QuantizedExpertSourceInterval {
    QuantizedExpertSourceInterval {
        tensor_name: tensor_name.to_owned(),
        expert_start: 0,
        expert_count: 1,
        source_file_offset,
        source_byte_count,
        virtual_payload_offset,
    }
}

pub(super) fn synthetic_layer_plan(layer_prefix: &str) -> QuantizedExpertLayerPlan {
    let expert_capacity = 8;
    let hidden_dimension = 1024;
    let quantization_bits = 6;
    let quantization_group_size = 128;
    let packed_width = hidden_dimension * quantization_bits as usize / 32;
    let packed_bytes_per_expert = packed_width * 4;
    let scales_bytes_per_expert = (hidden_dimension / quantization_group_size) * 2;
    let biases_bytes_per_expert = scales_bytes_per_expert;
    let weight_offset = 100_u64;
    let scales_offset = weight_offset + (packed_bytes_per_expert * expert_capacity) as u64;
    let biases_offset = scales_offset + (scales_bytes_per_expert * expert_capacity) as u64;
    let total_file_size = biases_offset + (biases_bytes_per_expert * expert_capacity) as u64;
    let make_source = |parameter_name: &str, bytes_per_expert: usize, payload_offset: u64| {
        QuantizedTensorSource {
            tensor_name: format!("{layer_prefix}.switch_mlp.{parameter_name}"),
            projection_name: "gate_proj".to_owned(),
            parameter_name: parameter_name.to_owned(),
            quantization_bits,
            quantization_group_size: quantization_group_size as i32,
            source_file: "model-00001-of-00005.safetensors".into(),
            source_file_size_bytes: total_file_size,
            dtype: if parameter_name == "weight" {
                SafetensorsDtype::Uint32
            } else {
                SafetensorsDtype::BFloat16
            },
            full_shape: if parameter_name == "weight" {
                vec![expert_capacity, packed_width]
            } else {
                vec![expert_capacity, hidden_dimension / quantization_group_size]
            },
            tensor_payload_offset: payload_offset,
            bytes_per_expert,
            expert_capacity,
        }
    };
    QuantizedExpertLayerPlan {
        layer_prefix: layer_prefix.to_owned(),
        tensor_sources: vec![
            make_source("weight", packed_bytes_per_expert, weight_offset),
            make_source("scales", scales_bytes_per_expert, scales_offset),
            make_source("biases", biases_bytes_per_expert, biases_offset),
        ],
        expert_capacity,
        quantization_bits,
        quantization_group_size: quantization_group_size as i32,
        quantization_mode: QuantizationMode::Affine,
    }
}
