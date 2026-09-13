//! Writes a Core ML NeuralNetwork snapshot for the expert-route predictor.
//!
//! One grouped 1x1 convolution per affine map so the Neural Engine sees a
//! single large conv, not dozens of tiny linears. No Python converter.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

const FLOAT32_ARRAY: u64 = 65_568;
const EXACT_ARRAY_MAPPING: u64 = 1;
const SPECIFICATION_VERSION: u64 = 4;

/// Frozen 1x1-convolution weights for every sparse layer, concatenated in
/// layer order. Training copies this snapshot; it does not live in MLX.
#[derive(Clone, Debug)]
pub struct PredictorAneConvolutionSnapshot {
    pub layer_count: usize,
    pub expert_count: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub conv1_weights: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2_weights: Vec<f32>,
    pub conv2_bias: Vec<f32>,
}

impl PredictorAneConvolutionSnapshot {
    #[must_use]
    pub fn grouped_input_channels(&self) -> usize {
        self.layer_count.saturating_mul(self.input_dim)
    }

    #[must_use]
    pub fn grouped_hidden_channels(&self) -> usize {
        self.layer_count.saturating_mul(self.hidden_dim)
    }

    #[must_use]
    pub fn grouped_output_channels(&self) -> usize {
        self.layer_count.saturating_mul(self.expert_count)
    }
}

/// Serializes the snapshot as a `.mlmodel` NeuralNetwork document.
pub fn write_predictor_mlmodel(
    snapshot: &PredictorAneConvolutionSnapshot,
    mlmodel_path: &Path,
) -> io::Result<()> {
    let document = encode_neural_network_model(snapshot).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "predictor convolution snapshot is incomplete",
        )
    })?;
    if let Some(parent_directory) = mlmodel_path.parent() {
        fs::create_dir_all(parent_directory)?;
    }
    let mut model_file = fs::File::create(mlmodel_path)?;
    model_file.write_all(&document)?;
    Ok(())
}

fn encode_neural_network_model(snapshot: &PredictorAneConvolutionSnapshot) -> Option<Vec<u8>> {
    if snapshot.layer_count == 0 || snapshot.input_dim == 0 || snapshot.hidden_dim == 0 {
        return None;
    }
    let input_channels = snapshot.grouped_input_channels();
    let hidden_channels = snapshot.grouped_hidden_channels();
    let output_channels = snapshot.grouped_output_channels();
    let expected_conv1 = hidden_channels.checked_mul(snapshot.input_dim)?;
    let expected_conv2 = output_channels.checked_mul(snapshot.hidden_dim)?;
    if snapshot.conv1_weights.len() != expected_conv1
        || snapshot.conv1_bias.len() != hidden_channels
        || snapshot.conv2_weights.len() != expected_conv2
        || snapshot.conv2_bias.len() != output_channels
    {
        return None;
    }

    let mut description = Vec::new();
    append_bytes(
        &mut description,
        1,
        &feature("head_inputs", &[1, input_channels as i64, 1, 1]),
    );
    append_bytes(
        &mut description,
        10,
        &feature("logits", &[1, output_channels as i64, 1, 1]),
    );

    let mut network = Vec::new();
    append_bytes(
        &mut network,
        1,
        &convolution_layer(
            "fc1",
            "head_inputs",
            "hidden",
            hidden_channels as u64,
            snapshot.input_dim as u64,
            snapshot.layer_count as u64,
            &snapshot.conv1_weights,
            &snapshot.conv1_bias,
        ),
    );
    append_bytes(
        &mut network,
        1,
        &leaky_relu_layer("lrelu", "hidden", "hidden_activated"),
    );
    append_bytes(
        &mut network,
        1,
        &convolution_layer(
            "fc2",
            "hidden_activated",
            "logits",
            output_channels as u64,
            snapshot.hidden_dim as u64,
            snapshot.layer_count as u64,
            &snapshot.conv2_weights,
            &snapshot.conv2_bias,
        ),
    );
    append_varint_field(&mut network, 5, EXACT_ARRAY_MAPPING);

    let mut model = Vec::new();
    append_varint_field(&mut model, 1, SPECIFICATION_VERSION);
    append_bytes(&mut model, 2, &description);
    append_bytes(&mut model, 500, &network);
    Some(model)
}

fn feature(name: &str, shape: &[i64]) -> Vec<u8> {
    let mut array_type = Vec::new();
    for dimension in shape {
        append_varint_field(&mut array_type, 1, *dimension as u64);
    }
    append_varint_field(&mut array_type, 2, FLOAT32_ARRAY);
    let mut feature_type = Vec::new();
    append_bytes(&mut feature_type, 5, &array_type);
    let mut feature = Vec::new();
    append_string(&mut feature, 1, name);
    append_bytes(&mut feature, 3, &feature_type);
    feature
}

fn convolution_layer(
    name: &str,
    input_name: &str,
    output_name: &str,
    output_channels: u64,
    kernel_channels: u64,
    group_count: u64,
    weights: &[f32],
    bias: &[f32],
) -> Vec<u8> {
    let mut convolution = Vec::new();
    append_varint_field(&mut convolution, 1, output_channels);
    append_varint_field(&mut convolution, 2, kernel_channels);
    append_varint_field(&mut convolution, 10, group_count);
    append_varint_field(&mut convolution, 20, 1);
    append_varint_field(&mut convolution, 20, 1);
    append_varint_field(&mut convolution, 30, 1);
    append_varint_field(&mut convolution, 30, 1);
    append_bytes(&mut convolution, 50, &[]);
    append_varint_field(&mut convolution, 70, 1);
    append_bytes(&mut convolution, 90, &weight_blob(weights));
    append_bytes(&mut convolution, 91, &weight_blob(bias));

    let mut layer = Vec::new();
    append_string(&mut layer, 1, name);
    append_string(&mut layer, 2, input_name);
    append_string(&mut layer, 3, output_name);
    append_bytes(&mut layer, 100, &convolution);
    layer
}

fn leaky_relu_layer(name: &str, input_name: &str, output_name: &str) -> Vec<u8> {
    let mut leaky = Vec::new();
    append_f32_field(&mut leaky, 1, 0.01);
    let mut activation = Vec::new();
    append_bytes(&mut activation, 15, &leaky);
    let mut layer = Vec::new();
    append_string(&mut layer, 1, name);
    append_string(&mut layer, 2, input_name);
    append_string(&mut layer, 3, output_name);
    append_bytes(&mut layer, 130, &activation);
    layer
}

fn weight_blob(values: &[f32]) -> Vec<u8> {
    let mut packed_floats = Vec::with_capacity(values.len() * 4);
    for value in values {
        packed_floats.extend_from_slice(&value.to_le_bytes());
    }
    let mut blob = Vec::new();
    append_bytes(&mut blob, 1, &packed_floats);
    blob
}

fn append_varint(buffer: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buffer.push((value as u8) | 0x80);
        value >>= 7;
    }
    buffer.push(value as u8);
}

fn append_key(buffer: &mut Vec<u8>, field: u32, wire: u32) {
    append_varint(buffer, u64::from((field << 3) | wire));
}

fn append_varint_field(buffer: &mut Vec<u8>, field: u32, value: u64) {
    append_key(buffer, field, 0);
    append_varint(buffer, value);
}

fn append_bytes(buffer: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    append_key(buffer, field, 2);
    append_varint(buffer, bytes.len() as u64);
    buffer.extend_from_slice(bytes);
}

fn append_string(buffer: &mut Vec<u8>, field: u32, value: &str) {
    append_bytes(buffer, field, value.as_bytes());
}

fn append_f32_field(buffer: &mut Vec<u8>, field: u32, value: f32) {
    append_key(buffer, field, 5);
    buffer.extend_from_slice(&value.to_le_bytes());
}
