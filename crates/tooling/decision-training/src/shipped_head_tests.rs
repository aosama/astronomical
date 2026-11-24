//! Unit tests for reading a folder's trainable part as it serves today.

use decision_contract::head_tuning_file::{TunedTensor, head_tuning_bytes};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

use super::*;
use crate::test_support::sequence_values;

/// Stored tensors: name, dtype, shape, values (widened from 16-bit where asked).
type StoredTensor = (&'static str, Dtype, Vec<usize>, Vec<f32>);

fn stored_bytes(dtype: Dtype, values: &[f32]) -> Vec<u8> {
    let mut encoded_bytes = Vec::new();
    for float_value in values {
        match dtype {
            Dtype::BF16 => encoded_bytes
                .extend_from_slice(&((float_value.to_bits() >> 16) as u16).to_le_bytes()),
            _ => encoded_bytes.extend_from_slice(&float_value.to_le_bytes()),
        }
    }
    encoded_bytes
}

fn write_safetensors(file_path: &Path, stored_tensors: &[StoredTensor]) {
    let mut encoded_tensors = Vec::with_capacity(stored_tensors.len());
    for (_, dtype, _, values) in stored_tensors {
        encoded_tensors.push(stored_bytes(*dtype, values));
    }
    let mut views = Vec::with_capacity(stored_tensors.len());
    for ((tensor_name, dtype, shape, _), encoded) in stored_tensors.iter().zip(&encoded_tensors) {
        views.push((
            *tensor_name,
            TensorView::new(*dtype, shape.clone(), encoded).unwrap(),
        ));
    }
    std::fs::write(file_path, safetensors::serialize(views, None).unwrap()).unwrap();
}

fn kev_folder(directory: &Path) {
    write_safetensors(
        &directory.join("kev_head.safetensors"),
        &[
            ("q.weight", Dtype::F32, vec![2, 3], sequence_values(6, 1)),
            ("q.bias", Dtype::F32, vec![2], vec![0.5, -0.5]),
            ("k.weight", Dtype::F32, vec![2, 3], sequence_values(6, 2)),
            ("k.bias", Dtype::F32, vec![2], vec![1.0, 2.0]),
        ],
    );
}

#[test]
fn should_start_kev_from_its_shipped_head() {
    let directory = tempfile::tempdir().unwrap();
    kev_folder(directory.path());
    let shipped_head = read_shipped_head(directory.path(), TunableHeadFamily::Kev).unwrap();
    let layout = shipped_head.trainable_head.layout();
    assert_eq!(layout.parameter_count, 16);
    assert_eq!(
        layout
            .tensor("k.bias")
            .values(&shipped_head.served_parameters),
        &[1.0, 2.0]
    );
    assert_eq!(
        layout
            .tensor("q.weight")
            .values(&shipped_head.served_parameters),
        sequence_values(6, 1).as_slice()
    );
}

#[test]
fn should_lay_earlier_training_over_the_shipped_weights() {
    let directory = tempfile::tempdir().unwrap();
    kev_folder(directory.path());
    let earlier = HeadTuning {
        tensors: vec![TunedTensor {
            name: "k.bias".to_owned(),
            shape: vec![2],
            values: vec![7.0, 8.0],
        }],
    };
    std::fs::write(
        directory.path().join(DECISION_HEAD_TUNING_FILE_NAME),
        head_tuning_bytes(&earlier).unwrap(),
    )
    .unwrap();
    let shipped_head = read_shipped_head(directory.path(), TunableHeadFamily::Kev).unwrap();
    let layout = shipped_head.trainable_head.layout();
    assert_eq!(
        layout
            .tensor("k.bias")
            .values(&shipped_head.served_parameters),
        &[7.0, 8.0]
    );
    assert_eq!(
        layout
            .tensor("q.bias")
            .values(&shipped_head.served_parameters),
        &[0.5, -0.5]
    );
}

#[test]
fn should_widen_clefs_bfloat16_head_and_find_its_widths() {
    let directory = tempfile::tempdir().unwrap();
    write_safetensors(
        &directory.path().join("joint_head.safetensors"),
        &[
            (
                "residual_scorer.0.weight",
                Dtype::BF16,
                vec![3, 8],
                vec![0.5; 24],
            ),
            (
                "residual_scorer.0.bias",
                Dtype::BF16,
                vec![3],
                vec![0.25; 3],
            ),
            (
                "residual_scorer.3.weight",
                Dtype::BF16,
                vec![1, 3],
                vec![1.0; 3],
            ),
            ("residual_scorer.3.bias", Dtype::BF16, vec![1], vec![-1.0]),
            ("prior_logit_scale", Dtype::BF16, Vec::new(), vec![2.0]),
            ("joint_logit_scale", Dtype::BF16, Vec::new(), vec![3.0]),
            ("residual_gate", Dtype::BF16, Vec::new(), vec![-0.5]),
            ("field_norm.weight", Dtype::BF16, vec![2], vec![1.0, 1.0]),
        ],
    );
    let shipped_head = read_shipped_head(directory.path(), TunableHeadFamily::Clef).unwrap();
    let layout = shipped_head.trainable_head.layout();
    assert_eq!(layout.tensor("residual_scorer.0.weight").shape, vec![3, 8]);
    assert_eq!(
        layout
            .tensor("residual_gate")
            .values(&shipped_head.served_parameters),
        &[-0.5]
    );
}

#[test]
fn should_refuse_a_folder_without_the_head_or_with_a_wrong_shape() {
    let directory = tempfile::tempdir().unwrap();
    assert!(matches!(
        read_shipped_head(directory.path(), TunableHeadFamily::Laya),
        Err(ShippedHeadError::Read(_))
    ));
    write_safetensors(
        &directory.path().join("model.safetensors"),
        &[
            ("scorer.layers.0.weight", Dtype::F32, vec![4], vec![1.0; 4]),
            ("scorer.layers.0.bias", Dtype::F32, vec![4], vec![0.0; 4]),
            (
                "scorer.layers.1.weight",
                Dtype::F32,
                vec![3, 4],
                vec![0.1; 12],
            ),
            ("scorer.layers.1.bias", Dtype::F32, vec![3], vec![0.0; 3]),
            // The output layer must read the hidden width (3), not 2.
            (
                "scorer.layers.3.weight",
                Dtype::F32,
                vec![1, 2],
                vec![1.0; 2],
            ),
            ("scorer.layers.3.bias", Dtype::F32, vec![1], vec![0.0]),
        ],
    );
    assert!(matches!(
        read_shipped_head(directory.path(), TunableHeadFamily::Laya),
        Err(ShippedHeadError::UnexpectedShape { tensor_name, .. }) if tensor_name == "scorer.layers.3.weight"
    ));
}
