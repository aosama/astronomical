//! Unit tests for the trained-head sidecar file.

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use super::*;

fn kev_tuning() -> HeadTuning {
    HeadTuning {
        tensors: vec![
            TunedTensor {
                name: "q.bias".to_owned(),
                shape: vec![2],
                values: vec![0.125, -3.0e-7],
            },
            TunedTensor {
                name: "q.weight".to_owned(),
                shape: vec![2, 3],
                values: vec![1.0, 2.0, 3.0, -4.0, 5.5, 1.0e-30],
            },
        ],
    }
}

fn single_tensor_file(
    tensor_name: &str,
    dtype: Dtype,
    value_bytes: &[u8],
    shape: &[usize],
) -> Vec<u8> {
    let tensor_view = TensorView::new(dtype, shape.to_vec(), value_bytes).unwrap();
    safetensors::serialize([(tensor_name, tensor_view)], None).unwrap()
}

#[test]
fn should_round_trip_every_value_bit_for_bit() {
    let file_bytes = head_tuning_bytes(&kev_tuning()).unwrap();
    let read_back = parse_head_tuning(&file_bytes, TunableHeadFamily::Kev).unwrap();
    assert_eq!(read_back, kev_tuning());
    let weight = read_back.tensor("q.weight").unwrap();
    assert_eq!(weight.values[5].to_bits(), 1.0e-30_f32.to_bits());
    assert!(read_back.tensor("k.weight").is_none());
}

#[test]
fn should_keep_a_scalar_as_a_shapeless_tensor() {
    let gate = HeadTuning {
        tensors: vec![TunedTensor {
            name: "residual_gate".to_owned(),
            shape: Vec::new(),
            values: vec![0.75],
        }],
    };
    let read_back =
        parse_head_tuning(&head_tuning_bytes(&gate).unwrap(), TunableHeadFamily::Clef).unwrap();
    assert_eq!(read_back, gate);
}

#[test]
fn should_refuse_a_tensor_another_family_or_no_family_may_change() {
    let file_bytes = head_tuning_bytes(&kev_tuning()).unwrap();
    assert_eq!(
        parse_head_tuning(&file_bytes, TunableHeadFamily::Laya),
        Err(HeadTuningFileError::UntunableTensor("q.bias".to_owned()))
    );
    let backbone_file = single_tensor_file(
        "model.layers.0.mlp.up_proj.weight",
        Dtype::F32,
        &1.0_f32.to_le_bytes(),
        &[1],
    );
    assert!(matches!(
        parse_head_tuning(&backbone_file, TunableHeadFamily::Kev),
        Err(HeadTuningFileError::UntunableTensor(_))
    ));
}

#[test]
fn should_refuse_reduced_precision_and_non_finite_values() {
    let bfloat_file = single_tensor_file("q.bias", Dtype::BF16, &[0, 63], &[1]);
    assert_eq!(
        parse_head_tuning(&bfloat_file, TunableHeadFamily::Kev),
        Err(HeadTuningFileError::NotFloat32("q.bias".to_owned()))
    );
    let nan_file = single_tensor_file("q.bias", Dtype::F32, &f32::NAN.to_le_bytes(), &[1]);
    assert_eq!(
        parse_head_tuning(&nan_file, TunableHeadFamily::Kev),
        Err(HeadTuningFileError::NonFiniteValue("q.bias".to_owned()))
    );
}

#[test]
fn should_refuse_an_empty_file_and_bytes_that_are_not_safetensors() {
    let empty_file = head_tuning_bytes(&HeadTuning::default()).unwrap();
    assert_eq!(
        parse_head_tuning(&empty_file, TunableHeadFamily::Clef),
        Err(HeadTuningFileError::NoTensors)
    );
    assert!(matches!(
        parse_head_tuning(
            b"Two households, both alike in dignity",
            TunableHeadFamily::Clef
        ),
        Err(HeadTuningFileError::NotSafetensors(_))
    ));
}

#[test]
fn should_refuse_to_write_values_that_do_not_fill_their_shape() {
    let short_tensor = HeadTuning {
        tensors: vec![TunedTensor {
            name: "q.weight".to_owned(),
            shape: vec![2, 2],
            values: vec![1.0],
        }],
    };
    assert!(head_tuning_bytes(&short_tensor).is_err());
}
