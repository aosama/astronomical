use super::*;

fn tensor(name: &str, values: &[f32]) -> TrainedTensor {
    TrainedTensor {
        name: name.to_owned(),
        shape: vec![values.len()],
        values: FloatVector(values.to_vec()),
    }
}

fn round_trip(handed_over_tensors: &HandedOverTensors) -> HandedOverTensors {
    read_trained_tensors_file(&trained_tensors_file_bytes(handed_over_tensors.clone()).unwrap())
        .unwrap()
}

/// Add-ons for twelve layers in forward order, so positions reach two
/// digits and name order differs from position order ("layers.10" sorts
/// before "layers.2"): only zero-padded positions read back in order.
fn whole_model_tensors() -> HandedOverTensors {
    let mut adapter_tensors = Vec::new();
    for layer_index in [2, 10, 1, 3, 4, 5, 6, 7, 8, 9, 11, 0] {
        let target = format!("model.layers.{layer_index}.mlp.up_proj");
        adapter_tensors.push(tensor(
            &format!("{target}.adapter_down"),
            &[layer_index as f32],
        ));
        adapter_tensors.push(tensor(
            &format!("{target}.adapter_up"),
            &[-(layer_index as f32)],
        ));
    }
    HandedOverTensors {
        trained_tensors: vec![
            tensor("scorer.weight", &[0.5, -0.25]),
            tensor("scorer.bias", &[1.0]),
        ],
        adapter_tensors,
    }
}

#[test]
fn should_read_back_every_tensor_in_the_order_it_was_written() {
    let handed_over_tensors = whole_model_tensors();
    assert!(handed_over_tensors.adapter_tensors.len() >= 12);

    assert_eq!(round_trip(&handed_over_tensors), handed_over_tensors);
}

#[test]
fn should_hand_over_final_layers_with_no_add_ons_and_add_ons_with_no_final_layers() {
    let final_layers_only = HandedOverTensors {
        trained_tensors: vec![tensor("scorer.bias", &[1.0])],
        adapter_tensors: Vec::new(),
    };
    let add_ons_only = HandedOverTensors {
        trained_tensors: Vec::new(),
        adapter_tensors: vec![tensor("layers.0.q.adapter_down", &[1.0])],
    };

    assert_eq!(round_trip(&final_layers_only), final_layers_only);
    assert_eq!(round_trip(&add_ons_only), add_ons_only);
}

#[test]
fn should_keep_a_scalar_and_a_name_with_slashes() {
    let handed_over_tensors = HandedOverTensors {
        trained_tensors: vec![
            TrainedTensor {
                name: "head/gate".to_owned(),
                shape: Vec::new(),
                values: FloatVector(vec![0.5]),
            },
            tensor("head/layers/0/bias", &[1.0, 2.0]),
        ],
        adapter_tensors: Vec::new(),
    };

    assert_eq!(round_trip(&handed_over_tensors), handed_over_tensors);
}

#[test]
fn should_refuse_to_write_a_value_that_is_not_a_finite_number() {
    let with_nan = HandedOverTensors {
        trained_tensors: vec![tensor("scorer.bias", &[f32::NAN])],
        adapter_tensors: Vec::new(),
    };

    assert_eq!(
        trained_tensors_file_bytes(with_nan),
        Err(TrainedTensorsFileError::NonFiniteValue(
            "scorer.bias".to_owned()
        ))
    );
}

fn file_of_named_tensors(tensor_name: &str) -> Vec<u8> {
    float32_tensors_bytes(&[TunedTensor {
        name: tensor_name.to_owned(),
        shape: vec![1],
        values: vec![1.0],
    }])
    .unwrap()
}

#[test]
fn should_refuse_tensors_without_both_marks_or_with_an_unknown_list() {
    for unmarked_name in [
        "scorer.bias",
        "trained/scorer.bias",
        "other/00000000/scorer.bias",
    ] {
        assert_eq!(
            read_trained_tensors_file(&file_of_named_tensors(unmarked_name)),
            Err(TrainedTensorsFileError::UnmarkedTensor(
                unmarked_name.to_owned()
            )),
            "{unmarked_name}"
        );
    }
}

#[test]
fn should_refuse_bytes_that_are_not_safetensors() {
    assert!(matches!(
        read_trained_tensors_file(b"Romeo, Romeo"),
        Err(TrainedTensorsFileError::Tensors(_))
    ));
}

#[test]
fn should_name_and_resolve_a_hand_over_file_inside_its_folder() {
    let file_name = handover_file_name(4_242, 1_794_886_098_413_661_000);
    let folder = Path::new("/private/var/folders/example/T");

    assert_eq!(
        handover_file_path(folder, &file_name),
        Some(folder.join(&file_name))
    );
}

#[test]
fn should_refuse_any_other_file_name() {
    let folder = Path::new("/private/var/folders/example/T");
    for other_name in [
        "../../Documents/romeo.txt",
        "astronomical-trained-tensors-1-2.safetensors/../x",
        "astronomical-trained-tensors-../../x.safetensors",
        "astronomical-trained-tensors-.safetensors",
        "romeo.safetensors",
        "astronomical-trained-tensors-12-34.json",
    ] {
        assert_eq!(handover_file_path(folder, other_name), None, "{other_name}");
    }
}
