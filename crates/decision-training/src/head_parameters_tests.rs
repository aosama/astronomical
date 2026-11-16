//! Unit tests for the flat parameter layout.

use super::*;

fn small_layout() -> ParameterLayout {
    ParameterLayout::new(&[
        ("q.weight", vec![2, 3]),
        ("q.bias", vec![2]),
        ("residual_gate", Vec::new()),
    ])
}

#[test]
fn should_lay_tensors_out_back_to_back_with_a_scalar_taking_one_slot() {
    let layout = small_layout();
    assert_eq!(layout.parameter_count, 9);
    assert_eq!(layout.tensor("q.bias").offset, 6);
    assert_eq!(layout.tensor("residual_gate").length, 1);
}

#[test]
fn should_hand_back_each_tensor_under_its_name_and_shape() {
    let layout = small_layout();
    let parameters: Vec<f32> = (0..9).map(|index| index as f32).collect();
    let head_tuning = layout.head_tuning(&parameters);
    let gate = head_tuning.tensor("residual_gate").unwrap();
    assert_eq!(gate.shape, Vec::<usize>::new());
    assert_eq!(gate.values, vec![8.0]);
    assert_eq!(head_tuning.tensor("q.bias").unwrap().values, vec![6.0, 7.0]);
}

#[test]
fn should_change_only_the_named_slice() {
    let layout = small_layout();
    let mut parameters = vec![0.0; 9];
    layout
        .tensor("q.bias")
        .values_mut(&mut parameters)
        .copy_from_slice(&[1.0, 2.0]);
    assert_eq!(
        parameters,
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0]
    );
}

#[test]
#[should_panic(expected = "the layout has no tensor k.weight")]
fn should_panic_on_a_name_it_never_laid_out() {
    let _ = small_layout().tensor("k.weight");
}

#[test]
fn should_hold_nothing_for_an_empty_layout() {
    let layout = ParameterLayout::new(&[]);
    assert_eq!(layout.parameter_count, 0);
    assert!(layout.head_tuning(&[]).is_empty());
}
