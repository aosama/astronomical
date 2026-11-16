//! Unit tests for the training journey's own steps. Training itself is
//! tested in the training crate; the whole journey runs live on the
//! Development instance.

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::calibration_file::{
    KindCalibration, calibration_document_text,
};
use astronomical_decision_training::labelled_example::parse_labelled_example;

use super::*;

fn calibration_of(kind: DecisionQuestionKind, temperature: f64) -> CalibrationOverride {
    let mut calibration = CalibrationOverride::default();
    calibration.set_kind_calibration(
        kind,
        KindCalibration {
            temperature,
            range_temperatures: Vec::new(),
            fitted_question_count: 40,
        },
    );
    calibration
}

fn examples_about(state_count: usize, copies: usize) -> Vec<LabelledExample> {
    let mut labelled_examples = Vec::new();
    for state_number in 0..state_count {
        for _ in 0..copies {
            let line_text = format!(
                r#"{{"state": "Romeo line {state_number}", "questions": {{"q": {{"type": "noul"}}}}, "labels": {{"q": true}}}}"#
            );
            labelled_examples.push(parse_labelled_example(&line_text, 1).unwrap());
        }
    }
    labelled_examples
}

#[test]
fn should_keep_an_earlier_calibration_for_kinds_training_did_not_refit() {
    let source_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        source_directory.path().join(DECISION_CALIBRATION_FILE_NAME),
        calibration_document_text(&calibration_of(DecisionQuestionKind::Noul, 2.5)),
    )
    .unwrap();
    let merged = merged_calibration(
        source_directory.path(),
        &calibration_of(DecisionQuestionKind::Choice, 1.5),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        merged.temperature_for(DecisionQuestionKind::Noul, 2),
        Some(2.5)
    );
    assert_eq!(
        merged.temperature_for(DecisionQuestionKind::Choice, 4),
        Some(1.5)
    );
}

#[test]
fn should_write_no_calibration_when_no_kind_has_one() {
    let source_directory = tempfile::tempdir().unwrap();
    let merged =
        merged_calibration(source_directory.path(), &CalibrationOverride::default()).unwrap();
    assert!(merged.is_none());
}

#[test]
fn should_refuse_an_earlier_calibration_it_cannot_read() {
    let source_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        source_directory.path().join(DECISION_CALIBRATION_FILE_NAME),
        b"Parting is such sweet sorrow",
    )
    .unwrap();
    assert!(matches!(
        merged_calibration(source_directory.path(), &CalibrationOverride::default()),
        Err(DecisionToolError::EarlierCalibrationUnreadable(_))
    ));
}

#[test]
fn should_train_only_the_decision_families() {
    assert_eq!(
        tunable_family(ModelFamily::Clef, "clef").unwrap(),
        TunableHeadFamily::Clef
    );
    assert_eq!(
        tunable_family(ModelFamily::Kev, "kev").unwrap(),
        TunableHeadFamily::Kev
    );
    assert!(matches!(
        tunable_family(ModelFamily::Qwen3_5, "qwen"),
        Err(DecisionToolError::NotTrainable { .. })
    ));
}

#[test]
fn should_count_distinct_states_not_lines_when_checking_for_enough_data() {
    let final_layers = TrainingScope::FinalLayers;
    assert!(check_enough_data(&examples_about(30, 7), final_layers).is_ok());
    // 210 questions, but only 10 different states.
    assert!(matches!(
        check_enough_data(&examples_about(10, 21), final_layers),
        Err(DecisionToolError::TooLittleData(_))
    ));
    assert!(matches!(
        check_enough_data(&examples_about(40, 1), final_layers),
        Err(DecisionToolError::TooLittleData(_))
    ));
    // Enough for the final layers is too little for the whole model.
    assert!(matches!(
        check_enough_data(&examples_about(30, 7), TrainingScope::WholeModel),
        Err(DecisionToolError::TooLittleData(_))
    ));
}

fn trained_tensor(name: &str, shape: Vec<usize>, values: Vec<f32>) -> TrainedTensor {
    TrainedTensor {
        name: name.to_owned(),
        shape,
        values: astronomical_decision_contract::FloatVector(values),
    }
}

#[test]
fn should_pair_each_layers_down_and_up_into_one_add_on() {
    let adapter_tensors = vec![
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_down",
            vec![1, 2],
            vec![0.5, -0.5],
        ),
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_up",
            vec![3, 1],
            vec![1.0, 2.0, 3.0],
        ),
    ];

    let Some(decision_adapters) = decision_adapters_of(&adapter_tensors).unwrap() else {
        panic!("two halves of one layer make one add-on");
    };

    assert_eq!(decision_adapters.adapters.len(), 1);
    let adapter = &decision_adapters.adapters[0];
    assert_eq!(adapter.target, "encoder.layers.0.attn.Wo");
    assert_eq!(
        (adapter.rank, adapter.input_width, adapter.output_width),
        (1, 2, 3)
    );
    assert_eq!(adapter.up_values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn should_write_no_add_ons_after_a_final_layer_training() {
    assert!(decision_adapters_of(&[]).unwrap().is_none());
}

#[test]
fn should_refuse_add_ons_that_do_not_pair_up() {
    let other_layers = vec![
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_down",
            vec![1, 2],
            vec![0.5, -0.5],
        ),
        trained_tensor(
            "encoder.layers.1.attn.Wo.adapter_up",
            vec![3, 1],
            vec![1.0, 2.0, 3.0],
        ),
    ];
    let lone_half = vec![trained_tensor(
        "encoder.layers.0.attn.Wo.adapter_down",
        vec![1, 2],
        vec![0.5, -0.5],
    )];
    let different_ranks = vec![
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_down",
            vec![1, 2],
            vec![0.5, -0.5],
        ),
        trained_tensor(
            "encoder.layers.0.attn.Wo.adapter_up",
            vec![1, 2],
            vec![1.0, 2.0],
        ),
    ];

    for unpaired in [other_layers, lone_half, different_ranks] {
        assert!(matches!(
            decision_adapters_of(&unpaired),
            Err(DecisionToolError::TrainingFailed(_))
        ));
    }
}

#[test]
fn should_train_the_final_layers_of_every_decision_family() {
    for family in [
        TunableHeadFamily::Kev,
        TunableHeadFamily::Laya,
        TunableHeadFamily::Clef,
    ] {
        assert!(check_scope_is_trainable(family, TrainingScope::FinalLayers, "model").is_ok());
    }
}

#[test]
fn should_train_laya_as_a_whole() {
    assert!(
        check_scope_is_trainable(TunableHeadFamily::Laya, TrainingScope::WholeModel, "laya")
            .is_ok()
    );
}

#[test]
fn should_refuse_a_whole_model_training_of_kev_or_clef_before_reading_any_example() {
    for family in [TunableHeadFamily::Kev, TunableHeadFamily::Clef] {
        let refusal =
            check_scope_is_trainable(family, TrainingScope::WholeModel, "kev-0.8b").unwrap_err();
        assert!(refusal.is_usage_error());
        assert!(
            refusal.to_string().contains("without --whole-model"),
            "{refusal}"
        );
    }
}
