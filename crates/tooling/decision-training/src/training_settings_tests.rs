//! Unit tests for the shared pieces of training.

use super::*;
use decision_contract::decision_training_job::TrainingScope;

#[test]
fn should_shuffle_into_a_reproducible_permutation_that_differs_by_seed() {
    let first = shuffled_order(50, 1, 3);
    let mut sorted = first.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..50).collect::<Vec<usize>>());
    assert_eq!(first, shuffled_order(50, 1, 3));
    assert_ne!(first, shuffled_order(50, 2, 3));
    assert_ne!(first, shuffled_order(50, 1, 4));
    assert!(shuffled_order(0, 1, 1).is_empty());
}

// --- per-family whole-model settings ---

#[test]
fn should_keep_the_shared_whole_model_settings_for_kev_and_clef() {
    assert_eq!(
        whole_model_training_settings(TunableHeadFamily::Kev),
        WHOLE_MODEL_TRAINING_SETTINGS
    );
    assert_eq!(
        whole_model_training_settings(TunableHeadFamily::Clef),
        WHOLE_MODEL_TRAINING_SETTINGS
    );
}

#[test]
fn should_give_laya_its_own_whole_model_settings() {
    assert_eq!(
        whole_model_training_settings(TunableHeadFamily::Laya),
        LAYA_WHOLE_MODEL_TRAINING_SETTINGS
    );
    assert_ne!(
        LAYA_WHOLE_MODEL_TRAINING_SETTINGS,
        WHOLE_MODEL_TRAINING_SETTINGS
    );
}

#[test]
fn should_change_only_the_pace_and_length_of_laya_s_whole_model_run() {
    let laya_settings = LAYA_WHOLE_MODEL_TRAINING_SETTINGS;
    let shared_settings = WHOLE_MODEL_TRAINING_SETTINGS;

    // What Laya changes: a faster, longer, more patient run.
    assert!(laya_settings.adam.learning_rate > shared_settings.adam.learning_rate);
    assert!(laya_settings.maximum_epochs > shared_settings.maximum_epochs);
    assert!(laya_settings.patience_epochs > shared_settings.patience_epochs);

    // What it keeps: how a step is built and what the run is anchored to.
    assert_eq!(
        laya_settings.batch_question_count,
        shared_settings.batch_question_count
    );
    assert_eq!(
        laya_settings.check_questions_per_batch,
        shared_settings.check_questions_per_batch
    );
    assert_eq!(laya_settings.check_weight, shared_settings.check_weight);
    assert_eq!(laya_settings.seeds, shared_settings.seeds);
    assert_eq!(
        laya_settings.adam.anchor_decay,
        shared_settings.adam.anchor_decay
    );
    assert_eq!(
        laya_settings.adam.maximum_gradient_norm,
        shared_settings.adam.maximum_gradient_norm
    );
}

#[test]
fn should_train_every_family_s_final_layers_with_the_final_layer_settings() {
    for family in [
        TunableHeadFamily::Laya,
        TunableHeadFamily::Kev,
        TunableHeadFamily::Clef,
    ] {
        assert_eq!(
            training_settings(TrainingScope::FinalLayers, family),
            FINAL_LAYERS_TRAINING_SETTINGS
        );
    }
}

#[test]
fn should_train_each_family_s_whole_model_with_its_own_whole_model_settings() {
    assert_eq!(
        training_settings(TrainingScope::WholeModel, TunableHeadFamily::Laya),
        LAYA_WHOLE_MODEL_TRAINING_SETTINGS
    );
    assert_eq!(
        training_settings(TrainingScope::WholeModel, TunableHeadFamily::Kev),
        WHOLE_MODEL_TRAINING_SETTINGS
    );
}

#[test]
fn should_keep_laya_s_recipe_at_the_values_its_measurements_chose() {
    // The doc comment's table was measured at exactly these values; an edit
    // to them needs a new measurement, not only a new number.
    let laya_settings = LAYA_WHOLE_MODEL_TRAINING_SETTINGS;
    assert_eq!(laya_settings.adam.learning_rate, 1e-3);
    assert_eq!(laya_settings.maximum_epochs, 30);
    assert_eq!(laya_settings.patience_epochs, 8);
    assert_eq!(
        laya_settings.adam.first_moment_decay,
        WHOLE_MODEL_TRAINING_SETTINGS.adam.first_moment_decay
    );
    assert_eq!(
        laya_settings.adam.second_moment_decay,
        WHOLE_MODEL_TRAINING_SETTINGS.adam.second_moment_decay
    );
    assert_eq!(
        laya_settings.adam.epsilon,
        WHOLE_MODEL_TRAINING_SETTINGS.adam.epsilon
    );
}
