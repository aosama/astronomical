//! Unit tests for judging a trained head, on scores built for each case.

use super::*;
use crate::test_support::scored_question;
use crate::training_losses::tempered_probabilities;

const QUESTIONS_PER_PART: usize = 60;

/// A three-option question the served model barely separates, its right
/// answer rotating over the options.
fn served_question(state_key: u64) -> ScoredQuestion {
    let correct_option = (state_key % 3) as usize;
    let mut served = scored_question(
        DecisionQuestionKind::Choice,
        &[0.2, 0.1, 0.0],
        correct_option,
        state_key,
    );
    // Served scores lean to option 0 whatever the right answer is.
    served.option_logits = vec![0.2, 0.1, 0.0];
    served
}

/// Trained scores: `+lift` on `favoured` option, the served scores elsewhere.
fn lifted(served: &ScoredQuestion, favoured: usize, lift: f32) -> Vec<f32> {
    let mut trained_logits = served.option_logits.clone();
    trained_logits[favoured] += lift;
    trained_logits
}

/// Pairs whose trained scores favour the right answer (`toward_right`), the
/// wrong one after it, or nothing.
fn scored_pairs(first_state: u64, trained_lift: Option<(bool, f32)>) -> Vec<ScoredPair> {
    let mut pairs = Vec::with_capacity(QUESTIONS_PER_PART);
    for state_offset in 0..QUESTIONS_PER_PART as u64 {
        let served = served_question(first_state + state_offset);
        let trained_logits = match trained_lift {
            Some((toward_right, lift)) => {
                let right = served.correct_option_index;
                let favoured = if toward_right { right } else { (right + 1) % 3 };
                lifted(&served, favoured, lift)
            }
            None => served.option_logits.clone(),
        };
        pairs.push(ScoredPair {
            served,
            trained_logits,
        });
    }
    pairs
}

fn agreeing_check(question_count: usize) -> Vec<CheckPair> {
    let mut check_pairs = Vec::new();
    for _ in 0..question_count {
        check_pairs.push(CheckPair {
            served_probabilities: tempered_probabilities(&[1.0, 0.0], 1.0),
            trained_logits: vec![2.0, 0.0],
        });
    }
    check_pairs
}

fn scores(trained_lift: Option<(bool, f32)>, check: Vec<CheckPair>) -> TrainingScores {
    TrainingScores {
        choosing: scored_pairs(0, trained_lift),
        judging: scored_pairs(1_000, trained_lift),
        check,
        judged_on: JudgedOn::HeldBackQuestions,
    }
}

fn has_concern(verdict: &TrainingVerdict, wanted: fn(&TrainingConcern) -> bool) -> bool {
    verdict.concerns.iter().any(wanted)
}

#[test]
fn should_judge_better_a_head_that_is_better_on_questions_nothing_else_read() {
    let verdict = judge_training(&scores(Some((true, 2.0)), agreeing_check(20))).unwrap();
    assert_eq!(
        verdict.recorded_verdict(),
        RecordedVerdict::JudgedBetter,
        "{:?}",
        verdict.concerns
    );
    assert!(verdict.trained_overall.accuracy() > verdict.shipped_overall.accuracy());
    assert_eq!(verdict.check_agreement.agreeing_count, 20);
}

#[test]
fn should_judge_worse_a_head_that_did_not_change_even_when_calibration_alone_helps() {
    let verdict = judge_training(&scores(None, Vec::new())).unwrap();
    assert_eq!(verdict.recorded_verdict(), RecordedVerdict::JudgedWorse);
    assert!(has_concern(&verdict, |concern| matches!(
        concern,
        TrainingConcern::NotLessSurprised { .. }
    )));
}

#[test]
fn should_judge_worse_a_head_that_got_worse_and_name_why() {
    let verdict = judge_training(&scores(Some((false, 2.0)), Vec::new())).unwrap();
    assert_eq!(verdict.recorded_verdict(), RecordedVerdict::JudgedWorse);
    assert!(has_concern(&verdict, |concern| matches!(
        concern,
        TrainingConcern::LessAccurate { .. }
    )));
}

#[test]
fn should_name_a_changed_check_set_as_a_concern() {
    let mut check = agreeing_check(20);
    for check_pair in check.iter_mut().take(2) {
        check_pair.trained_logits = vec![0.0, 2.0];
    }
    let verdict = judge_training(&scores(Some((true, 2.0)), check)).unwrap();
    assert_eq!(verdict.check_agreement.agreeing_count, 18);
    assert!(has_concern(&verdict, |concern| matches!(
        concern,
        TrainingConcern::CheckAnswersChanged(_)
    )));
}

#[test]
fn should_record_not_judged_when_compared_on_its_training_examples() {
    let mut training_scores = scores(Some((true, 2.0)), Vec::new());
    training_scores.judged_on = JudgedOn::LearnedFromExamples;
    let verdict = judge_training(&training_scores).unwrap();
    assert_eq!(verdict.recorded_verdict(), RecordedVerdict::NotJudged);
}

#[test]
fn should_judge_a_single_question_compared_on_its_own_example() {
    let single = vec![ScoredPair {
        served: served_question(1),
        trained_logits: lifted(&served_question(1), 1, 3.0),
    }];
    let verdict = judge_training(&TrainingScores {
        choosing: single.clone(),
        judging: single,
        check: Vec::new(),
        judged_on: JudgedOn::LearnedFromExamples,
    })
    .unwrap();
    assert_eq!(verdict.trained_overall.question_count, 1);
    assert_eq!(verdict.recorded_verdict(), RecordedVerdict::NotJudged);
}

#[test]
fn should_see_a_collapse_only_when_the_labels_vary() {
    let mut always_first = Vec::new();
    for state_key in 0..20 {
        always_first.push(scored_question(
            DecisionQuestionKind::Noul,
            &[2.0, 0.0],
            (state_key % 2) as usize,
            state_key,
        ));
    }
    assert!(has_collapsed(&always_first, DecisionQuestionKind::Noul));
    for question in &mut always_first {
        question.correct_option_index = 0;
    }
    assert!(!has_collapsed(&always_first, DecisionQuestionKind::Noul));
    assert!(!has_collapsed(&always_first, DecisionQuestionKind::Choice));
}
