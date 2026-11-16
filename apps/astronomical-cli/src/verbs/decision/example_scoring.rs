//! Sending examples to the daemon one at a time and collecting the answers,
//! with a progress line every interval.

use astronomical_decision_contract::{OrderedJsonValue, QuestionOutcome};

use super::decision_tool::{DecisionToolDependencies, DecisionToolError};
use super::progress_reporting::TimedProgress;
use crate::DaemonProbe;
use crate::verbs::decide::{DecisionRequest, request_decisions};

/// One state and its questions, as an example file or the check set holds them.
pub(super) struct ExampleRequest<'example> {
    pub(super) state: &'example OrderedJsonValue,
    pub(super) questions_document: &'example OrderedJsonValue,
}

/// What one batch of examples is scored for, and how its progress reads.
pub(super) struct ScoringPass<'pass> {
    pub(super) decision_model_id: &'pass str,
    pub(super) stage_name: &'static str,
}

/// Every example's answers, in example order.
pub(super) async fn answer_examples(
    example_requests: &[ExampleRequest<'_>],
    scoring_pass: &ScoringPass<'_>,
    daemon_probe: &DaemonProbe,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<Vec<Vec<QuestionOutcome>>, DecisionToolError> {
    let example_count = example_requests.len() as u64;
    let mut timed_progress = TimedProgress::new(
        scoring_pass.stage_name,
        "examples",
        dependencies.progress_interval,
    );
    let mut answered_examples = Vec::with_capacity(example_requests.len());
    for (example_position, example_request) in example_requests.iter().enumerate() {
        timed_progress.report(dependencies.stderr, example_position as u64, example_count);
        let decision_request = DecisionRequest {
            state: example_request.state.clone(),
            questions: example_request.questions_document.clone(),
            decision_model_id: scoring_pass.decision_model_id.to_owned(),
        };
        let decisions_answered = request_decisions(
            decision_request,
            daemon_probe,
            dependencies.wait_bounds,
            &mut *dependencies.stderr,
        )
        .await?;
        answered_examples.push(decisions_answered.question_outcomes);
    }
    timed_progress.report(dependencies.stderr, example_count, example_count);
    Ok(answered_examples)
}
