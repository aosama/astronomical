//! Deterministic decisions answers for daemon lifecycle tests.
//!
//! The scripted worker answers every question with an even spread over its
//! options: no model runs, so the answers say nothing about the state, but
//! the shape, order and lifecycle events are exactly the real worker's.

use astronomical_decision_contract::{
    DecisionFailure, QuestionInputCoverage, QuestionOutcome, answer_from_probabilities,
};
use astronomical_ipc_protocol::{DecisionsCommand, ProtocolError, ProtocolWriter, WorkerEvent};
use tokio::io::AsyncWrite;

/// Tokens the scripted worker reports having read per question.
const SCRIPTED_TOKENS_PER_QUESTION: u32 = 16;

/// Answers one decisions command and finalizes it, as the real worker does.
pub(super) async fn answer_decisions_command<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
    decisions_command: DecisionsCommand,
) -> Result<(), ProtocolError>
where
    WriteTransport: AsyncWrite + Unpin,
{
    let request_id = decisions_command.request_id;
    let terminal_event = match scripted_question_outcomes(&decisions_command) {
        Ok(question_outcomes) => WorkerEvent::DecisionsCompleted {
            request_id,
            input_token_count: SCRIPTED_TOKENS_PER_QUESTION * question_outcomes.len() as u32,
            question_outcomes,
            elapsed_millis: 1,
        },
        Err(reason) => WorkerEvent::DecisionsFailed { request_id, reason },
    };
    event_writer.send_event(&terminal_event).await?;
    event_writer
        .send_event(&WorkerEvent::DecisionsFinalized {
            request_id,
            elapsed_millis: 2,
            mlx_memory_snapshot: None,
        })
        .await
}

fn scripted_question_outcomes(
    decisions_command: &DecisionsCommand,
) -> Result<Vec<QuestionOutcome>, DecisionFailure> {
    let questions = match decisions_command.validated_questions() {
        Ok(questions) => questions,
        Err(validation_error) => {
            return Err(DecisionFailure::invalid_request(
                &validation_error.to_string(),
            ));
        }
    };
    let mut question_outcomes = Vec::with_capacity(questions.len());
    for question in &questions {
        let option_count = question.option_count();
        let even_probabilities = vec![1.0 / option_count as f64; option_count];
        let answer = match answer_from_probabilities(question, &even_probabilities) {
            Ok(answer) => answer,
            Err(answer_error) => {
                return Err(DecisionFailure::fatal_execution(&answer_error.to_string()));
            }
        };
        question_outcomes.push(QuestionOutcome {
            question_name: question.name.clone(),
            answer,
            option_logits: vec![0.0; option_count],
            applied_temperature: 1.0,
            input_coverage: QuestionInputCoverage {
                state_token_count: SCRIPTED_TOKENS_PER_QUESTION,
                state_tokens_read: SCRIPTED_TOKENS_PER_QUESTION,
                option_token_cap: None,
                distinguishable_option_count: option_count as u32,
            },
        });
    }
    Ok(question_outcomes)
}
