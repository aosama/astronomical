use super::generated_token_will_be_terminal;
use crate::engine_request_test_support::test_engine_request;

const END_OF_SEQUENCE_TOKEN_ID: u32 = 2;
const ORDINARY_TOKEN_ID: u32 = 7;

fn request_with_output_budget(
    generated_token_count: u16,
    maximum_output_tokens: u16,
) -> crate::engine_request::TextEngineRequest<()> {
    let mut active_request = test_engine_request(());
    active_request.generated_token_count = generated_token_count;
    active_request.maximum_output_tokens = maximum_output_tokens;
    active_request
}

// Positive: an end-of-sequence token ends the reply whatever budget is left.
#[test]
fn should_end_the_reply_on_an_end_of_sequence_token() {
    let active_request = request_with_output_budget(0, 100);

    assert!(generated_token_will_be_terminal(
        &[END_OF_SEQUENCE_TOKEN_ID],
        &active_request,
        END_OF_SEQUENCE_TOKEN_ID,
    ));
}

// Edge: the token that uses the last of the output budget ends the reply.
#[test]
fn should_end_the_reply_on_the_token_that_spends_the_last_budget() {
    let active_request = request_with_output_budget(99, 100);

    assert!(generated_token_will_be_terminal(
        &[END_OF_SEQUENCE_TOKEN_ID],
        &active_request,
        ORDINARY_TOKEN_ID,
    ));
}

// Negative: an ordinary token with budget to spare continues the reply.
#[test]
fn should_continue_the_reply_with_budget_to_spare() {
    let active_request = request_with_output_budget(98, 100);

    assert!(!generated_token_will_be_terminal(
        &[END_OF_SEQUENCE_TOKEN_ID],
        &active_request,
        ORDINARY_TOKEN_ID,
    ));
}
