//! Unit tests for the pipelined step source generator. The generated Metal
//! bodies are additionally compiled and executed end to end by the GPU kernel
//! tests; these pin the fragment assembly contracts themselves so a broken
//! template renders no further than source generation.

use super::*;

fn unrolled_sources(
    token_index: usize,
    next_staged_token: Option<usize>,
) -> PipeliningStageSources {
    pipelining_stage_sources(StepPipelining::Unrolled {
        token_index,
        next_staged_token,
    })
}

#[test]
fn should_embed_the_literal_token_scatter_slot_and_refill_for_a_mid_block_unrolled_step() {
    let stage_sources = unrolled_sources(5, Some(6));

    assert_eq!(stage_sources.token_expression, "5");
    // Token 5 sits at position 1 of its four-step scatter group.
    assert!(
        stage_sources
            .partial_publication_source
            .contains("step_query_partials[1]")
    );
    // The refill stages token 6's operands unconditionally: inside a full
    // block there is no bounds guard to emit.
    assert!(stage_sources.next_refill_source.contains("staged_keys[6]"));
    assert!(
        stage_sources
            .next_refill_source
            .contains("staged_decays[6]")
    );
    assert!(
        stage_sources
            .next_refill_source
            .contains("staged_values[6][value_row_in_block]")
    );
    assert!(!stage_sources.next_refill_source.contains("if ("));
}

#[test]
fn should_omit_the_refill_for_the_last_unrolled_step_with_no_next_staged_token() {
    let stage_sources = unrolled_sources(TIME_BLOCK_SIZE - 1, None);

    assert_eq!(stage_sources.token_expression, "11");
    assert!(stage_sources.next_refill_source.is_empty());
    // Token 11 lands in the last scatter slot.
    assert!(
        stage_sources
            .partial_publication_source
            .contains("step_query_partials[3]")
    );
}

#[test]
fn should_use_the_rolled_loop_variable_guarded_refill_and_carry_publication_for_the_tail_block() {
    let stage_sources = pipelining_stage_sources(StepPipelining::Rolled);

    assert_eq!(stage_sources.token_expression, "token_in_block");
    assert!(
        stage_sources
            .next_refill_source
            .contains("if (token_in_block + 1 < tokens_in_block)")
    );
    assert!(
        stage_sources
            .partial_publication_source
            .contains("previous_step_query_partial =")
    );
}

#[test]
fn should_emit_no_reduction_when_no_earlier_partials_are_ready() {
    let reduction_source =
        previous_step_reduction_source(PreviousStepReduction::NoPreviousSteps, "0");

    assert!(reduction_source.is_empty());
}

#[test]
fn should_guard_the_carry_reduction_on_a_positive_token_expression() {
    let reduction_source =
        previous_step_reduction_source(PreviousStepReduction::CarryPreviousStep, "7");

    assert!(reduction_source.contains("if (7 > 0)"));
    assert!(reduction_source.contains("staged_outputs[7 - 1]"));
    assert!(reduction_source.contains("simd_shuffle_down"));
}

#[test]
fn should_scatter_reduce_four_steps_from_the_first_reduced_token() {
    let reduction_source = previous_step_reduction_source(
        PreviousStepReduction::ScatterFourSteps {
            first_reduced_token: TIME_BLOCK_SIZE - 4,
        },
        "token_in_block",
    );

    // The scatter reads the four-slot partial array and writes rows starting
    // at the first reduced token; the step's own token expression is unused.
    assert!(reduction_source.contains("step_query_partials["));
    assert!(reduction_source.contains("staged_outputs[8 +"));
    assert!(!reduction_source.contains("token_in_block"));
}

fn marker_fragments() -> StepBodyFragments {
    StepBodyFragments {
        token_expression: "9".to_owned(),
        next_refill_source: "NEXT_REFILL;\n".to_owned(),
        reduction_source: "REDUCTION;\n".to_owned(),
        checkpoint_write_source: "CHECKPOINT_WRITE;\n".to_owned(),
        partial_publication_source: "PARTIAL_PUBLICATION;\n".to_owned(),
    }
}

#[test]
fn should_interpolate_every_fragment_into_its_slot_in_the_step_body() {
    let step_body = render_step_body(marker_fragments());

    assert!(step_body.contains("float4 current_keys[4];"));
    assert!(step_body.contains("state_fragment[fragment_index] *= current_decay;"));
    assert!(step_body.contains("NEXT_REFILL;"));
    assert!(step_body.contains("REDUCTION;"));
    assert!(step_body.contains("CHECKPOINT_WRITE;"));
    assert!(step_body.contains("PARTIAL_PUBLICATION;"));
    assert!(step_body.contains("staged_queries[9]"));
}

#[test]
fn should_place_the_fragments_in_pipeline_order() {
    let step_body = render_step_body(marker_fragments());

    let refill_position = step_body.find("NEXT_REFILL;").expect("refill present");
    let reduction_position = step_body.find("REDUCTION;").expect("reduction present");
    let checkpoint_position = step_body
        .find("CHECKPOINT_WRITE;")
        .expect("checkpoint write present");
    let partial_position = step_body
        .find("PARTIAL_PUBLICATION;")
        .expect("partial publication present");

    // The refill precedes the key-dot loop, the reduction lands between the
    // key-dot loop and the remembered-value reduction, and the checkpoint
    // write and partial publication close the step body.
    assert!(refill_position < reduction_position);
    assert!(reduction_position < checkpoint_position);
    assert!(checkpoint_position < partial_position);
}

#[test]
fn should_keep_balanced_braces_with_empty_fragments() {
    let step_body = render_step_body(StepBodyFragments {
        token_expression: String::new(),
        next_refill_source: String::new(),
        reduction_source: String::new(),
        checkpoint_write_source: String::new(),
        partial_publication_source: String::new(),
    });

    // Even with every fragment empty, the template's own Metal skeleton must
    // survive intact: balanced braces and no unfilled format holes.
    assert_eq!(
        step_body.matches('{').count(),
        step_body.matches('}').count()
    );
    assert!(!step_body.contains("{next_refill_source}"));
    assert!(!step_body.contains("{token_expression}"));
}

#[test]
fn should_inject_the_checkpoint_write_at_each_step_token_expression() {
    let checkpoint_write_source_template =
        format!("record({});\n", CHECKPOINT_TOKEN_EXPRESSION_MARKER);

    let unrolled_step_body = pipelined_step_source(
        StepPipelining::Unrolled {
            token_index: 2,
            next_staged_token: Some(3),
        },
        PreviousStepReduction::ScatterFourSteps {
            first_reduced_token: 0,
        },
        &checkpoint_write_source_template,
    );
    assert!(unrolled_step_body.contains("record(2);"));
    // No unreplaced marker may survive rendering.
    assert!(!unrolled_step_body.contains(CHECKPOINT_TOKEN_EXPRESSION_MARKER));

    let rolled_step_body = pipelined_step_source(
        StepPipelining::Rolled,
        PreviousStepReduction::CarryPreviousStep,
        &checkpoint_write_source_template,
    );
    assert!(rolled_step_body.contains("record(token_in_block);"));
}
