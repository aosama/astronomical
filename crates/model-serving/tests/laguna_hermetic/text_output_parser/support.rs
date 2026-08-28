use astronomical_model_serving::{LagunaOutputParser, LagunaOutputParserError};

use super::super::text_support::{SyntheticLagunaTextArtifact, declared_literary_tools};

pub(super) fn literary_output_parser() -> LagunaOutputParser {
    literary_output_parser_starting_in_reasoning(false)
}

pub(super) fn literary_output_parser_starting_in_reasoning(
    generation_starts_in_reasoning: bool,
) -> LagunaOutputParser {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    LagunaOutputParser::new(
        &text_descriptor,
        &declared_literary_tools(),
        generation_starts_in_reasoning,
    )
    .expect("the poolside_v1 descriptor and declared tools should construct a parser")
}

pub(super) fn assert_bounded_error(parser_error: &LagunaOutputParserError) {
    // Public malformed-output diagnostics must remain safe even when generated content is huge.
    assert!(parser_error.to_string().len() <= 256);
}
