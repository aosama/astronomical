use astronomical_model_serving::LagunaOutputParser;

use super::super::text_support;
use super::super::text_support::SyntheticLagunaTextArtifact;

pub(super) fn literary_output_parser() -> LagunaOutputParser {
    literary_output_parser_starting_in_reasoning(false)
}

pub(super) fn literary_output_parser_starting_in_reasoning(
    generation_starts_in_reasoning: bool,
) -> LagunaOutputParser {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    LagunaOutputParser::new(
        &text_descriptor,
        &text_support::declared_literary_tools(),
        generation_starts_in_reasoning,
    )
    .expect("the poolside_v1 descriptor and declared tools should construct a parser")
}
