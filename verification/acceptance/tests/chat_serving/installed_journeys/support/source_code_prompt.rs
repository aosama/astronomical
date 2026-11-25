//! A long prompt whose reply is code, beside the Romeo and Juliet summary
//! whose reply is an essay: drafting pays differently on the two.

use super::romeo_and_juliet::InstructedSource;

/// Real Python source, cut where the prompt is long enough, for the model
/// to continue; pass it to `prepare_instructed_source_prompt`.
pub(crate) const SOURCE_CODE_CONTINUATION: InstructedSource = InstructedSource {
    instruction: "Continue this Python source from exactly where it stops. Write only code, with no explanation before or after it.\n\n",
    source_text: include_str!(
        "../../../../../../verification/acceptance/tests/fixtures/model_metrics_progressive_source_code.txt"
    ),
};
