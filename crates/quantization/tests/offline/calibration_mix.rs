//! The bundled mix loads end to end with a fake tokenizer.

use astronomical_quantization::calibration::{CalibrationMix, CalibrationTokenizer};
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::progress::{ProgressReporter, StageProgress};

struct WhitespaceTokenizer;

impl CalibrationTokenizer for WhitespaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError> {
        Ok(text
            .split_whitespace()
            .map(|word| word.len() as u32)
            .collect())
    }
}

#[derive(Default)]
struct CountingReporter {
    report_count: usize,
}

impl ProgressReporter for CountingReporter {
    fn report(&mut self, progress: &StageProgress) {
        let _ = progress;
        self.report_count += 1;
    }
}

#[test]
#[ignore = "pending implementation of CalibrationMix::bundled"]
fn should_load_the_bundled_mix_into_the_requested_number_of_sequences() {
    let mut reporter = CountingReporter::default();

    let mix = CalibrationMix::bundled(&WhitespaceTokenizer, 24, 512, 42, &mut reporter)
        .expect("the bundled mix is long enough for 24 sequences of 512 words");

    assert_eq!(mix.samples().len(), 24);
    assert!(
        mix.samples()
            .iter()
            .all(|sample| sample.token_ids.len() == 512)
    );
    assert_eq!(mix.summary().token_count_by_domain.len(), 6);
    assert!(
        reporter.report_count >= 1,
        "loading must report progress at least once"
    );
}
