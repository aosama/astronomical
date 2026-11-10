//! Temperature calibration: turning a model's option logits into probabilities.
//!
//! Decision models ship one or more fitted temperatures. Dividing the logits
//! by the right temperature before the softmax is what makes "0.9" mean the
//! answer is right about nine times in ten, so the temperature lookup and the
//! softmax live here, shared by every model family.

use crate::question::DecisionQuestionKind;

/// The temperature that leaves logits unchanged.
pub const NEUTRAL_TEMPERATURE: f64 = 1.0;

/// The lowest temperature that is applied as fitted.
///
/// A temperature below 1 sharpens the distribution. One published checkpoint
/// ships 0.1 for questions with eleven or more options, which would report a
/// 0.24 top probability as 0.99. The reference implementation refuses to
/// sharpen harder than 2x for that reason, and this matches it.
pub const MINIMUM_APPLIED_TEMPERATURE: f64 = 0.5;

/// The highest temperature that is applied as fitted.
pub const MAXIMUM_APPLIED_TEMPERATURE: f64 = 5.0;

/// A usable temperature: `raw_temperature` confined to the applied range.
///
/// A value that is not a finite number falls back to the neutral temperature,
/// because a corrupt calibration entry must not turn every probability into
/// `NaN`.
#[must_use]
pub fn clamp_temperature(raw_temperature: f64) -> f64 {
    if !raw_temperature.is_finite() {
        return NEUTRAL_TEMPERATURE;
    }
    raw_temperature.clamp(MINIMUM_APPLIED_TEMPERATURE, MAXIMUM_APPLIED_TEMPERATURE)
}

/// The option-count range a per-count temperature was fitted for.
///
/// A two-option question and a twenty-option question need different scaling,
/// so checkpoints fit one temperature per range; these are the ranges the
/// published checkpoints use.
#[must_use]
pub const fn option_count_range_name(option_count: usize) -> &'static str {
    match option_count {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    }
}

/// The key a checkpoint stores a per-option-count temperature under, such as `choice:3-5`.
#[must_use]
pub fn temperature_bucket_key(kind: DecisionQuestionKind, option_count: usize) -> String {
    format!(
        "{}:{}",
        kind.wire_name(),
        option_count_range_name(option_count)
    )
}

/// One fitted temperature per question kind, as a checkpoint ships them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KindTemperatures {
    pub choice: f64,
    pub score: f64,
    pub noul: f64,
}

/// One fitted temperature for a question kind and option-count range.
#[derive(Clone, Debug, PartialEq)]
pub struct BucketTemperature {
    /// A key in the form [`temperature_bucket_key`] produces.
    pub bucket_key: String,
    pub temperature: f64,
}

/// Every temperature a checkpoint ships, already confined to the applied range.
#[derive(Clone, Debug, PartialEq)]
pub struct TemperatureTable {
    kind_temperatures: KindTemperatures,
    bucket_temperatures: Vec<BucketTemperature>,
}

impl TemperatureTable {
    /// Builds a table from a checkpoint's raw values, clamping each one.
    #[must_use]
    pub fn from_raw_temperatures(
        raw_kind_temperatures: KindTemperatures,
        raw_bucket_temperatures: &[BucketTemperature],
    ) -> Self {
        let mut bucket_temperatures = Vec::with_capacity(raw_bucket_temperatures.len());
        for raw_bucket_temperature in raw_bucket_temperatures {
            bucket_temperatures.push(BucketTemperature {
                bucket_key: raw_bucket_temperature.bucket_key.clone(),
                temperature: clamp_temperature(raw_bucket_temperature.temperature),
            });
        }
        Self {
            kind_temperatures: KindTemperatures {
                choice: clamp_temperature(raw_kind_temperatures.choice),
                score: clamp_temperature(raw_kind_temperatures.score),
                noul: clamp_temperature(raw_kind_temperatures.noul),
            },
            bucket_temperatures,
        }
    }

    /// A table that applies one temperature to every question.
    #[must_use]
    pub fn uniform(raw_temperature: f64) -> Self {
        Self::from_raw_temperatures(
            KindTemperatures {
                choice: raw_temperature,
                score: raw_temperature,
                noul: raw_temperature,
            },
            &[],
        )
    }

    /// The temperature for one question: its option-count range's when the
    /// checkpoint fitted one, otherwise its kind's.
    #[must_use]
    pub fn temperature_for(&self, kind: DecisionQuestionKind, option_count: usize) -> f64 {
        let wanted_bucket_key = temperature_bucket_key(kind, option_count);
        for bucket_temperature in &self.bucket_temperatures {
            if bucket_temperature.bucket_key == wanted_bucket_key {
                return bucket_temperature.temperature;
            }
        }
        match kind {
            DecisionQuestionKind::Choice => self.kind_temperatures.choice,
            DecisionQuestionKind::Score => self.kind_temperatures.score,
            DecisionQuestionKind::Noul => self.kind_temperatures.noul,
        }
    }
}

/// Softmax of `option_logits / temperature`, one probability per option.
///
/// The largest scaled logit is subtracted before exponentiating so a large
/// logit cannot overflow to infinity; the result is unchanged because the
/// subtraction cancels in the normalization. The arithmetic runs in `f64`
/// so the sum of many small probabilities does not drift.
///
/// `temperature` is confined to the applied range here as well as in
/// [`TemperatureTable`], so a caller that passes a raw value can never divide
/// by zero and return `NaN` for every option.
#[must_use]
pub fn calibrated_probabilities(option_logits: &[f32], temperature: f64) -> Vec<f64> {
    if option_logits.is_empty() {
        return Vec::new();
    }
    let temperature = clamp_temperature(temperature);
    let mut scaled_logits = Vec::with_capacity(option_logits.len());
    let mut largest_scaled_logit = f64::NEG_INFINITY;
    for option_logit in option_logits {
        let scaled_logit = f64::from(*option_logit) / temperature;
        largest_scaled_logit = largest_scaled_logit.max(scaled_logit);
        scaled_logits.push(scaled_logit);
    }
    let mut unnormalized_weights = Vec::with_capacity(scaled_logits.len());
    let mut weight_total = 0.0_f64;
    for scaled_logit in scaled_logits {
        let unnormalized_weight = (scaled_logit - largest_scaled_logit).exp();
        weight_total += unnormalized_weight;
        unnormalized_weights.push(unnormalized_weight);
    }
    let mut probabilities = Vec::with_capacity(unnormalized_weights.len());
    for unnormalized_weight in unnormalized_weights {
        probabilities.push(unnormalized_weight / weight_total);
    }
    probabilities
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;
