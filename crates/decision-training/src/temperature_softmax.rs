//! Log-probabilities of option logits at a temperature, in `f64`.
//!
//! Fitting and the metrics need `log p` of the right option. Taking the log
//! of a probability that underflowed to zero gives infinity, so the
//! log-softmax is computed directly with the log-sum-exp form instead.

/// `log softmax(option_logits / temperature)`, one value per option.
///
/// `temperature` is used as given; callers confine it to the range the
/// serving engine would apply before calling, so these values match what a
/// person would actually be served.
#[must_use]
pub fn log_probabilities_at(option_logits: &[f32], temperature: f64) -> Vec<f64> {
    let mut scaled_logits = Vec::with_capacity(option_logits.len());
    let mut largest_scaled_logit = f64::NEG_INFINITY;
    for option_logit in option_logits {
        let scaled_logit = f64::from(*option_logit) / temperature;
        largest_scaled_logit = largest_scaled_logit.max(scaled_logit);
        scaled_logits.push(scaled_logit);
    }
    let mut shifted_exponential_total = 0.0_f64;
    for scaled_logit in &scaled_logits {
        shifted_exponential_total += (scaled_logit - largest_scaled_logit).exp();
    }
    let log_normalizer = largest_scaled_logit + shifted_exponential_total.ln();
    let mut log_probabilities = Vec::with_capacity(scaled_logits.len());
    for scaled_logit in scaled_logits {
        log_probabilities.push(scaled_logit - log_normalizer);
    }
    log_probabilities
}

/// `softmax(option_logits / temperature)`, one probability per option.
#[must_use]
pub fn probabilities_at(option_logits: &[f32], temperature: f64) -> Vec<f64> {
    let mut probabilities = log_probabilities_at(option_logits, temperature);
    for probability in &mut probabilities {
        *probability = probability.exp();
    }
    probabilities
}

/// The index of the largest value; the first one when several tie, as the
/// contract picks a chosen option.
#[must_use]
pub fn first_largest_index(option_values: &[f64]) -> usize {
    let mut largest_index = 0;
    for (option_index, option_value) in option_values.iter().enumerate() {
        if *option_value > option_values[largest_index] {
            largest_index = option_index;
        }
    }
    largest_index
}

#[cfg(test)]
#[path = "temperature_softmax_tests.rs"]
mod tests;
