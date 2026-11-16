//! The two losses head training minimizes, with their gradients with respect
//! to the logits.
//!
//! - **Label-smoothed cross-entropy** on the person's labels: the right
//!   option's target is `1 − ε` and every option shares `ε` evenly. Plain
//!   cross-entropy keeps pushing a right answer's logit up for ever, which on
//!   a few hundred examples is how a head learns to be certain of everything;
//!   Clef's own recipe smooths labels by 0.05.
//! - **Agreement with the shipped model** on the built-in check set: the
//!   Kullback-Leibler divergence from the shipped probabilities to the
//!   trained ones. It keeps questions nobody labelled answering as before.
//!
//! Both take probabilities at the temperature the model serves with, so the
//! loss sees what a person sees.

/// The share of the target spread evenly over every option.
pub const LABEL_SMOOTHING: f64 = 0.05;

/// `softmax(logits / temperature)` in 64-bit.
#[must_use]
pub fn tempered_probabilities(option_logits: &[f32], temperature: f64) -> Vec<f64> {
    let mut largest_scaled_logit = f64::NEG_INFINITY;
    for option_logit in option_logits {
        largest_scaled_logit = largest_scaled_logit.max(f64::from(*option_logit) / temperature);
    }
    let mut probabilities = Vec::with_capacity(option_logits.len());
    let mut total = 0.0_f64;
    for option_logit in option_logits {
        let weight = (f64::from(*option_logit) / temperature - largest_scaled_logit).exp();
        total += weight;
        probabilities.push(weight);
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

/// The label-smoothed cross-entropy of one question and `d loss / d logit`.
#[must_use]
pub fn smoothed_cross_entropy(
    option_logits: &[f32],
    temperature: f64,
    correct_option_index: usize,
) -> (f64, Vec<f32>) {
    let probabilities = tempered_probabilities(option_logits, temperature);
    let option_count = probabilities.len() as f64;
    let mut loss = 0.0_f64;
    let mut logit_gradients = Vec::with_capacity(probabilities.len());
    for (option_index, probability) in probabilities.iter().enumerate() {
        let mut target = LABEL_SMOOTHING / option_count;
        if option_index == correct_option_index {
            target += 1.0 - LABEL_SMOOTHING;
        }
        loss -= target * probability.max(f64::MIN_POSITIVE).ln();
        logit_gradients.push(((probability - target) / temperature) as f32);
    }
    (loss, logit_gradients)
}

/// `KL(shipped ‖ trained)` for one question and `d loss / d logit`.
#[must_use]
pub fn shipped_agreement_loss(
    option_logits: &[f32],
    temperature: f64,
    shipped_probabilities: &[f64],
) -> (f64, Vec<f32>) {
    let probabilities = tempered_probabilities(option_logits, temperature);
    let mut loss = 0.0_f64;
    let mut logit_gradients = Vec::with_capacity(probabilities.len());
    for (probability, shipped_probability) in probabilities.iter().zip(shipped_probabilities) {
        if *shipped_probability > 0.0 {
            loss += shipped_probability
                * (shipped_probability.ln() - probability.max(f64::MIN_POSITIVE).ln());
        }
        logit_gradients.push(((probability - shipped_probability) / temperature) as f32);
    }
    (loss, logit_gradients)
}

/// The unsmoothed surprise at the right answer, `−ln p_correct`: what early
/// stopping and the gate judge by.
#[must_use]
pub fn negative_log_likelihood(
    option_logits: &[f32],
    temperature: f64,
    correct_option_index: usize,
) -> f64 {
    let probabilities = tempered_probabilities(option_logits, temperature);
    -probabilities[correct_option_index]
        .max(f64::MIN_POSITIVE)
        .ln()
}

#[cfg(test)]
#[path = "training_losses_tests.rs"]
mod tests;
