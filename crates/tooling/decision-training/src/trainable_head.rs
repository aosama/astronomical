//! What every family's trainable final layers can do on the CPU: score a
//! question from its head inputs, and say how each parameter moves the
//! score.
//!
//! The backward pass is written by hand for each family rather than taken
//! from an automatic-differentiation library: the trainable parts are a few
//! small layers, and every hand-written gradient is pinned by a
//! finite-difference check in the tests, which is the strongest guarantee
//! that training moves the weights the way the loss says.

use decision_contract::HeadInputs;
use thiserror::Error;

use crate::head_parameters::ParameterLayout;

/// One family's trainable part, with its shapes fixed.
pub trait TrainableHead: Sync {
    /// Where each trainable tensor sits in the flat parameter vector.
    fn layout(&self) -> &ParameterLayout;

    /// Checks that `head_inputs` are this family's and this head's width,
    /// and returns the option count.
    fn check_inputs(&self, head_inputs: &HeadInputs) -> Result<usize, HeadInputsMismatch>;

    /// One logit per option, before any temperature. `head_inputs` must have
    /// passed [`Self::check_inputs`]; other inputs score as no options.
    fn option_logits(&self, parameters: &[f32], head_inputs: &HeadInputs) -> Vec<f32>;

    /// Adds `d loss / d parameter` to `gradient` for every parameter, given
    /// `d loss / d logit` for every option.
    fn add_gradient(
        &self,
        parameters: &[f32],
        head_inputs: &HeadInputs,
        logit_gradients: &[f32],
        gradient: &mut [f32],
    );
}

/// Why head inputs do not fit a trainable head.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum HeadInputsMismatch {
    #[error("the inputs are {found}'s, not {expected}'s")]
    WrongFamily {
        expected: &'static str,
        found: &'static str,
    },
    #[error("an input vector is {found} wide where the head reads {expected}")]
    WrongWidth { expected: usize, found: usize },
    #[error("the per-option inputs disagree about how many options there are")]
    OptionCountDisagrees,
    #[error("a question needs at least one option")]
    NoOptions,
}

/// The family a set of head inputs belongs to, for messages.
#[must_use]
pub const fn input_family_name(head_inputs: &HeadInputs) -> &'static str {
    match head_inputs {
        HeadInputs::Kev { .. } => "Kev",
        HeadInputs::Laya { .. } => "Laya",
        HeadInputs::Clef { .. } => "Clef",
    }
}
