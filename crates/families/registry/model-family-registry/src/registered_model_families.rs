//! Every model family's description, in the order a folder is classified
//! (issue #2473, Step 5).
//!
//! One explicit list, rather than registration at link time, so a reader
//! sees every family in one place and a family left out fails the registry
//! tests instead of vanishing at run time. Within each way of recognizing a
//! folder (pipeline index, marker file, `model_type`) the first family that
//! claims it wins, so the list's order is the classification order. Each
//! family declares its description in its own recognition crate, in the
//! family's folder, and the worker reads its builder from the same line.

use model_discovery::ModelFamilyDescription;

/// Reads a line of `model_families.list` as the family's description; the
/// builder half of the line is the worker's and is never compiled here.
macro_rules! model_family {
    ($family_description:ident, $builder_kind:ident ( $($runtime_builder:tt)* )) => {
        crate::$family_description
    };
}

/// One line per family in `model_families.list`; each description lives in
/// the family's own recognition crate.
pub(crate) static REGISTERED_MODEL_FAMILIES: &[ModelFamilyDescription] =
    &include!("model_families.list");

#[cfg(test)]
#[path = "registered_model_families_tests.rs"]
mod registered_model_families_tests;
