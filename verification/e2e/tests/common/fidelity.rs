//! Explicit fidelity selection for the whole-product model journeys.

use astronomical_test_harness::test_run::{TestRunConfiguration, TestRunContext};

/// Which daemon and worker pair a journey drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFidelity {
    /// The fixture `astronomical-daemon-test-daemon` plus the scripted
    /// `astronomical-daemon-test-worker`: every assertion runs, no model
    /// artifacts are needed, so the suite stays runnable anywhere.
    Scripted,
    /// The real `astronomicald` spawning the real inference worker against
    /// the installed artifacts: the original acceptance journeys.
    Real,
}

impl ModelFidelity {
    #[must_use]
    pub fn current() -> Self {
        Self::from_configuration(&TestRunContext::current().configuration)
    }

    fn from_configuration(configuration: &TestRunConfiguration) -> Self {
        if configuration.real_models {
            Self::Real
        } else {
            Self::Scripted
        }
    }
}

#[cfg(test)]
mod fidelity_tests {
    use super::{ModelFidelity, TestRunConfiguration};

    #[test]
    fn should_default_to_scripted_models_without_a_real_model_request() {
        assert_eq!(
            ModelFidelity::from_configuration(&TestRunConfiguration::default()),
            ModelFidelity::Scripted
        );
    }

    #[test]
    fn should_select_real_models_only_when_the_run_configuration_requests_them() {
        assert_eq!(
            ModelFidelity::from_configuration(&TestRunConfiguration {
                real_models: true,
                ..TestRunConfiguration::default()
            }),
            ModelFidelity::Real
        );
    }
}
