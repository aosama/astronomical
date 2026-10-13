//! The shared fidelity switch for the root acceptance journeys.

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
    /// Reads the journey switch: `ASTRONOMICAL_REAL_MODEL=1` selects the
    /// real daemon and artifacts; anything else runs the scripted pair.
    #[must_use]
    pub fn from_environment() -> Self {
        if std::env::var("ASTRONOMICAL_REAL_MODEL").as_deref() == Ok("1") {
            Self::Real
        } else {
            Self::Scripted
        }
    }
}
