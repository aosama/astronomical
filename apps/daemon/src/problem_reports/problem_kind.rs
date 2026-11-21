//! The kinds of problem a report can describe, each with the plain words the
//! person reads on screen and the stable name the developers group by.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProblemKind {
    /// A model could not be loaded because the Mac lacked the memory.
    ModelLoadNotEnoughMemory,
    /// A model could not be loaded for another reason.
    ModelLoadFailed,
    /// A reply stopped because the graphics chip ran out of memory.
    ReplyOutOfMemory,
    /// A catalog model download stopped before it finished.
    DownloadFailed,
    /// The engine's worker process ended without being asked to.
    EngineStoppedUnexpectedly,
    /// The engine did not start.
    EngineDidNotStart,
}

impl ProblemKind {
    /// The words shown after "Problem:" in the report the person reads.
    pub(crate) const fn plain_description(self) -> &'static str {
        match self {
            Self::ModelLoadNotEnoughMemory => "model load failed, not enough memory",
            Self::ModelLoadFailed => "model load failed",
            Self::ReplyOutOfMemory => "reply stopped, graphics memory ran out",
            Self::DownloadFailed => "model download stopped before finishing",
            Self::EngineStoppedUnexpectedly => "engine stopped unexpectedly",
            Self::EngineDidNotStart => "engine did not start",
        }
    }
}
