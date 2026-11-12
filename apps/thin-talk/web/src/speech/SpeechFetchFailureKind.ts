/** Why a piece of speech could not be fetched, in the terms the Speak button
 * shows: each kind leads to a different next step for the listener. */
export enum SpeechFetchFailureKind {
  /** No voice model is installed; the listener can get one. */
  NO_VOICE = "noVoice",
  /** Several voice models are installed and none is the default. */
  AMBIGUOUS_VOICE = "ambiguousVoice",
  /** The engine is answering someone else, or restarting; try again soon. */
  BUSY = "busy",
  /** Anything else: the engine could not speak this text. */
  FAILED = "failed",
}
