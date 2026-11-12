/** What one reply's Speak button says and does. */
export enum SpeakPhase {
  /** "Speak": nothing is happening for this reply. */
  RESTING = "resting",
  /** "Preparing…": the first piece is on its way; a click cancels. */
  PREPARING = "preparing",
  /** "Stop": the reply is being read aloud; a click stops it. */
  PLAYING = "playing",
  /** "Get a voice…": no voice model is installed; a click opens Models. */
  NO_VOICE = "noVoice",
  /** "Busy, try again": the engine is answering something else. */
  BUSY = "busy",
  /** "Couldn't speak": the engine or the audio output could not read this reply. */
  FAILED = "failed",
}
