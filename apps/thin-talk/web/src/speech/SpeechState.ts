import { SpeakPhase } from "./SpeakPhase";

/** Which reply is being spoken and how far along it is. */
export interface SpeechState {
  /** The reply the phase belongs to; every other reply rests. */
  readonly messageID: string | null;
  readonly phase: SpeakPhase;
}

/** Nothing is being spoken. One shared object, so a view can tell "no change"
 * from a new state by identity and skip redrawing every Speak button. */
export const RESTING_SPEECH_STATE: SpeechState = { messageID: null, phase: SpeakPhase.RESTING };
