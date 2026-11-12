import { SpeechFetchFailureKind } from "./SpeechFetchFailureKind";

/** A piece of speech that could not be fetched, classified for the button. */
export class SpeechFetchFailure extends Error {
  public readonly kind: SpeechFetchFailureKind;

  public constructor(kind: SpeechFetchFailureKind, detail: string) {
    super(detail);
    this.kind = kind;
  }
}
