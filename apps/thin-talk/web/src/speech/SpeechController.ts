import { speakableTextOf } from "./SpeakableText";
import { speechPiecesOf } from "./SpeechPieces";
import { SpeakPhase } from "./SpeakPhase";
import { RESTING_SPEECH_STATE } from "./SpeechState";
import type { SpeechState } from "./SpeechState";
import { SpeechFetchFailure } from "./SpeechFetchFailure";
import { SpeechFetchFailureKind } from "./SpeechFetchFailureKind";
import type { SpeechPieceRequest } from "./SpeechClient";
import type { SpeechAudioOutput } from "./SpeechAudioOutput";

/** How long a passing note ("Busy, try again") stays on the button. */
const PASSING_NOTE_MILLISECONDS = 2500;

export interface SpeechControllerDependencies {
  output: SpeechAudioOutput;
  fetchSpokenPiece: (pieceRequest: SpeechPieceRequest) => Promise<ArrayBuffer>;
  /** The installed voice models' ids, for when the daemon cannot pick one. */
  listVoiceModels: () => Promise<string[]>;
  openModelsPage: () => void;
}

/** What the controller needs to know about the conversation on screen. */
export interface ConversationSnapshot {
  /** An answer is being written, so the engine is busy and its model is
   * about to be needed. */
  isGenerating: boolean;
  activeSessionID: string | null;
  shownMessageIDs: ReadonlySet<string>;
}

/**
 * Reads one reply aloud at a time and says, through its state, what each
 * Speak button shows.
 *
 * Every piece of the reply is fetched back to back as soon as Speak is
 * pressed, and played from memory. Fetching ahead matters beyond smoothness:
 * the engine holds one model at a time, and once the last piece is fetched it
 * is free to load the chat model again (it does so when the next question
 * arrives), even while the listener is still hearing the reply. Sending a new
 * question still ends the reading: the reply scrolls away with its Stop
 * button, and an answer cannot be heard over a voice.
 */
export class SpeechController {
  private readonly dependencies: SpeechControllerDependencies;
  private readonly listeners: Array<(state: SpeechState) => void> = [];
  private currentState: SpeechState = RESTING_SPEECH_STATE;
  private fetchAbortController: AbortController | null = null;
  private passingNoteTimer: ReturnType<typeof setTimeout> | null = null;
  private namedVoiceModel: string | null = null;
  private latestSessionID: string | null = null;

  public constructor(dependencies: SpeechControllerDependencies) {
    this.dependencies = dependencies;
  }

  public get state(): SpeechState {
    return this.currentState;
  }

  public subscribe(listener: (state: SpeechState) => void): void {
    this.listeners.push(listener);
  }

  /** A press of one reply's Speak button. Must run inside the click, so the
   * audio output may start. */
  public press(messageID: string, markdown: string): void {
    const isThisReply = this.currentState.messageID === messageID;
    if (isThisReply && (this.currentState.phase === SpeakPhase.PREPARING || this.currentState.phase === SpeakPhase.PLAYING)) {
      this.stop();
      return;
    }
    if (isThisReply && this.currentState.phase === SpeakPhase.NO_VOICE) {
      this.publish(RESTING_SPEECH_STATE);
      this.dependencies.openModelsPage();
      return;
    }
    this.start(messageID, markdown);
  }

  /** Stops any reading at once: a send, a stop, or the reply going away. */
  public stop(): void {
    this.fetchAbortController?.abort();
    this.fetchAbortController = null;
    this.dependencies.output.silence();
    this.publish(RESTING_SPEECH_STATE);
  }

  /**
   * Ends the reading when the conversation moves on from it: an answer starts
   * being written (a send, a retry or a skill all arrive this way), another
   * chat opens, or the reply being read is no longer shown (deleted, or its
   * chat closed). Called with every published conversation state, which for a
   * streaming answer is every token, so it does nothing unless something is
   * being read.
   */
  public followConversation(conversation: ConversationSnapshot): void {
    const hasSessionChanged = conversation.activeSessionID !== this.latestSessionID;
    this.latestSessionID = conversation.activeSessionID;
    const speakingMessageID = this.currentState.messageID;
    if (speakingMessageID === null) {
      return;
    }
    if (conversation.isGenerating || hasSessionChanged || !conversation.shownMessageIDs.has(speakingMessageID)) {
      this.stop();
    }
  }

  private start(messageID: string, markdown: string): void {
    const pieces = speechPiecesOf(speakableTextOf(markdown));
    if (pieces.length === 0) {
      return;
    }
    this.fetchAbortController?.abort();
    const fetchAbortController = new AbortController();
    this.fetchAbortController = fetchAbortController;
    try {
      this.dependencies.output.wake({
        onFirstSound: () => this.publishIfCurrent(fetchAbortController, { messageID, phase: SpeakPhase.PLAYING }),
        onFinished: () => this.publishIfCurrent(fetchAbortController, RESTING_SPEECH_STATE),
        onFailed: () => this.endWithNote(messageID, fetchAbortController, SpeakPhase.FAILED),
      });
    } catch {
      // No audio context could be made: say so rather than doing nothing.
      this.endWithNote(messageID, fetchAbortController, SpeakPhase.FAILED);
      return;
    }
    this.publish({ messageID, phase: SpeakPhase.PREPARING });
    // One seed for the whole reply keeps one voice from piece to piece.
    const seed = Math.floor(Math.random() * 2 ** 31);
    void this.fetchEveryPiece(messageID, pieces, seed, fetchAbortController);
  }

  private async fetchEveryPiece(
    messageID: string,
    pieces: readonly string[],
    seed: number,
    fetchAbortController: AbortController,
  ): Promise<void> {
    let enqueuedPieceCount = 0;
    for (const piece of pieces) {
      let pieceBytes: ArrayBuffer;
      try {
        pieceBytes = await this.fetchWithVoiceChoice(piece, seed, fetchAbortController.signal);
      } catch (fetchError) {
        if (!fetchAbortController.signal.aborted) {
          this.endAfterFailure(messageID, fetchError, fetchAbortController, enqueuedPieceCount > 0);
        }
        return;
      }
      if (fetchAbortController.signal.aborted) {
        return;
      }
      this.dependencies.output.enqueue(pieceBytes);
      enqueuedPieceCount += 1;
    }
    if (this.fetchAbortController === fetchAbortController) {
      this.fetchAbortController = null;
    }
    this.dependencies.output.markComplete();
  }

  /** Lets the daemon pick the voice model; when several are installed and none
   * is the default, names one and remembers it for later presses. A named
   * model that then fails is forgotten, so one uninstalled since is not tried
   * for the rest of the session. */
  private async fetchWithVoiceChoice(text: string, seed: number, signal: AbortSignal): Promise<ArrayBuffer> {
    try {
      return await this.dependencies.fetchSpokenPiece({ text, seed, model: this.namedVoiceModel, signal });
    } catch (fetchError) {
      const wasNamedBefore = this.namedVoiceModel !== null;
      const needsNamedModel = fetchError instanceof SpeechFetchFailure && fetchError.kind === SpeechFetchFailureKind.AMBIGUOUS_VOICE;
      if (!needsNamedModel || wasNamedBefore) {
        if (wasNamedBefore && !signal.aborted) {
          this.namedVoiceModel = null;
        }
        throw fetchError;
      }
      this.namedVoiceModel = SpeechController.preferredVoiceModel(await this.dependencies.listVoiceModels());
      if (this.namedVoiceModel === null) {
        throw fetchError;
      }
      return this.dependencies.fetchSpokenPiece({ text, seed, model: this.namedVoiceModel, signal });
    }
  }

  /** Kokoro first, the built-in default and the voice most people know. */
  private static preferredVoiceModel(voiceModelIDs: readonly string[]): string | null {
    const kokoroModelID = voiceModelIDs.find((voiceModelID) => voiceModelID.toLowerCase().includes("kokoro"));
    return kokoroModelID ?? voiceModelIDs[0] ?? null;
  }

  /** A failure after some audio was fetched just ends the reply early: what
   * was fetched still plays. With nothing to play, the button says what went wrong. */
  private endAfterFailure(
    messageID: string,
    fetchError: unknown,
    fetchAbortController: AbortController,
    hasFetchedAudio: boolean,
  ): void {
    if (hasFetchedAudio) {
      this.dependencies.output.markComplete();
      return;
    }
    const failureKind = fetchError instanceof SpeechFetchFailure ? fetchError.kind : SpeechFetchFailureKind.FAILED;
    if (failureKind === SpeechFetchFailureKind.NO_VOICE) {
      this.endWithNote(messageID, fetchAbortController, SpeakPhase.NO_VOICE);
      return;
    }
    const notePhase = failureKind === SpeechFetchFailureKind.BUSY ? SpeakPhase.BUSY : SpeakPhase.FAILED;
    this.endWithNote(messageID, fetchAbortController, notePhase);
  }

  /** Ends the reading and leaves a note on the reply's button: a lasting one
   * for "get a voice", a passing one for the rest. */
  private endWithNote(messageID: string, fetchAbortController: AbortController, notePhase: SpeakPhase): void {
    if (this.fetchAbortController === fetchAbortController) {
      this.fetchAbortController = null;
    }
    fetchAbortController.abort();
    this.dependencies.output.silence();
    this.publish({ messageID, phase: notePhase });
    if (notePhase === SpeakPhase.NO_VOICE) {
      return;
    }
    this.passingNoteTimer = setTimeout(() => {
      if (this.currentState.messageID === messageID && this.currentState.phase === notePhase) {
        this.publish(RESTING_SPEECH_STATE);
      }
    }, PASSING_NOTE_MILLISECONDS);
  }

  private publishIfCurrent(fetchAbortController: AbortController, nextState: SpeechState): void {
    const isStillCurrent =
      this.fetchAbortController === fetchAbortController || (this.fetchAbortController === null && !fetchAbortController.signal.aborted);
    if (isStillCurrent) {
      this.publish(nextState);
    }
  }

  private publish(nextState: SpeechState): void {
    if (this.passingNoteTimer !== null) {
      clearTimeout(this.passingNoteTimer);
      this.passingNoteTimer = null;
    }
    this.currentState = nextState;
    for (const listener of this.listeners) {
      listener(nextState);
    }
  }
}
