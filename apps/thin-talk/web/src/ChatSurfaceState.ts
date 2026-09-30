import type { TranscriptSnapshot } from "./TranscriptView";
import { ComposerState } from "./ComposerState";
import { SessionSummary } from "./SessionSummary";

/**
 * The complete surface state one controller refresh publishes: the transcript
 * half, the composer half, the sidebar's session list, and the chrome values.
 * Views render exactly this; they hold no conversation state of their own.
 */
export class ChatSurfaceState {
  public readonly transcript: TranscriptSnapshot;
  public readonly composer: ComposerState;
  public readonly sessions: readonly SessionSummary[];
  public readonly activeSessionID: string | null;
  public readonly composerFontSize: number;

  public constructor(init: {
    transcript: TranscriptSnapshot;
    composer: ComposerState;
    sessions: readonly SessionSummary[];
    activeSessionID: string | null;
    composerFontSize: number;
  }) {
    this.transcript = init.transcript;
    this.composer = init.composer;
    this.sessions = init.sessions;
    this.activeSessionID = init.activeSessionID;
    this.composerFontSize = init.composerFontSize;
  }
}
