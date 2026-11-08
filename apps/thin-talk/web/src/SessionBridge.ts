import type { ConversationMenuChoice } from "./ConversationMenuChoice";
import type { ImageForModel } from "./ConversationTool";
import { SessionBridgeOp } from "./SessionBridgeOp";
import { SessionBridgeRequest } from "./SessionBridgeRequest";
import { SessionBridgeError, SessionBridgeTimeoutError } from "./SessionBridgeError";
import { SessionDocument } from "./SessionDocument";
import { SessionSummary } from "./SessionSummary";
import { SessionPreferences } from "./SessionPreferences";
import { parseConversationSkill } from "./ConversationSkill";
import { isGeneratedImageName } from "./ToolStep";

interface PendingSessionCall {
  resolve: (payload: unknown) => void;
  reject: (error: Error) => void;
  timer: number;
}

/** How long a session call may run before the page gives up on the host. */
const CALL_TIMEOUT_MILLISECONDS = 10_000;
/** Storing a generated image carries megabytes of base64 across the bridge
 * and through a decode and a file write, so it gets more time than a session
 * file: giving up early would report a failed image the host still saves. */
const STORE_IMAGE_TIMEOUT_MILLISECONDS = 30_000;
/** Calls that wait for the user (an approval sheet, an open panel) can take
 * as long as the user takes; this only stops a call the host never answers
 * from waiting forever. */
const USER_CHOICE_TIMEOUT_MILLISECONDS = 10 * 60_000;

/** How one use of a host tool ended (see HostToolOutcome in Swift).
 * "unavailable" means the user was never asked; "refused" means they said no. */
export type HostToolUse =
  | { readonly outcome: "approved" }
  | { readonly outcome: "chosenText"; readonly fileName: string; readonly text: string }
  | { readonly outcome: "chosenImage"; readonly fileName: string; readonly base64: string; readonly mimeType: "image/png" | "image/jpeg" }
  | { readonly outcome: "refused" | "nothingChosen" | "cancelled" }
  | { readonly outcome: "unavailable"; readonly reason: string };

/**
 * The typed client side of the session file bridge. Every durable read and write
 * goes through here as a promise: the page posts a `sessionCall` to the host and
 * the host answers by invoking `window.__thintalkBridge.resolve/reject` with the
 * call id. Installing that resolver is this class's contract with the host, so
 * it happens at construction.
 */
export class SessionBridge {
  private readonly post: (request: SessionBridgeRequest) => void;
  private readonly pendingCalls = new Map<string, PendingSessionCall>();
  private nextCallNumber = 0;
  private readonly pageInstanceTag = Math.random().toString(36).slice(2, 10);

  public constructor(post: (request: SessionBridgeRequest) => void) {
    this.post = post;
    window.__thintalkBridge = {
      resolve: (callId: string, payload: unknown) => this.resolveCall(callId, payload),
      reject: (callId: string, message: string) => this.rejectCall(callId, message),
    };
  }

  public list(): Promise<SessionSummary[]> {
    return this.call(SessionBridgeOp.LIST, null).then((payload) =>
      SessionBridge.summariesFromPlain(payload),
    );
  }

  /** Loads one session, or null when the host has no file under that id. */
  public load(id: string): Promise<SessionDocument | null> {
    return this.call(SessionBridgeOp.LOAD, { id }).then((payload) =>
      payload === null ? null : SessionDocument.fromPlain(payload),
    );
  }

  public save(document: SessionDocument): Promise<void> {
    return this.call(SessionBridgeOp.SAVE, document.toPlain()).then(() => undefined);
  }

  public delete(id: string): Promise<void> {
    return this.call(SessionBridgeOp.DELETE, { id }).then(() => undefined);
  }

  public rename(id: string, title: string): Promise<void> {
    return this.call(SessionBridgeOp.RENAME, { id, title }).then(() => undefined);
  }

  public loadPrefs(): Promise<SessionPreferences> {
    return this.call(SessionBridgeOp.LOAD_PREFS, null).then((payload) =>
      SessionPreferences.fromPlain(payload),
    );
  }

  public savePrefs(preferences: SessionPreferences): Promise<void> {
    return this.call(SessionBridgeOp.SAVE_PREFS, preferences.toPlain()).then(() => undefined);
  }

  /** Astro's saved profile as stored, or null before the first save; the
   * astro module reads it, so the bridge stays free of Astro's rules. */
  public loadAstroProfile(): Promise<unknown> {
    return this.call(SessionBridgeOp.LOAD_ASTRO_PROFILE, null);
  }

  public saveAstroProfile(plainProfile: Record<string, unknown>): Promise<void> {
    return this.call(SessionBridgeOp.SAVE_ASTRO_PROFILE, plainProfile).then(() => undefined);
  }

  /** Hands a generated image (base64 PNG or JPEG) to the host to keep, and
   * returns the stored name the conversation records and loads it by. */
  public storeGeneratedImage(base64Image: string): Promise<string> {
    return this.call(SessionBridgeOp.STORE_GENERATED_IMAGE, { base64: base64Image }, STORE_IMAGE_TIMEOUT_MILLISECONDS).then(
      (payload) => {
        const imageName = typeof payload === "object" && payload !== null ? (payload as Record<string, unknown>)["imageName"] : null;
        // A name the transcript would refuse to show must fail the tool here,
        // or the model is told the image was shown when it was not.
        if (typeof imageName !== "string" || !isGeneratedImageName(imageName)) {
          throw new Error("the host did not return a usable stored image name");
        }
        return imageName;
      },
    );
  }

  /** One use of a host tool: the approval sheet and, for the file tools, the
   * open panel, run by the host as a single call so the page cannot reach a
   * panel without the sheet. Stopping the turn cancels it on the host too,
   * which closes the sheet or panel. */
  public async useHostTool(toolName: string, conversationID: string, signal: AbortSignal): Promise<HostToolUse> {
    const hostAnswer = await this.cancellableCall(
      SessionBridgeOp.USE_HOST_TOOL,
      { toolName, conversationID },
      USER_CHOICE_TIMEOUT_MILLISECONDS,
      signal,
    );
    return SessionBridge.hostToolUseFromPlain(hostAnswer);
  }

  private static hostToolUseFromPlain(hostAnswer: unknown): HostToolUse {
    const record = typeof hostAnswer === "object" && hostAnswer !== null ? (hostAnswer as Record<string, unknown>) : {};
    const fileName = record["fileName"];
    switch (record["outcome"]) {
      case "approved":
      case "refused":
      case "nothingChosen":
      case "cancelled":
        return { outcome: record["outcome"] };
      case "unavailable":
        return { outcome: "unavailable", reason: typeof record["reason"] === "string" ? record["reason"] : "unknown" };
      case "chosen":
        if (typeof fileName === "string" && typeof record["text"] === "string") {
          return { outcome: "chosenText", fileName, text: record["text"] };
        }
        if (typeof fileName === "string" && typeof record["base64"] === "string" && (record["mimeType"] === "image/png" || record["mimeType"] === "image/jpeg")) {
          return { outcome: "chosenImage", fileName, base64: record["base64"], mimeType: record["mimeType"] };
        }
        throw new Error("the host returned an unreadable file");
      default:
        throw new Error("the host returned an unknown tool outcome");
    }
  }

  /** Shows the native right-click menu for one sidebar conversation and
   * returns what the user picked; a delete has already been confirmed. The
   * menu waits for a person, so it gets the user-choice timeout. */
  public async showConversationMenu(conversationID: string): Promise<ConversationMenuChoice> {
    const hostAnswer = await this.call(SessionBridgeOp.SHOW_CONVERSATION_MENU, { conversationID }, USER_CHOICE_TIMEOUT_MILLISECONDS);
    const menuChoice = typeof hostAnswer === "object" && hostAnswer !== null ? (hostAnswer as Record<string, unknown>)["choice"] : null;
    return menuChoice === "rename" || menuChoice === "delete" ? menuChoice : "dismissed";
  }

  /** Reads a generated image back, so a model that can see is shown it again
   * on a later turn. Null when the image is gone or its name is not one the
   * app stores. */
  public async loadGeneratedImage(imageName: string): Promise<ImageForModel | null> {
    if (!isGeneratedImageName(imageName)) {
      return null;
    }
    const hostAnswer = await this.call(SessionBridgeOp.LOAD_GENERATED_IMAGE, { imageName }, STORE_IMAGE_TIMEOUT_MILLISECONDS);
    const imageRecord = typeof hostAnswer === "object" && hostAnswer !== null ? (hostAnswer as Record<string, unknown>) : {};
    const base64 = imageRecord["base64"];
    const mimeType = imageRecord["mimeType"];
    return typeof base64 === "string" && typeof mimeType === "string" ? { base64, mimeType } : null;
  }

  /** Deletes generated images by their stored names, as when the
   * conversation that showed them is deleted. */
  public deleteGeneratedImages(imageNames: readonly string[]): Promise<void> {
    return this.call(SessionBridgeOp.DELETE_GENERATED_IMAGES, { imageNames: [...imageNames] }).then(() => undefined);
  }

  /** A call that the turn's stop signal can end: it rejects at once on the
   * page and asks the host to cancel the same call. */
  private cancellableCall(op: SessionBridgeOp, payload: unknown, timeoutMilliseconds: number, signal: AbortSignal): Promise<unknown> {
    if (signal.aborted) {
      return Promise.reject(new Error(`${op} was stopped`));
    }
    const callId = this.nextCallID();
    const answer = this.call(op, payload, timeoutMilliseconds, callId);
    return new Promise<unknown>((resolve, reject) => {
      const cancelOnHost = (): void => {
        this.settlePendingCall(callId);
        void this.call(SessionBridgeOp.CANCEL_HOST_TOOL_CALL, { callId }).catch(() => undefined);
        reject(new Error(`${op} was stopped`));
      };
      signal.addEventListener("abort", cancelOnHost, { once: true });
      answer.then(resolve, reject).finally(() => signal.removeEventListener("abort", cancelOnHost));
    });
  }

  /** Call ids carry a value unique to this page, so a late answer meant for a
   * page that has since reloaded cannot settle one of this page's calls. */
  private nextCallID(): string {
    this.nextCallNumber += 1;
    return `call-${this.pageInstanceTag}-${this.nextCallNumber}`;
  }

  private settlePendingCall(callId: string): void {
    const pendingCall = this.pendingCalls.get(callId);
    if (pendingCall != null) {
      window.clearTimeout(pendingCall.timer);
      this.pendingCalls.delete(callId);
    }
  }

  private call(
    op: SessionBridgeOp,
    payload: unknown,
    timeoutMilliseconds: number = CALL_TIMEOUT_MILLISECONDS,
    callId: string = this.nextCallID(),
  ): Promise<unknown> {
    return new Promise<unknown>((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this.pendingCalls.delete(callId);
        reject(new SessionBridgeTimeoutError(op, timeoutMilliseconds));
      }, timeoutMilliseconds);
      this.pendingCalls.set(callId, { resolve, reject, timer });
      this.post(new SessionBridgeRequest({ callId, op, payload }));
    });
  }

  private resolveCall(callId: string, payload: unknown): void {
    const pendingCall = this.pendingCalls.get(callId);
    if (!pendingCall) {
      return;
    }
    this.pendingCalls.delete(callId);
    window.clearTimeout(pendingCall.timer);
    pendingCall.resolve(payload);
  }

  private rejectCall(callId: string, message: string): void {
    const pendingCall = this.pendingCalls.get(callId);
    if (!pendingCall) {
      return;
    }
    this.pendingCalls.delete(callId);
    window.clearTimeout(pendingCall.timer);
    pendingCall.reject(new SessionBridgeError(message));
  }

  private static summariesFromPlain(payload: unknown): SessionSummary[] {
    if (!Array.isArray(payload)) {
      throw new SessionBridgeError("the session list reply is not an array");
    }
    return payload.map((rawSummary) => {
      if (typeof rawSummary !== "object" || rawSummary === null) {
        throw new SessionBridgeError("a session list entry is not an object");
      }
      const record = rawSummary as Record<string, unknown>;
      if (typeof record["id"] !== "string" || typeof record["updatedAt"] !== "string") {
        throw new SessionBridgeError("a session list entry is missing its id or timestamp");
      }
      const title = typeof record["title"] === "string" ? record["title"] : "";
      return new SessionSummary({ id: record["id"], title, updatedAt: record["updatedAt"], skill: parseConversationSkill(record["skill"]) });
    });
  }
}
