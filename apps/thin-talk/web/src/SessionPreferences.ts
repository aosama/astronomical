import { ThinkingEffort } from "./ThinkingEffort";

/** The sidebar's width range while dragging its edge, in CSS pixels: narrow
 * enough to give the conversation room, wide enough for long titles. */
export const MIN_SIDEBAR_WIDTH_PIXELS = 180;
export const MAX_SIDEBAR_WIDTH_PIXELS = 520;

/**
 * The user's durable chat preferences, persisted through the session bridge so a
 * deliberate choice survives a relaunch while a fresh install still starts on
 * the defaults.
 */
export class SessionPreferences {
  public thinkingEffort: ThinkingEffort;
  /** The chat model the user chose last, or null before any choice. A stored id
   * that no longer appears in the daemon's catalog falls back to the first
   * model at load time. */
  public modelID: string | null;
  /** The image model the user chose last, or null before any choice. A stored
   * id that is no longer installed falls back to the Library's Pictures pick. */
  public imageModelID: string | null;
  /** The Tools switch. On by default: tools are offered only to models that
   * declare tool support, and the tools offered so far read nothing personal
   * and change nothing. */
  public toolsEnabled: boolean;
  /** The width the user dragged the sidebar to, or null for the stylesheet's
   * default width. */
  public sidebarWidthPixels: number | null;

  public constructor(init: {
    thinkingEffort?: ThinkingEffort;
    modelID?: string | null;
    imageModelID?: string | null;
    toolsEnabled?: boolean;
    sidebarWidthPixels?: number | null;
  }) {
    const configuredEffort: ThinkingEffort | undefined = init.thinkingEffort;
    this.thinkingEffort = configuredEffort != null ? configuredEffort : ThinkingEffort.DEFAULT;
    const configuredModelID: string | null | undefined = init.modelID;
    this.modelID = configuredModelID != null ? configuredModelID : null;
    const configuredImageModelID: string | null | undefined = init.imageModelID;
    this.imageModelID = configuredImageModelID != null ? configuredImageModelID : null;
    this.toolsEnabled = init.toolsEnabled !== false;
    const configuredSidebarWidth: number | null | undefined = init.sidebarWidthPixels;
    this.sidebarWidthPixels = configuredSidebarWidth != null ? SessionPreferences.clampSidebarWidth(configuredSidebarWidth) : null;
  }

  public static clampSidebarWidth(widthPixels: number): number {
    return Math.round(Math.max(MIN_SIDEBAR_WIDTH_PIXELS, Math.min(MAX_SIDEBAR_WIDTH_PIXELS, widthPixels)));
  }

  public static default(): SessionPreferences {
    return new SessionPreferences({});
  }

  public toPlain(): {
    thinkingEffort: string;
    modelID: string | null;
    imageModelID: string | null;
    toolsEnabled: boolean;
    sidebarWidthPixels: number | null;
  } {
    return {
      thinkingEffort: this.thinkingEffort,
      modelID: this.modelID,
      imageModelID: this.imageModelID,
      toolsEnabled: this.toolsEnabled,
      sidebarWidthPixels: this.sidebarWidthPixels,
    };
  }

  public static fromPlain(plain: unknown): SessionPreferences {
    if (typeof plain !== "object" || plain === null) {
      return SessionPreferences.default();
    }
    const record = plain as {
      thinkingEffort?: unknown;
      modelID?: unknown;
      imageModelID?: unknown;
      toolsEnabled?: unknown;
      sidebarWidthPixels?: unknown;
    };
    const effort = typeof record.thinkingEffort === "string" ? ThinkingEffort.parse(record.thinkingEffort) : ThinkingEffort.DEFAULT;
    const modelID = typeof record.modelID === "string" ? record.modelID : null;
    const imageModelID = typeof record.imageModelID === "string" ? record.imageModelID : null;
    // Only an explicit false turns tools off, so preferences saved before the
    // switch existed keep the default.
    const toolsEnabled = record.toolsEnabled !== false;
    const storedSidebarWidth = record.sidebarWidthPixels;
    const sidebarWidthPixels = typeof storedSidebarWidth === "number" && Number.isFinite(storedSidebarWidth) ? storedSidebarWidth : null;
    return new SessionPreferences({ thinkingEffort: effort, modelID, imageModelID, toolsEnabled, sidebarWidthPixels });
  }
}
