import type { ChatController } from "./ChatController";
import type { ChatSurfaceState } from "./ChatSurfaceState";
import { exportableMessages } from "./ExportSource";
import type { MessagePicker } from "./MessagePicker";
import type { BriefNotice } from "./BriefNotice";
import type { SessionBridge } from "./SessionBridge";
import type { TranscriptView } from "./TranscriptView";
import { ASTRO_NAME } from "./astro/AstroPresentation";
import type { ExportEnvironment, ExportFeature } from "./export/ExportEnvironment";
import type { ExportScope } from "./export/ExportOptions";

/** Said when there is nothing finished to export yet, instead of a menu
 * item or toolbar button that silently does nothing. */
export const NOTHING_TO_EXPORT_NOTICE = "There's nothing to export yet. Ask something first.";

/** Served by the host from the same private scheme as the page's own script,
 * which the page's content policy already allows. */
const EXPORT_SCRIPT_SOURCE = "thintalk-asset://shell/export.js";

export interface ExportLauncherTargets {
  readonly sheetElement: HTMLElement;
  readonly controller: ChatController;
  readonly transcript: TranscriptView;
  readonly bridge: SessionBridge;
  readonly picker: MessagePicker;
  readonly notice: BriefNotice;
  /** Loads the export script; replaced in tests, where it is bundled in. */
  readonly loadExportFeature?: () => Promise<ExportFeature>;
}

/** Adds the export script to the page once and waits for it to register. */
function loadExportScript(): Promise<ExportFeature> {
  const registeredFeature = window.__thintalkExport;
  if (registeredFeature !== undefined) {
    return Promise.resolve(registeredFeature);
  }
  return new Promise((resolve, reject) => {
    const exportScript = document.createElement("script");
    exportScript.src = EXPORT_SCRIPT_SOURCE;
    exportScript.onload = () => (window.__thintalkExport !== undefined ? resolve(window.__thintalkExport) : reject(new Error("export.js did not register")));
    exportScript.onerror = () => reject(new Error("export.js could not be loaded"));
    document.head.appendChild(exportScript);
  });
}

/**
 * Opens the export sheet from any of its three ways in (an answer's Export
 * button, the toolbar and File menu, a conversation's right-click menu), and
 * gives the separately loaded export script what it needs from the page.
 */
export class ExportLauncher {
  private readonly targets: ExportLauncherTargets;
  private latestState: ChatSurfaceState | null = null;
  private pendingFeature: Promise<ExportFeature> | null = null;

  public constructor(targets: ExportLauncherTargets) {
    this.targets = targets;
    targets.controller.subscribe((publishedState) => {
      this.latestState = publishedState;
    });
  }

  public get hasExportableMessages(): boolean {
    // A conversation opened a moment ago may not be drawn yet.
    this.targets.transcript.flushPendingRender();
    return exportableMessages(this.targets.transcript).length > 0;
  }

  /** Opens the sheet on the chat on screen. An empty chat gets a short
   * notice instead of a sheet with nothing in it. */
  public async open(scope: ExportScope): Promise<void> {
    if (!this.hasExportableMessages) {
      this.targets.notice.show(NOTHING_TO_EXPORT_NOTICE);
      return;
    }
    this.pendingFeature ??= (this.targets.loadExportFeature ?? loadExportScript)().catch((error: unknown) => {
      // A failed load is retried on the next open rather than remembered.
      this.pendingFeature = null;
      throw error;
    });
    const exportFeature = await this.pendingFeature;
    exportFeature.open(this.environment(), scope);
  }

  private conversationTitle(): string {
    const activeSessionID = this.latestState?.activeSessionID ?? null;
    return this.latestState?.sessions.find((session) => session.id === activeSessionID)?.title ?? "";
  }

  private environment(): ExportEnvironment {
    const { transcript, bridge, picker, notice } = this.targets;
    return {
      sheetElement: this.targets.sheetElement,
      conversationTitle: () => this.conversationTitle(),
      assistantName: () => (transcript.persona.isAstroShown ? ASTRO_NAME : null),
      exportableMessages: () => exportableMessages(transcript),
      loadPicture: (imageName) => bridge.loadGeneratedImage(imageName),
      saveDocument: (saveRequest) => bridge.saveExport(saveRequest),
      pickMessages: (alreadyPickedIDs) =>
        picker.pick(
          exportableMessages(transcript).map((message) => message.messageID),
          alreadyPickedIDs,
        ),
      announceSaved: (fileName) => notice.show(`Saved “${fileName}”`),
    };
  }
}
