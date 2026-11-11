import type { ExportFormat } from "./ExportFormat";
import type { ExportScope } from "./ExportOptions";

/** One finished message as the export feature receives it: the rendered
 * answer body, cloned out of the transcript, and the stored names of the
 * pictures it generated. The feature never reads the transcript itself. */
export interface ExportableMessage {
  readonly messageID: string;
  readonly isQuestion: boolean;
  /** A detached copy of the message's rendered body; the feature may change it. */
  readonly renderedBody: HTMLElement;
  readonly generatedImageNames: readonly string[];
  /** The alternative text for each generated picture, by stored name. */
  readonly generatedImageDescriptions: ReadonlyMap<string, string>;
}

/** A stored picture's bytes, as the host returns them. */
export interface ExportPictureData {
  readonly base64: string;
  readonly mimeType: string;
}

/** The finished file handed to the host, which asks where to save it. A PDF
 * travels as the page it is printed from; the host owns the printing. */
export interface ExportSaveRequest {
  readonly format: ExportFormat;
  /** The suggested file name without its extension. */
  readonly suggestedName: string;
  /** The standalone page, for a web page or a PDF. */
  readonly html: string | null;
  /** The file's bytes in base64, for a Word document. */
  readonly base64: string | null;
}

/** How a save ended. The page never learns where the file went. */
export type ExportSaveOutcome =
  | { readonly outcome: "saved"; readonly fileName: string }
  | { readonly outcome: "cancelled" };

/**
 * Everything the export feature needs from the rest of the page. It is an
 * interface because the feature is a separate script, loaded the first time
 * someone exports (it carries the Word library, which is nearly as large as
 * the chat itself): the two scripts share no objects, only this contract.
 */
export interface ExportEnvironment {
  /** The empty dialog element the sheet draws itself in. */
  readonly sheetElement: HTMLElement;
  conversationTitle(): string;
  /** "Astro" while Astro is shown; null for the plain chat. */
  assistantName(): string | null;
  exportableMessages(): readonly ExportableMessage[];
  loadPicture(imageName: string): Promise<ExportPictureData | null>;
  /** Draws a diagram's source again for a white page, as SVG markup; null
   * when it cannot, and the drawing on screen is used as it is. */
  drawDiagramForPaper(diagramSource: string): Promise<string | null>;
  saveDocument(request: ExportSaveRequest): Promise<ExportSaveOutcome>;
  /** Hides the sheet's host and lets the person tick messages; resolves with
   * the ticked ids, or null when they backed out. */
  pickMessages(alreadyPickedIDs: readonly string[]): Promise<readonly string[] | null>;
  /** Tells the person the file was saved. */
  announceSaved(fileName: string): void;
}

/** What the lazily loaded script registers on the window. */
export interface ExportFeature {
  open(environment: ExportEnvironment, scope: ExportScope): void;
}
