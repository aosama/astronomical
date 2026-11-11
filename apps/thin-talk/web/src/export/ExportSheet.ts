import { exportPreviewMarkup, PRINTED_MARGIN_INCHES } from "./ExportArticle";
import { ExportContentBuilder, messagesInScope } from "./ExportContentBuilder";
import type { ExportContent } from "./ExportContentBuilder";
import type { ExportEnvironment } from "./ExportEnvironment";
import { ExportFormat, exportFormatOffer } from "./ExportFormat";
import { defaultExportOptions } from "./ExportOptions";
import type { ExportOptions, ExportScope } from "./ExportOptions";
import { ExportSheetView } from "./ExportSheetView";
import type { ExportScopeKind, ExportSheetFacts, ExportToggle } from "./ExportSheetView";
import { exportSaveRequest } from "./ExportWriter";
import { rasteriseDiagramOnCanvas } from "./DiagramRasteriser";
import type { DiagramRasteriser } from "./DiagramRasteriser";

const CSS_PIXELS_PER_INCH = 96;
/** The height of text on a printed Letter page: eleven inches less the two
 * margins. A4 is a little taller, so the estimate errs towards one page more,
 * which is why the sheet says "about". */
const PRINTED_TEXT_HEIGHT_PIXELS = (11 - 2 * PRINTED_MARGIN_INCHES) * CSS_PIXELS_PER_INCH;
const PRINTED_MARGINS_HEIGHT_PIXELS = 2 * PRINTED_MARGIN_INCHES * CSS_PIXELS_PER_INCH;
/** The preview page's full width: a Letter sheet, 8.5 inches. */
const PREVIEW_PAGE_WIDTH_PIXELS = 8.5 * CSS_PIXELS_PER_INCH;

/**
 * The text height of the previewed document at full size, from the size its
 * page was laid out at. The page is known to be 8.5 inches wide, so its
 * width gives the scale, which keeps the measure right however the preview
 * is scaled or however the browser reports a scaled size.
 */
export function previewTextHeightPixels(drawnWidthPixels: number, drawnHeightPixels: number): number {
  if (drawnWidthPixels <= 0) {
    return 0;
  }
  const fullSizeHeightPixels = drawnHeightPixels * (PREVIEW_PAGE_WIDTH_PIXELS / drawnWidthPixels);
  return Math.max(0, fullSizeHeightPixels - PRINTED_MARGINS_HEIGHT_PIXELS);
}

const SAVE_FAILED_STATUS = "That file couldn't be saved. Try again.";
const NOTHING_TO_EXPORT_STATUS = "There's nothing to export with these choices yet.";

/** "About 3 pages", or what a web page is instead. */
export function pageCountText(format: ExportFormat, measuredHeightPixels: number): string {
  if (format === ExportFormat.WEB_PAGE) {
    return "One page that scrolls";
  }
  const estimatedPages = Math.max(1, Math.ceil(measuredHeightPixels / PRINTED_TEXT_HEIGHT_PIXELS));
  return estimatedPages === 1 ? "1 page" : `About ${estimatedPages} pages`;
}

/**
 * Runs one export: the options the person changes, a preview rebuilt from
 * the real document after every change, and the save.
 */
export class ExportSheet {
  private readonly environment: ExportEnvironment;
  private readonly view: ExportSheetView;
  private readonly rasteriseDiagram: DiagramRasteriser;
  private builder: ExportContentBuilder;
  private options: ExportOptions = defaultExportOptions({ kind: "wholeChat" });
  private openedAnswerID: string | null = null;
  private pickedMessageIDs: readonly string[] = [];
  private latestContent: ExportContent | null = null;
  /** Rebuilding is asynchronous (pictures are fetched), so only the newest
   * rebuild may draw; an older one finishing late would show stale choices. */
  private previewGeneration = 0;

  public constructor(environment: ExportEnvironment, rasteriseDiagram: DiagramRasteriser = rasteriseDiagramOnCanvas) {
    this.environment = environment;
    this.rasteriseDiagram = rasteriseDiagram;
    this.builder = this.freshBuilder();
    this.view = new ExportSheetView(environment.sheetElement, {
      chooseScope: (scopeKind) => void this.chooseScope(scopeKind),
      chooseFormat: (format) => this.update({ ...this.options, format }),
      toggle: (toggle, isOn) => this.toggle(toggle, isOn),
      save: () => void this.save(),
      cancel: () => this.view.hide(),
    });
  }

  private freshBuilder(): ExportContentBuilder {
    const isDarkAppearance = typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches;
    return new ExportContentBuilder(this.environment, isDarkAppearance);
  }

  /** Opens the sheet on a fresh reading of the conversation, starting from
   * the choices of the last export that was saved. */
  public open(scope: ExportScope): void {
    this.builder = this.freshBuilder();
    this.openedAnswerID = scope.kind === "oneAnswer" ? scope.messageID : null;
    this.pickedMessageIDs = scope.kind === "pickedMessages" ? scope.messageIDs : [];
    this.view.setStatus("");
    this.update(defaultExportOptions(scope, this.environment.rememberedChoices()));
    this.view.show();
  }

  private facts(): ExportSheetFacts {
    const scopedMessages = messagesInScope(this.environment.exportableMessages(), this.options.scope);
    return {
      canExportOneAnswer: this.openedAnswerID !== null,
      hasQuestions: scopedMessages.some((message) => message.isQuestion),
      hasPictures: scopedMessages.some((message) => message.generatedImageNames.length > 0),
      pickedMessageCount: this.pickedMessageIDs.length,
    };
  }

  private update(options: ExportOptions): void {
    this.options = options;
    this.view.render(options, this.facts());
    void this.refreshPreview();
  }

  private toggle(toggle: ExportToggle, isOn: boolean): void {
    this.update({ ...this.options, [toggle]: isOn });
  }

  private async chooseScope(scopeKind: ExportScopeKind): Promise<void> {
    if (scopeKind === "wholeChat") {
      this.update({ ...this.options, scope: { kind: "wholeChat" }, includesQuestions: true });
      return;
    }
    if (scopeKind === "oneAnswer" && this.openedAnswerID !== null) {
      this.update({ ...this.options, scope: { kind: "oneAnswer", messageID: this.openedAnswerID }, includesQuestions: false });
      return;
    }
    if (scopeKind === "pickedMessages") {
      await this.pickMessages();
    }
  }

  /** The sheet steps aside while the person ticks messages in the
   * conversation itself, and comes back with their choice. Backing out, or
   * ticking nothing, returns to the sheet as it was. */
  private async pickMessages(): Promise<void> {
    this.view.hide();
    const pickedMessageIDs = await this.environment.pickMessages(this.pickedMessageIDs);
    if (pickedMessageIDs !== null && pickedMessageIDs.length > 0) {
      this.pickedMessageIDs = pickedMessageIDs;
      this.update({ ...this.options, scope: { kind: "pickedMessages", messageIDs: pickedMessageIDs }, includesQuestions: true });
    }
    this.view.show();
  }

  private async refreshPreview(): Promise<void> {
    this.previewGeneration += 1;
    const generation = this.previewGeneration;
    const content = await this.builder.build(this.options);
    if (generation !== this.previewGeneration) {
      return;
    }
    this.latestContent = content;
    this.view.showPreview(exportPreviewMarkup(content), "");
    this.view.setCanSave(content.entries.length > 0);
    this.view.setStatus(content.entries.length > 0 ? "" : NOTHING_TO_EXPORT_STATUS);
    await this.view.previewPicturesLoaded();
    if (generation === this.previewGeneration) {
      this.view.fitPreviewFrame();
      this.view.setPageCount(pageCountText(this.options.format, this.measuredTextHeightPixels()));
    }
  }

  /** The preview's text height at full size. Measured once its pictures have
   * loaded: a picture still loading has no height, and the count then jumped
   * by a page as it arrived. */
  private measuredTextHeightPixels(): number {
    // Layout sizes, which a transform does not change: the page at full size.
    const previewArticle = this.view.previewArticle();
    return previewArticle !== null ? previewTextHeightPixels(previewArticle.offsetWidth, previewArticle.offsetHeight) : 0;
  }

  private async save(): Promise<void> {
    const content = this.latestContent;
    if (content === null || content.entries.length === 0) {
      return;
    }
    const saveLabel = exportFormatOffer(this.options.format).saveLabel;
    this.view.setSaving(true, saveLabel);
    this.view.setStatus("");
    try {
      const saveRequest = await exportSaveRequest(content, this.options.format, this.rasteriseDiagram);
      const saveOutcome = await this.environment.saveDocument(saveRequest);
      if (saveOutcome.outcome === "saved") {
        // Kept only once a file was really saved: choices tried and
        // cancelled are not what the person settled on.
        const { format, includesPictures, includesAiNote } = this.options;
        this.environment.rememberChoices({ format, includesPictures, includesAiNote });
        this.view.hide();
        this.environment.announceSaved(saveOutcome.fileName);
      }
    } catch {
      this.view.setStatus(SAVE_FAILED_STATUS);
    } finally {
      this.view.setSaving(false, saveLabel);
    }
  }
}
