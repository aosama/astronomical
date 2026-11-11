import type { ExportContent, ExportEntry } from "./ExportContentBuilder";
import { PRINTED_MARGIN_INCHES, textWidthInches } from "./PaperSize";
import type { PaperSize } from "./PaperSize";
import { KEPT_TOGETHER_CLASS, markupKeepingHeadingsWithTheirBlocks } from "./HeadingKeeper";

/**
 * The exported document as HTML, and the one stylesheet that dresses it.
 *
 * All three formats start here. The web page is this markup in a file; the
 * PDF is the same file printed by the host; the Word document is the article
 * and the shared rules read by the Word writer. One source is what keeps the
 * preview honest: what the sheet shows is what every format is made from.
 */

/** Rules every format reads. Code is set at 8.5 points so an 80-character
 * line, the width models draw text diagrams to, fits the 6.5-inch column
 * without wrapping; at 9.5 points the box diagrams broke across lines.
 * Sizes are in points because the page is paper
 * first: a point is the same size in a browser's print layout and in Word.
 * They are keyed on the article rather than the body so the export sheet's
 * preview, which has no body of its own, reads them unchanged. */
export const SHARED_EXPORT_RULES = `
.export-article { color: #1c1b19; font-size: 11pt; line-height: 1.5; }
h1.export-title { font-size: 22pt; line-height: 1.2; margin: 0 0 2pt; }
p.export-date { color: #6a6761; margin: 0 0 18pt; }
.export-question { color: #4a4843; border-left: 3px solid #c9c6bf; padding-left: 10pt; margin: 16pt 0 8pt; }
.export-answer { margin: 0 0 14pt; }
h1 { font-size: 17pt; }
h2 { font-size: 14pt; }
h3 { font-size: 12pt; }
p { margin: 0 0 8pt; }
code { font-family: "Courier New", monospace; background: #f1f0ec; }
pre { background: #f1f0ec; padding: 8pt; font-size: 8.5pt; white-space: pre-wrap; }
pre code { background: none; }
table { border-collapse: collapse; margin: 8pt 0; }
th, td { border: 1px solid #a9a69f; padding: 4pt 8pt; text-align: left; vertical-align: top; }
th { background: #f1f0ec; }
li.export-task { list-style: none; }
blockquote { color: #4a4843; border-left: 3px solid #c9c6bf; margin: 8pt 0; padding-left: 10pt; }
a { color: #5a33b8; }
p.export-note { color: #6a6761; font-size: 9pt; margin-top: 24pt; }
`;

/** WebKit lays a printed page out 1.25 times wider than the paper's text
 * column and shrinks the drawing to fit, so a CSS pixel prints as 0.8 of a
 * point where it should be 0.75. Every size came out sixteen fifteenths too
 * large: an 11-point body was 11.73 points on paper, lines wrapped sooner and
 * pages broke earlier than the preview, which draws the true sizes. Zooming
 * the printed body by fifteen sixteenths makes a point on paper a point.
 * Scaling the host's print job by the same factor was tried first and cut
 * tables and diagrams across pages: the engine and the print system then
 * disagreed on where a page ends.
 *
 * Only the page the host prints as the PDF is zoomed. A saved web page may
 * be printed from any browser, and Chrome already prints a point as a point:
 * zoomed, a 20-point line came out of Chrome at 18.75. */
export const PRINT_TRUE_SIZE_ZOOM = 15 / 16;
const HOST_PRINTED_PAGE_RULES = `@media print { body { zoom: ${PRINT_TRUE_SIZE_ZOOM}; } }`;

/** What a printed page never cuts in two. The page estimate reads the same
 * list, so the count the sheet shows follows the breaks the printer makes. */
export const WHOLE_BLOCK_SELECTOR = `pre, table, figure, img, .${KEPT_TOGETHER_CLASS}`;

/** Rules only a browser reads, on screen or when the host prints the PDF.
 * Word is given neither a font (it keeps its own default, which exists on
 * the machine that opens the file) nor page-break hints it would not follow. */
export const PAGE_ONLY_EXPORT_RULES = `
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; background: #ffffff; }
.export-article { max-width: 6.5in; margin: 0 auto; padding: 0.6in 0.5in; font-family: -apple-system, "Helvetica Neue", "Segoe UI", Arial, sans-serif; }
@media print { .export-article { max-width: none; margin: 0; padding: 0; } }
pre, code { font-family: ui-monospace, Menlo, Consolas, "Courier New", monospace; }
h1, h2, h3, h4 { break-after: avoid; }
${WHOLE_BLOCK_SELECTOR} { break-inside: avoid; }
figure { margin: 10pt 0; }
img { max-width: 100%; height: auto; }
.export-picture img { max-width: 4.5in; border-radius: 4pt; }
.export-diagram svg { max-width: 100%; max-height: 4.8in; width: auto; height: auto; }
.export-diagram--dark { background: #1f1f1d; padding: 10pt; border-radius: 4pt; }
math[display="block"] { margin: 8pt 0; }
p.export-note { border-top: 1px solid #c9c6bf; padding-top: 6pt; }
`;

/** A saved page may be opened anywhere, so it is told to load nothing but
 * what it carries: no script, no remote picture, no font. A model that slipped
 * a remote address past the cleaner still could not make the file call out. */
const STANDALONE_PAGE_POLICY = "default-src 'none'; img-src data:; style-src 'unsafe-inline'";

export function escapeExportText(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/** How an entry's body becomes markup: as it is for Word, or with headings
 * grouped with their blocks for a page a browser breaks. */
type EntryBodyMarkup = (body: HTMLElement) => string;

function plainBodyMarkup(body: HTMLElement): string {
  return body.innerHTML;
}

function articleHtml(content: ExportContent, bodyMarkup: EntryBodyMarkup): string {
  const noteHtml = content.aiNote !== null ? `<p class="export-note">${escapeExportText(content.aiNote)}</p>` : "";
  let entriesHtml = "";
  for (const entry of content.entries) {
    entriesHtml += `<div class="${entryClass(entry)}">${bodyMarkup(entry.body)}</div>`;
  }
  return (
    `<article class="export-article">` +
    `<h1 class="export-title">${escapeExportText(content.title)}</h1>` +
    `<p class="export-date">${escapeExportText(content.dateText)}</p>` +
    entriesHtml +
    noteHtml +
    `</article>`
  );
}

function entryClass(entry: ExportEntry): string {
  return entry.isQuestion ? "export-question" : "export-answer";
}

/** The document's markup without a page around it, as the Word writer reads
 * it. Word keeps a heading with its text through its own heading styles. */
export function exportArticleHtml(content: ExportContent): string {
  return articleHtml(content, plainBodyMarkup);
}

/** The article for a page a browser lays out and breaks: the preview, the
 * saved web page and the page printed as the PDF. */
function pagedArticleHtml(content: ExportContent): string {
  return articleHtml(content, markupKeepingHeadingsWithTheirBlocks);
}

/** The preview draws the page at this fraction of its size. */
export const PREVIEW_SCALE = 0.3;
/** The element whose height the sheet sets to the scaled page's height. */
export const PREVIEW_FRAME_CLASS = "export-preview-frame";

/**
 * The document as the export sheet previews it: a sheet of the Mac's paper
 * with the printed page's one-inch margins and text column, so code wraps and
 * pages break where they will on paper.
 *
 * The page is laid out at full size and shrunk with a transform. CSS zoom
 * was tried first and looked wrong: WebKit holds zoomed text to its minimum
 * font size, so 8.5-point code drawn at a third of its size came out larger
 * than scaled, wrapped where the paper did not, and a five-page PDF was
 * previewed as fourteen. A transform scales the finished drawing instead.
 * Its layout keeps the full height, so the frame around it is given the
 * scaled height once the page is drawn (`ExportSheetView.fitPreviewFrame`).
 */
export function exportPreviewMarkup(content: ExportContent, paper: PaperSize): string {
  const previewPageRules =
    `:host{display:block}` +
    `.${PREVIEW_FRAME_CLASS}{overflow:hidden}` +
    `.export-preview-scaler{width:${paper.widthInches}in;transform:scale(${PREVIEW_SCALE});transform-origin:0 0}` +
    `.export-article{box-sizing:content-box;max-width:none;width:${textWidthInches(paper)}in;margin:0;padding:${PRINTED_MARGIN_INCHES}in}`;
  return (
    `<style>${SHARED_EXPORT_RULES}${PAGE_ONLY_EXPORT_RULES}${previewPageRules}</style>` +
    `<div class="${PREVIEW_FRAME_CLASS}"><div class="export-preview-scaler">${pagedArticleHtml(content)}</div></div>`
  );
}

function standalonePage(content: ExportContent, extraRules: string): string {
  return (
    `<!DOCTYPE html><html><head><meta charset="utf-8">` +
    `<meta name="viewport" content="width=device-width, initial-scale=1">` +
    `<meta http-equiv="Content-Security-Policy" content="${STANDALONE_PAGE_POLICY}">` +
    `<title>${escapeExportText(content.title)}</title>` +
    `<style>${SHARED_EXPORT_RULES}${PAGE_ONLY_EXPORT_RULES}${extraRules}</style>` +
    `</head><body>${pagedArticleHtml(content)}</body></html>`
  );
}

/** The whole file for a saved web page. */
export function standaloneExportPage(content: ExportContent): string {
  return standalonePage(content, "");
}

/** The page the host prints as the PDF: the saved web page with the one
 * rule that makes the host's print engine draw sizes as written. */
export function hostPrintedExportPage(content: ExportContent): string {
  return standalonePage(content, HOST_PRINTED_PAGE_RULES);
}
