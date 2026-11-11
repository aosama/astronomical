import { hostPrintedExportPage, standaloneExportPage } from "./ExportArticle";
import type { ExportContent } from "./ExportContentBuilder";
import { suggestedExportName } from "./ExportContentBuilder";
import type { ExportSaveRequest } from "./ExportEnvironment";
import { ExportFormat } from "./ExportFormat";
import type { DiagramRasteriser } from "./DiagramRasteriser";
import type { PaperSize } from "./PaperSize";
import { writeWordDocument } from "./WordDocumentWriter";

/** Turns finished content into the file the host saves. The web page and the
 * PDF are the same standalone page, the PDF's with one print rule more; only
 * Word needs a file built here. */
export async function exportSaveRequest(content: ExportContent, format: ExportFormat, rasteriseDiagram: DiagramRasteriser, paper: PaperSize): Promise<ExportSaveRequest> {
  const suggestedName = suggestedExportName(content.title);
  if (format === ExportFormat.WORD) {
    return { format, suggestedName, html: null, base64: await writeWordDocument(content, rasteriseDiagram, paper) };
  }
  const html = format === ExportFormat.PDF ? hostPrintedExportPage(content) : standaloneExportPage(content);
  return { format, suggestedName, html, base64: null };
}
